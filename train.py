#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# For Kaggle TPU VM
# !pip install transformers

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Import')

import os
import sys
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
print('tf:', tf.__version__)
import transformers as tr
print('tr:', tr.__version__)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()

parser.add_argument('--model_dir_or_name', default='Salesforce/codet5-base', type=str, help='Model directory or name')
parser.add_argument('--data_tfrec_dir',    default='/kaggle/input/ai4code-tfrec', type=str,   help='Data directory with TFRecord files')
parser.add_argument('--tpu',               default='local',     type=str,   help='TPU GRPC/name, or local, or None')
parser.add_argument('--job',               default='train',     type=str,   help='Job to perform')
parser.add_argument('--metric_name',       default='mse',       type=str,   help='Metric name')
parser.add_argument('--monitor',           default='val_loss',  type=str,   help='Value to monitor')
parser.add_argument('--monitor_mode',      default='min',       type=str,   help='Monitor mode')
parser.add_argument('--n_folds',           default=5,           type=int,   help='Number of folds')
parser.add_argument('--initial_fold',      default=0,           type=int,   help='Initial fold (from 0)')
parser.add_argument('--final_fold',        default=2,           type=int,   help='Final fold. To train single fold set `initial_fold + 1`')
parser.add_argument('--dim',               default=1024,        type=int,   help='Max seq len')
parser.add_argument('--n_examples_total',  default=2_166_064,   type=int,   help='Total number of training examples')
parser.add_argument('--n_epochs',          default=50,          type=int,   help='Number of epochs to train (hard limit). Early stop is applied')
parser.add_argument('--batch_size',        default=64,          type=int,   help='Batch size')
parser.add_argument('--lr',                default=0.000_025,   type=float, help='Learning rate')

args = parser.parse_args()
# args = parser.parse_args([]) # to run in a notebook cell with default values

# Number of sub-train examples i.e. all folds except one (e.g. 4/5 of full train)
args.n_examples_train = args.n_examples_total - (args.n_examples_total // args.n_folds)
print('Settings')
for a in sorted([a for a in vars(args) if '__' not in a]): print('%-20s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_accel(tpu=None):
    """
    Seamlessly init any accelerator: CPU, GPU, multi-GPU, TPU

    Parameters:
    tpu : str or None
        TPU node GRPC or name 
        E.g. 'grpc://10.70.50.202:8470' or 'node-1' or 'local'

    Returns:
    strategy : 
        Strategy
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    except tf.errors.NotFoundError:
        strategy = tf.distribute.MirroredStrategy()
        print('TPU was not found')
    print('Num replicas:', strategy.num_replicas_in_sync)
    return strategy

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_tfdata(files_glob, deterministic=True, batch_size=32, auto=-1, 
                parse_example=None, repeat=False, buffer_size=None, 
                cache=False, drop_remainder=False):
    """
    Creates tf.data.TFRecordDataset with appropriate parameters

    Parameters:
    files_glob : str
        Glob wildcard for TFRecord files
    deterministic : bool
    batch_size : int
    auto : int
        Number of parallel reads/calls. -1 means automatic
    parse_example : callable
        Processing function
    repeat : bool
        Whether to repeat dataset
    buffer_size : int or None
        Shuffle buffer size. None means do not shuffle.
    cache : bool
        Whether to cache data
    drop_remainder : bool
        Whether to drop remainder

    Returns:
    ds : 
        Initialized dataset
    """
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    files = tf.data.Dataset.list_files(files_glob, shuffle=not deterministic).with_options(options)
    print('N tfrec files:', len(files))
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=auto)
    ds = ds.with_options(options)
    ds = ds.map(parse_example, num_parallel_calls=auto)
    if repeat:
        ds = ds.repeat()
    if buffer_size:
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(auto)
    if cache:
        ds = ds.cache()
    return ds

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class KeepLastCKPT(tf.keras.callbacks.Callback):
    """
    Sorts and removes all ckpt except the last.
    Parameters:
    wildcard : str
        Wildcard for weight file names
    """
    #
    def __init__(self, wildcard):
        super(KeepLastCKPT, self).__init__()
        self.wildcard = wildcard
    #
    def on_epoch_begin(self, epoch, logs=None):
        files = sorted(tf.io.gfile.glob(self.wildcard))
        if len(files):
            for file in files[:-1]:
                tf.io.gfile.remove(file)
            print('Kept ckpt: %s' % files[-1])
        else:
            print('No ckpt to keep')
    #
    def on_train_end(self, logs=None):
        files = sorted(tf.io.gfile.glob(self.wildcard))
        if len(files):
            for file in files[:-1]:
                tf.io.gfile.remove(file)
            print('\nKept ckpt (final): %s' % files[-1])
        else:
            print('\nNo ckpt to keep (final)')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

feature_description = {
    'image':    tf.io.FixedLenFeature([args.dim], tf.int64),
    'label':    tf.io.FixedLenFeature([], tf.float32),
}


def parse_example(example_proto):
    """
    Parse TFRec example
    """
    d = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.cast(d['image'], tf.int32)
    label = tf.cast(d['label'], tf.float32)
    return image, label

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def init_model(print_summary=True, from_pretrained=True):
    """
    Init model with pretrained or random weights

    Parameters:
    print_summary : bool
        Whether to print model summary
    from_pretrained : bool
        Init model with petrained/random weights

    Returns:
    model : 
        Initialized model
    """
    if from_pretrained:
        transformer = tr.TFT5EncoderModel.from_pretrained(
                        args.model_dir_or_name, from_pt=True)
    else:
        config = tr.AutoConfig.from_pretrained(args.model_dir_or_name)
        transformer = tr.TFT5EncoderModel.from_config(config)
    input_ids = tf.keras.layers.Input(shape=(args.dim,), dtype=tf.int32)
    sequence_output = transformer(input_ids)[0] # (batch, len, hidden)
    cls_token = sequence_output[:, 0, :] # (batch, hidden)
    out = tf.keras.layers.Dense(1, activation='linear')(cls_token)
    model = tf.keras.models.Model(inputs=input_ids, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), 
                  loss=tf.keras.losses.MeanAbsoluteError(), 
                  metrics=[args.metric_name])
    if print_summary:
        model.summary()
    return model

#------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------

for fold_id in range(args.initial_fold, args.final_fold):
    print('\n*****')
    print('Fold:', fold_id)
    print('*****\n')
    print('Clear session...')
    tf.keras.backend.clear_session()
    print('FULL BATCH SHAPE: %d x %d' % (args.batch_size, args.dim,))
    print('LR: %.8f' % args.lr)
    print('Init TPU')
    strategy = init_accel(args.tpu)
    #----------------------------------------------------------------------
    # Globs
    all_fold_ids = np.array(range(args.n_folds))
    train_fold_ids = all_fold_ids[all_fold_ids != fold_id]
    train_glob = os.path.join(args.data_tfrec_dir, ('fold.[' + '%d'*(args.n_folds-1) + '].tfrecord*') % tuple(train_fold_ids))
    val_glob   = os.path.join(args.data_tfrec_dir, 'fold.[%d].tfrecord*' % fold_id)
    print('TRAIN GLOB:', train_glob)
    print('VAL   GLOB:', val_glob)
    #----------------------------------------------------------------------
    print('Init datasets')
    train_ds = init_tfdata(train_glob, 
                           deterministic=False,  
                           batch_size=args.batch_size, 
                           auto=-1,
                           parse_example=parse_example, 
                           repeat=True,
                           buffer_size=2048, 
                           drop_remainder=False,
                           cache=False)
    val_ds = init_tfdata(val_glob, 
                         deterministic=True,  
                         batch_size=args.batch_size * 2, 
                         auto=-1,
                         parse_example=parse_example,
                         repeat=False,  
                         buffer_size=None,
                         drop_remainder=False,
                         cache=False)
    #----------------------------------------------------------------------
    print('Init model')
    with strategy.scope():
        model = init_model(print_summary=True, from_pretrained='train' in args.job)
    #----------------------------------------------------------------------
    print('Init callbacks')
    call_ckpt = tf.keras.callbacks.ModelCheckpoint('model-f%d-e{epoch:03d}-{val_loss:.4f}-{val_%s:.4f}.h5' % (fold_id, args.metric_name),
                                                   monitor=args.monitor,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode=args.monitor_mode,
                                                   verbose=1)
    call_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=args.monitor,
                                                          factor=0.5, 
                                                          patience=2, 
                                                          min_delta=1e-4,
                                                          min_lr=1e-8,
                                                          verbose=1,
                                                          mode=args.monitor_mode)
    call_early_stop = tf.keras.callbacks.EarlyStopping(monitor=args.monitor,
                                                       patience=4,
                                                       min_delta=1e-4,
                                                       mode=args.monitor_mode,
                                                       verbose=1)
    call_keep_last = KeepLastCKPT(wildcard='model-f%d-e*.h5' % fold_id)
    #----------------------------------------------------------------------
    if 'train' in args.job:
        print('Fit (fold %d)' % fold_id)
        h = model.fit(
            train_ds,
            steps_per_epoch=args.n_examples_train // args.batch_size,
            epochs=args.n_epochs,
            initial_epoch=0,
            validation_data=val_ds,
            callbacks=[call_ckpt,
                       call_reduce_lr,
                       call_early_stop,
                       call_keep_last,])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------




