#------------------------------------------------------------------------------
# Import
#------------------------------------------------------------------------------

print('Import...')

import os
import glob
import json
import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import tensorflow as tf
print('tf:', tf.__version__)
import transformers as tr
print('tr:', tr.__version__)
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--data_dir', default='/kaggle/input/AI4Code', type=str, help='Data dir')
parser.add_argument('--out_dir', default='ai4code-tfrec', type=str, help='Out dir to save TFRecords')
parser.add_argument('--model_dir_or_name', default='Salesforce/codet5-base', type=str, help='Model directory or name')
parser.add_argument('--n_splits', default=5, type=int, help='Number of folds (splits)')
parser.add_argument('--dim', default=1024, type=int, help='Sequence length')
parser.add_argument('--step', default=1, type=int, help='Step to generate length range')
args = parser.parse_args()
# args = parser.parse_args([]) # to run in a notebook cell with default values

print('Settings')
for a in sorted([a for a in vars(args) if '__' not in a]): print('%-20s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
# Definitions
#------------------------------------------------------------------------------

def clean(s):
    """
    Clean spaces and new lines
    """
    s = s.replace('\\n', '\n')
    s = ' '.join(s.split())
    return s

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_len_pairs(dim=1024, step=1):
    """
    Create list of length pairs where the 1st element is 
    a query length and the second one is a context length.
    We will iterate over this pairs in order to pack as much 
    useful information as possible trying to avoid padding
    
    Parameters:
    dim : int
        Maximum sequence length
    step : int
        Step to generate length range
    
    Returns:
    len_pairs : list
        List of length pairs
    """
    cont_list_1 = list(range(2, dim-1, step))
    q_list_1 = [128] * len(cont_list_1)
    len_pairs_1 = list(zip(q_list_1, cont_list_1))
    
    q_list_2 = list(range(128, dim-1, step))[1:]
    cont_list_2 = [dim] * len(q_list_2)
    len_pairs_2 = list(zip(q_list_2, cont_list_2))
    
    len_pairs = len_pairs_1 + len_pairs_2
    return len_pairs

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_cv_split(data_dir, n_splits):
    """
    Create group split based on ancestor_id
    
    Parameters:
    data_dir : str
        Data directory
    n_splits : int
        Number of folds (splits)

    Returns:
    train_df : pd.DataFrame
        DataFrame with a corresponding fold_id column
    """
    # files (139256)
    files = sorted(glob.glob(os.path.join(data_dir, 'train', '*.json')))
    # dataframe
    train_df = pd.DataFrame()
    train_df['file'] = files
    train_df['id'] = train_df['file'].map(lambda x: os.path.basename(x).split('.')[0])
    # ancestors
    ancestor_df = pd.read_csv(os.path.join(data_dir, 'train_ancestors.csv'))
    train_df = pd.merge(train_df, ancestor_df[['id', 'ancestor_id']], on='id', how='left')
    # order
    order_df = pd.read_csv(os.path.join(data_dir, 'train_orders.csv'))
    order_df['cell_order'] = order_df['cell_order'].str.split()    
    train_df = pd.merge(train_df, order_df, on='id', how='left')
    # split    
    train_df['fold_id'] = 0
    train_df = train_df.reset_index(drop=True)
    kf = GroupKFold(n_splits=n_splits)
    for fold_id, (train_index, val_index) in enumerate(kf.split(train_df, groups=train_df['ancestor_id'])):
            train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id
    train_df = train_df.sample(frac=1.0, random_state=34)
    train_df = train_df.reset_index(drop=True)
    # ret
    return train_df

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def sample_single_nb(row, len_pairs, max_len=1024):
    """
    Create training examples for a single notebook

    CONCEPT:
    Point-wise ranking approach. Label is a float in range 0-1.
    Context for each markdown cell is dynamic, aiming to get as many meaningful tokens as possible. 
    Specifically we reserve a maximum 128 tokens for markdown cell (query) and the rest goes for context. 
    Context pool contains all code cells followed by all markdown cells except the query. 
    Code cells in the context pool are always ordered, markdown cells are shuffled for each example. 
    Sampling from a context pool is based on evenly distributed amounts of tokens taken from each cell. 
    Specifically we start from 1 token per cell. If the total amount of sampled tokens is less than max length 
    we try to take 2 tokens from each cell, and so on. Eventually if we cannot collect enough tokens padding is applied.
    
    Parameters:
    row : pd.Series
        Row from a training DataFrame corresponding to a single notebook
    len_pairs : list
        List of length pairs to select from
    max_len : int
        Maximum sequence length

    Returns:
    nb_examples : list
        Examples for a given notebook (lists of encoded tokens)
    labels : list
        Labels for each example (float in range 0-1)
    ids : list
        Example ids
    """
    # read notebook
    nb_df = pd.read_json(row['file'])
    nb_df = nb_df.rename_axis('cell_id')
    nb_ord_df = nb_df.loc[row['cell_order'], :]
    nb_ord_df['rank'] = range(len(nb_ord_df))
    nb_ord_df['rank_rel'] = nb_ord_df['rank'] / len(nb_ord_df)
    nb_ord_df = nb_ord_df.reset_index()
    nb_ord_df['global_id'] = row['id'] + '_' + nb_ord_df['cell_id']
    # clean both md and code
    nb_ord_df['source'] = nb_ord_df['source'].map(clean)
    # create separate lists of md and code
    md_cells = nb_ord_df.loc[nb_ord_df['cell_type'] == 'markdown', 'source'].tolist()
    code_cells = nb_ord_df.loc[nb_ord_df['cell_type'] == 'code', 'source'].tolist()
    labels = nb_ord_df.loc[nb_ord_df['cell_type'] == 'markdown', 'rank_rel'].tolist()
    ids = nb_ord_df.loc[nb_ord_df['cell_type'] == 'markdown', 'global_id'].tolist()
    # encode
    md_cells_encoded = tokenizer.batch_encode_plus(
        md_cells,
        add_special_tokens=False,
        max_length=None,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors=None,)['input_ids']
    code_cells_encoded = tokenizer.batch_encode_plus(
        code_cells,
        add_special_tokens=False,
        max_length=None,
        padding=False,
        truncation=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors=None,)['input_ids']

    # See CONCEPT in docstring

    # 'md_cells_encoded' will stay ordered and we will take queries from it
    # 'md_cells_encoded_context' is a copy which we will shuffle 
    # and use as context together with the code cells
    md_cells_encoded_context = md_cells_encoded.copy()
    # all examples for a given notebook
    nb_examples = [] 
    # loop over all md-cells (queries) in a single notebook
    for md_cell in md_cells_encoded: 
        # shuffle md-cells used as a context for each example
        np.random.shuffle(md_cells_encoded_context)
        # loop over maximum lengths
        for md_max_len, code_max_len in len_pairs:        
            # single example: single md-cell (query) plus context (code-cells + md-cells)
            example_token_ids = []
            # apply truncation and add sep
            example_token_ids.extend([tokenizer.bos_token_id] + md_cell[:(md_max_len-3)] + 
                                     [tokenizer.sep_token_id, tokenizer.sep_token_id])
            # loop over context cells (code-cells + md-cells)
            for code_cell in code_cells_encoded + md_cells_encoded_context:
                # exclude current md-cell (query) from the context
                if md_cell != code_cell: 
                    example_token_ids.extend(code_cell[:(code_max_len)])
                # stop adding new context cells
                if len(example_token_ids) >= max_len: 
                    break
            # stop trying different lengths
            if len(example_token_ids) >= max_len:
                    break
        
        # truncate and EOS
        example_token_ids = example_token_ids[:(max_len-1)] + [tokenizer.sep_token_id]
        # pad (in case we have very small amount of very short cells we will need padding)
        example_token_ids += [tokenizer.pad_token_id] * (max_len-len(example_token_ids))
        # check
        assert len(example_token_ids) == max_len, 'len != max_len'
        # add example
        nb_examples.append(example_token_ids)

    # number of examples must be equal to the number of md-cells (queries)
    assert len(md_cells_encoded) == len(nb_examples), 'N examples != N md-cells'
    # ret
    return nb_examples, labels, ids

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class TFRecordProcessor(object):
    """
    Writes sharded TFRecord files
    """
    def __init__(self):
        self.n_examples = 0
    def _bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    def _process_example(self, ind, A, B, C, D):
        self.n_examples += 1
        feature = collections.OrderedDict()
        feature['image_id'] = self._bytes_feature(A[ind].encode('utf-8'))
        feature['image'] =    self._int_feature(list(B[ind]))
        feature['label_id'] = self._bytes_feature(C[ind].encode('utf-8'))
        feature['label'] =    self._float_feature([D[ind]])
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        self._writer.write(example_proto.SerializeToString())
    def write_tfrecords(self, A, B, C, D, n_shards=1, file_out='train.tfrecord'):
        n_examples_per_shard = A.shape[0] // n_shards
        n_examples_remainder = A.shape[0] %  n_shards   
        self.n_examples = 0
        for shard in range(n_shards):
            self._writer = tf.io.TFRecordWriter('%s-%05d-of-%05d' % (file_out, shard, n_shards))
            start = shard * n_examples_per_shard
            if shard == (n_shards - 1):
                end = (shard + 1) * n_examples_per_shard + n_examples_remainder
            else:
                end = (shard + 1) * n_examples_per_shard
            print('Shard %d of %d: (%d examples)' % (shard, n_shards, (end - start)))
            for i in range(start, end):
                self._process_example(i, A, B, C, D)
                print(i, end='\r')
            self._writer.close()
        return self.n_examples

#------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------

if __name__ == '__main__':

    np.random.seed(33)
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = tr.AutoTokenizer.from_pretrained(args.model_dir_or_name)
    print('Init tokenizer:', tokenizer.__class__.__name__)    
    tfrp = TFRecordProcessor()
    train_df = create_cv_split(args.data_dir, args.n_splits)
    len_pairs = create_len_pairs(args.dim, args.step)
    
    # loop over folds
    for fold_id in range(args.n_splits):        
        fold_df = train_df[train_df['fold_id'] == fold_id].copy()
        print('Select fold-specific DataFrame (fold %d) with shape: %s' % (fold_id, str(fold_df.shape)))

        fold_examples = []
        fold_labels = []
        fold_ids = []
    
        # loop over rows (notebooks) in a fold
        for counter, (_, row) in enumerate(fold_df.iterrows()):
            nb_examples, labels, ids = sample_single_nb(row, len_pairs, max_len=args.dim)
            fold_examples.extend(nb_examples)
            fold_labels.extend(labels)
            fold_ids.extend(ids)
            print(counter, end='\r')
    
        # create numpy arrays
        X = np.array(fold_examples)
        print(X.shape)
        y = np.array(fold_labels)
        ids = np.array(fold_ids)
    
        print('Write TFRecords (fold %d)' % fold_id)
        n_written = tfrp.write_tfrecords(
            ids, 
            X, 
            y.astype(str), 
            y, 
            n_shards=1, 
            file_out=os.path.join(args.out_dir, 'fold.%d.tfrecord' % fold_id))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
