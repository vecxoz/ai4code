#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

print('Import')

import os
import sys
import glob
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import tensorflow as tf
print('tf:', tf.__version__)
import transformers as tr
print('tr:', tr.__version__)
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--data_dir', default='/kaggle/input/AI4Code', type=str, help='Data dir')
parser.add_argument('--weight_dir', default='/kaggle/input/ai4code-weights', type=str, help='Dir with a trained weights')
parser.add_argument('--model_dir_or_name', default='/kaggle/input/model-codet5base', type=str, help='Model directory or name')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--dim', default=1024, type=int, help='Sequence length')
parser.add_argument('--step', default=1, type=int, help='Step to generate length range')
args = parser.parse_args()
# args = parser.parse_args([]) # to run in a notebook cell with default values

print('Settings')
for a in sorted([a for a in vars(args) if '__' not in a]): print('%-20s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def create_df(data_dir):
    """
    Create DataFrame listing all files to predict
    """
    files = sorted(glob.glob(os.path.join(data_dir, 'test', '*.json')))
    df = pd.DataFrame()
    df['file'] = files
    df['id'] = df['file'].map(lambda x: os.path.basename(x).split('.')[0])
    return df

#------------------------------------------------------------------------------
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

def sample_single_nb(row, len_pairs, max_len=1024):
    """
    Create training examples for a single notebook.
    
    This is a slightly modified inference version
    of the `create_data.sample_single_nb` function.
    Please see corresponding docstring for concept and other details.
    """
    # read notebook
    nb_df = pd.read_json(row['file'])
    nb_df = nb_df.rename_axis('cell_id')
    nb_df = nb_df.reset_index()
    nb_df['global_id'] = row['id'] + '_' + nb_df['cell_id']
    # clean both md and code
    nb_df['source'] = nb_df['source'].map(clean)
    # create separate lists of md and code
    md_cells = nb_df.loc[nb_df['cell_type'] == 'markdown', 'source'].tolist()
    code_cells = nb_df.loc[nb_df['cell_type'] == 'code', 'source'].tolist()
    ids = nb_df.loc[nb_df['cell_type'] == 'markdown', 'global_id'].tolist()
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
    
    # create examples based on query and context
    md_cells_encoded_context = md_cells_encoded.copy()
    nb_examples = []
    for md_cell in md_cells_encoded: 
        np.random.shuffle(md_cells_encoded_context)    
        for md_max_len, code_max_len in len_pairs:
            example_token_ids = []
            example_token_ids.extend([tokenizer.bos_token_id] + md_cell[:(md_max_len-3)] + 
                                     [tokenizer.sep_token_id, tokenizer.sep_token_id])
            for code_cell in code_cells_encoded + md_cells_encoded_context:
                if md_cell != code_cell:
                    example_token_ids.extend(code_cell[:(code_max_len)])
                if len(example_token_ids) >= max_len:
                    break
            if len(example_token_ids) >= max_len:
                    break

        # truncate and EOS
        example_token_ids = example_token_ids[:(max_len-1)] + [tokenizer.sep_token_id]    
        # pad
        example_token_ids += [tokenizer.pad_token_id] * (max_len-len(example_token_ids))
        # check
        assert len(example_token_ids) == max_len, 'len != max_len'
        # add example
        nb_examples.append(np.array(example_token_ids, dtype=np.int32))

    # check
    assert len(md_cells_encoded) == len(nb_examples), 'N examples != N md-cells'
    # ret
    return nb_examples, None, ids

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
    model.compile(optimizer=tf.keras.optimizers.Adam(0.000_025), 
                  loss=tf.keras.losses.MeanAbsoluteError(), 
                  metrics=['mse'])
    if print_summary:
        model.summary()
    return model

#------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------

if __name__ == '__main__':

    # Data

    np.random.seed(3333)
    tokenizer = tr.AutoTokenizer.from_pretrained(args.model_dir_or_name)
    print('Init tokenizer:', tokenizer.__class__.__name__)    
    print('Create data')
    test_df = create_df(args.data_dir)
    print('test_df.shape:', test_df.shape) # (4, 2)
    len_pairs = create_len_pairs(args.dim, args.step)
    
    test_examples = []
    test_ids = []    
    # loop over rows (notebooks)
    for counter, (_, row) in enumerate(test_df.iterrows()):
        nb_examples, labels, ids = sample_single_nb(row, len_pairs, max_len=args.dim)
        test_examples.extend(nb_examples)
        test_ids.extend(ids)
        print(counter, end='\r')
    
    # create numpy arrays
    X = np.vstack(test_examples)
    print(X.shape)
    ids = np.array(test_ids)

    # Model
    
    print('Init model')    
    model = init_model(from_pretrained=False)
    # collect trained weights
    mm = sorted(glob.glob(os.path.join(args.weight_dir, '*.h5')))
    
    preds = []
    for m in mm:
        model.load_weights(m)
        print('Predict:', m)
        y_pred = model.predict(X, batch_size=args.batch_size, verbose=1)
        preds.append(y_pred.ravel().copy())
        print(y_pred[:5].ravel())
    
    # Ensemble
    
    y_pred = 0.6 * preds[0] + 0.4 * preds[1]
    
    # Process predictions
    
    pred_df = pd.DataFrame()
    pred_df['global_id'] = ids
    pred_df['nb_id'] = pred_df['global_id'].map(lambda x: x.split('_')[0])
    pred_df['cell_id'] = pred_df['global_id'].map(lambda x: x.split('_')[1])
    pred_df['y_pred'] = y_pred
    
    final_nb_ids = []
    final_cell_ids_pred = []
    
    for counter, (nb_id, group_df) in enumerate(pred_df.groupby('nb_id')): 
        final_nb_ids.append(nb_id)
        file = test_df[test_df['id'] == nb_id]['file'].values[0]
        nb_df = pd.read_json(file)
        nb_df = nb_df.rename_axis('cell_id')
        nb_df = nb_df.reset_index()    
        code_df = nb_df[nb_df['cell_type'] == 'code'].copy()
        code_df['y_pred'] = np.array(range(len(code_df))) / len(code_df)
        md_df = nb_df[nb_df['cell_type'] == 'markdown'].copy()
        md_df = pd.merge(md_df, group_df[['cell_id', 'y_pred']], on='cell_id', how='left')
        all_df = pd.concat([code_df, md_df])
        all_df = all_df.sort_values('y_pred')    
        all_cell_ids_ordered = all_df['cell_id'].values.tolist()
        assert len(all_cell_ids_ordered) == len(nb_df), 'len inequal'
        final_cell_ids_pred.append(all_cell_ids_ordered)
        print(counter, end='\r')
    
    subm_df = pd.DataFrame()
    subm_df['id'] = final_nb_ids
    subm_df['cell_order'] = final_cell_ids_pred
    subm_df['cell_order'] = subm_df['cell_order'].map(lambda x: ' '.join(x))
    subm_df.to_csv('submission.csv', index=False)
    subm_df.head()

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
