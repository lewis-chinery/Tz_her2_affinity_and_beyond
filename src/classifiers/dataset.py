import numpy as np

from src.utils import get_ordered_AA_one_letter_codes


def get_X_y(df, seq_col="seq", label_col="label", one_dim=False):
    '''
    '''
    X_strs = df[seq_col].tolist()
    if one_dim:
        X = [seq_to_1D_onehot(seq) for seq in X_strs]
    else:
        X = [seq_to_2D_onehot(seq) for seq in X_strs]
    y = df[label_col].tolist()
    return np.asarray(X), np.asarray(y)


def AA_to_onehot(AA):
    '''
    '''
    all_AAs = get_ordered_AA_one_letter_codes()
    onehot = np.zeros(20)
    onehot[all_AAs.index(AA)] = 1
    return onehot


def seq_to_1D_onehot(seq):
    '''
    '''
    return np.array(seq_to_2D_onehot(seq)).flatten().tolist()


def seq_to_2D_onehot(seq):
    '''
    '''
    onehot_seq = []
    for AA in seq:
        onehot_AA = AA_to_onehot(AA)
        onehot_seq.append(onehot_AA.tolist())
    return onehot_seq
