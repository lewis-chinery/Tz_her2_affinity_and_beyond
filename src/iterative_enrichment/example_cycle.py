import pandas as pd
from src.classifiers.dataset import get_X_y
from sklearn.metrics import precision_recall_curve


def reduce_df_class_imbalance_to_target(df, target_class_imbalance, random_state=42,
                                        label_col="label"):
    '''
    Reduce the class imbalance of a dataframe to a target class imbalance.

    :param df: dataframe
    :param target_class_imbalance: target class imbalance, float between 0 and 1
    :param random_state: random state, int
    :param label_col: name of label column, str, contains 0s and 1s
    :return: dataframe with reduced class imbalance
    '''
    num_neg = df[df[label_col]==0].shape[0]
    num_pos = df[df[label_col]==1].shape[0]
    target_num_pos = int(target_class_imbalance/(1-target_class_imbalance)*num_neg)
    assert target_num_pos <= num_pos, "Target class imbalance is higher than original class imbalance"

    # randomly sample target_num_pos from positive examples
    df_pos = df[df[label_col]==1]
    df_neg = df[df[label_col]==0]
    df_pos = df_pos.sample(target_num_pos, random_state=random_state)
    df = pd.concat([df_pos, df_neg])
    return df


def sample_seqs(df, num_sequences_to_sample, df_experimentally_tested, CNN=None, cutoff=0, random_state=None):
    '''
    Sample sequences from a dataframe

    :param df: dataframe to sample from
    :param num_sequences_to_sample: number of sequences to sample
    :param df_experimentally_tested: dataframe containing previously sampled sequences (to avoid sampling duplicates)
    :param CNN: trained CNN to use for screening sequences
    :param cutoff: cutoff for CNN
    :param random_state: random state
    :return: dataframe containing sampled sequences
    '''
    df_sampled_sequences = pd.DataFrame(columns=df.columns)
    while len(df_sampled_sequences) < num_sequences_to_sample:
        df_sample = df.sample(1, random_state=random_state)
        sampled_sequence = df_sample["seq"].values[0]
        if (sampled_sequence not in df_experimentally_tested["seq"]) & \
            screen_sampled_seq_using_trained_cnn(df_sample, CNN, cutoff):
            df_sampled_sequences = pd.concat([df_sampled_sequences, df_sample])
    return df_sampled_sequences


def screen_sampled_seq_using_trained_cnn(df_sample, CNN, cutoff):
    '''
    Check if a sampled sequence passes CNN cutoff

    :param df_sample: dataframe containing sampled sequence (single row)
    :param CNN: trained CNN to use for screening sequences, or None
    :param cutoff: cutoff for CNN
    :return: True if sequence passes cutoff, False otherwise
    '''
    if CNN is None:
        valid_seq = True
    else:
        X_sample, y_sample = get_X_y(df_sample)
        predictions = CNN.predict(X_sample)
        valid_seq = predictions.flatten()[0] > cutoff
    return valid_seq


def get_X_y_train_val(df_experimentally_tested, train_pc=0.8, random_state=42):
    '''
    Process dataframe into X and y for training and validation

    :param df_experimentally_tested: dataframe containing all seqs for CNN training/val
    :param train_pc: fraction of data to use for training
    :param random_state: random state
    :return: X_train, y_train, X_val, y_val ndarrys
    '''
    train = df_experimentally_tested.sample(frac=train_pc, random_state=random_state)
    val   = df_experimentally_tested.drop(train.index)
    X_train, y_train = get_X_y(train)
    X_val,   y_val   = get_X_y(val)
    return X_train, y_train, X_val, y_val


def get_cutoff_for_desired_recall(y_val, predictions, desired_recall=0.8):
    '''
    Identify CNN cutoff to use for desired recall

    :param y_val: true labels
    :param predictions: predicted labels
    :param desired_recall: desired recall
    :return: cutoff
    '''
    _, recall, thresholds = precision_recall_curve(y_val, predictions.flatten())
    best_idx = 0
    smallest_diff = 1
    for idx, r in enumerate(recall):
        if abs(r-desired_recall) < smallest_diff:
            smallest_diff = abs(r-desired_recall)
            best_idx = idx
    return thresholds[best_idx]
