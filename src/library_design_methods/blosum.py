import math
import numpy as np
import pandas as pd
import blosum as bl

from src.utils import get_ordered_AA_one_letter_codes


def get_20x20_blosum_matrix(cluster_pc=45):
    '''
    Get normal 20 x 20 BLOSUM matrix
    Positive/negative scores are given to more/less likely substitutions

    :param cluster_pc: int minimum indentity of clustered sequences e.g. 45, 62 etc.
    :returns: ndarray of blosum scores
    '''
    # original BLOSUM contains extra rows etc
    matrix = bl.BLOSUM(cluster_pc)

    # only core 20 AAs
    reduced_blosum_arr = np.zeros((20,20))
    AAs = get_ordered_AA_one_letter_codes()
    for idx_row, AA_row in enumerate(AAs):
        for idx_col, AA_col in enumerate(AAs):
            reduced_blosum_arr[idx_row, idx_col] = matrix[AA_row][AA_col]

    return reduced_blosum_arr.astype(int)


def normalise_matrix_by_rows(matrix):
    '''
    Normalise a matrix over its rows

    :param matrix: ndarray 2D
    :returns: ndarray when rows num to 1
    '''
    normalised_matrix = np.zeros_like(matrix)
    for i in range(normalised_matrix.shape[0]):
        normalised_matrix[i] = matrix[i] / np.sum(matrix, axis=1)[i]

    return normalised_matrix


def get_blosum_h3_AA_probs():
    '''
    Get normalised frequency that each AA type appears in the H3 from SAbDab
    Numbers obtained 09/02/23

    :returns: dict of key, values (AA one letter code, frequency)
    '''
    # counts from SAbDab
    h3_AA_counts = {'A': 6710, 'C': 555, 'D': 6442, 'E': 1636, 'F': 3404,
                    'G': 6582, 'H': 1081, 'I': 1342, 'K': 970, 'L': 2509,
                    'M': 1218, 'N': 1658, 'P': 2090, 'Q': 767, 'R': 5196,
                    'S': 4350, 'T': 2835, 'V': 2975, 'W': 1686, 'Y': 9028}

    total = sum(list(h3_AA_counts.values()))
    h3_AA_probs = {AA: count/total for AA, count in h3_AA_counts.items()}
    return h3_AA_probs


def get_blosum_prob_matrix(blosum_matrix, h3_AA_probs, _lambda=0.25):
    '''
    Reverse blosum calculation to go from negative and positive integers to probabilities
    Using the logic and notation from the following Nature article
    https://www.nature.com/articles/nbt0804-1035

    :param blosum_matrix: ndarray 20x20 blosum matrix
    :param h3_AA_probs: dict (AA, normalised freq AA is found in H3s in SAbDab)
    :param _lambda: float scaling factor
        default changed from 0.5*math.ln(2) ~0.35 to 0.25 to downweight original AA
    :returns: blosum probabilties that can be used for sampling from
    '''
    probability_blosum_matrix = np.zeros((20,20))
    AAs = get_ordered_AA_one_letter_codes()

    for row_idx, row in enumerate(blosum_matrix):
        row_AA = AAs[row_idx]

        # reverse blosum calculation
        for col_idx, s_ab in enumerate(row):  # s_ab is blosum matrix value
            col_AA = AAs[col_idx]
            f_a = h3_AA_probs[row_AA]
            f_b = h3_AA_probs[col_AA]
            probability_blosum_matrix[row_idx][col_idx] = f_a * f_b * math.exp(_lambda * s_ab)

    # tidy up to ensure rows sum to 1
    probability_blosum_matrix = normalise_matrix_by_rows(probability_blosum_matrix)

    return probability_blosum_matrix


def get_blosum_probs_for_seq(seq, blosum_matrix, h3_AA_probs, _lambda=0.25):
    '''
    Go blosum probabilities for a given H3 amino acid sequence

    :param seq: str sequence of one letter AA codes
    :param blosum_matrix: ndarray 20x20 blosum matrix
    :param h3_AA_probs: dict (AA, normalised freq AA is found in H3s in SAbDab)
    :param _lambda: float scaling factor 
    :returns: ndarray of size len(seq) x 20 with normalised blosum probabilities
    '''
    probability_blosum_matrix = get_blosum_prob_matrix(blosum_matrix, h3_AA_probs, _lambda=_lambda)
    AAs = get_ordered_AA_one_letter_codes()

    # loop over AAs in seq and get corresponding blosum probabilities
    weights_arr = np.zeros((len(seq),20))
    for i, AA in enumerate(seq):
        AA_idx = AAs.index(AA)
        weights_arr[i] = probability_blosum_matrix[AA_idx]

    # format to match logomaker output for plotting and general consistency
    normalised_mat = pd.DataFrame(weights_arr, columns=AAs)
    normalised_mat.index.name = 'pos'

    return normalised_mat
