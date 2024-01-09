from collections import Counter
import pandas as pd
import numpy as np
import random


def get_Trastuzumab_len_10_H3_seq():
    '''
    The 10 residues of the H3 that Mason et al. and HER2-aff-large mutate
    '''
    return "WGGDGFYAMD"


def get_Trastuzumab_len_13_H3_seq():
    '''
    The 13 residues of the H3 that Shanehsazzadeh et al. mutate
    '''
    return "SRWGGDGFYAMDY"


def get_Trastuzumab_H_seq():
    '''
    Full heavy chain Fv sequence
    '''
    return "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"


def get_Trastuzumab_L_seq():
    '''
    Full light chain Fv sequence
    '''
    return "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"


def mask_seq(full_seq, seq_to_mask):
    '''
    Mask seq_to_mask in full_seq with "*"

    :param full_seq: str of full sequence
    :param seq_to_mask: str of sequence to mask (must be continuous in full_seq)
    :returns: str of full_seq with seq_to_mask masked with "*"
    '''
    return full_seq.replace(seq_to_mask, '*' * len(seq_to_mask))


def get_ordered_AA_one_letter_codes():
    '''
    Get list of amino acid one letter codes in alphabetical order
    '''
    return ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def get_edit_distance(original_seq, mutated_seq):
    '''
    Get number of mutations seq is from original
    Requires seqs to be of same length

    :param original_seq: str of AAs e.g. Trastuzumab's H3
    :param mutated_seq: str of AAs of mutated seq
    :returns: int of number of mutations
    '''
    edit_distance = 0
    for idx, original_AA in enumerate(original_seq):
        if mutated_seq[idx] != original_AA:
            edit_distance += 1  
    return edit_distance


def make_nice_df_from_seqs(seqs, original_seq):
    '''
    Make df from list of seqs with extra columns

    :param seqs: list of str of AAs
    :param original_seq: str of AAs e.g. Trastuzumab's H3
    :returns: df with columns "seq" and "edit_distance"
    '''
    df = pd.DataFrame(seqs, columns=["seq"])
    df["edit_distance"] = df.apply(lambda row: get_edit_distance(original_seq, row["seq"]), axis=1)
    return df


def join_dfs(dfs):
    '''
    Join dfs with same cols

    :param dfs: list of dataframes
    :returns: single df with all contents
    '''
    return pd.concat(dfs, join="inner")


def get_observed_frequnecies_from_list_of_seqs(seqs):
    '''
    Get observed frequencies of AAs at each position in list of seqs

    :param seqs: list of str of AAs. All seqs must be of same length
    :returns: ndarray of shape (len(seqs[0]), 20) with observed frequencies
    '''
    ordered_AAs = get_ordered_AA_one_letter_codes()
    total = len(seqs)
    probabilities = []

    for seq_position in range(len(seqs[0])):
        # get AA counts at each position
        seq_position_AAs = [seq[seq_position] for seq in seqs]
        AA_count_dict = Counter(seq_position_AAs)
        # pad counts with zeros for AAs not observed
        for AA in ordered_AAs:
            if AA in AA_count_dict.keys():
                pass
            else:
                AA_count_dict[AA] = 0
        # get probabilities
        probabilities.append([AA_count_dict[AA]/total for AA in ordered_AAs])
        
    # format to match logomaker output for plotting and general consistency
    normalised_mat = pd.DataFrame(probabilities, columns=ordered_AAs)
    normalised_mat.index.name = 'pos'
    return normalised_mat


def get_weighted_AA_choice_for_given_position(position, seq_probability_df):
    '''
    Choose new amino acid for a given position in the sequence based on some probability e.g. blosum

    :param position: int zero-indexed position in the sequence that an AA is being chosen for
    :param seq_probability_df: df e.g. output of get_blosum_probs_for_seq 
    :returns: char one letter AA code that has been selected for given position
    '''
    # all possible amino acids
    mutations = get_ordered_AA_one_letter_codes()
    # weights decided based on e.g. blosum probabilities from wild type AA at certain position
    weights = np.array(seq_probability_df)[position].tolist()
    return random.choices(mutations, weights=weights, k=1)[0]


def generate_new_seq_from_probabilities(seq_probability_df):
    '''
    Loop through original sequence and generate a new sequence of the same length
    based on some probability matrix

    :param seq_probability_df: df e.g. output of get_blosum_probs_for_seq 
    :returns: str some new sequence
    '''
    new_seq = ""
    for idx in range(seq_probability_df.shape[0]):
        new_seq += get_weighted_AA_choice_for_given_position(idx, seq_probability_df)
    return new_seq


def generate_new_seqs_from_probabilities(seq_probability_df, number_of_seqs_to_generate, target_counts=None, max_tries=None, original_seq=None):
    '''
    Generate multiple new sequences from some probability matrix

    :param seq_probability_df: df e.g. output of get_blosum_probs_for_seq 
    :param number_of_seqs_to_generate: int number of new sequences to generate
    :returns: list of str of new sequences
    '''
    new_seqs = []
    if target_counts is None:
        for _ in range(number_of_seqs_to_generate):
            new_seqs.append(generate_new_seq_from_probabilities(seq_probability_df))
    else:
        target_total = sum(target_counts.values())
        actual_counts = {k: 0 for k, v in target_counts.items()}
        max_tries = number_of_seqs_to_generate*10 if max_tries is None else max_tries
        loop_count = 0
        while (sum(actual_counts.values()) < target_total):
            try:
                loop_count += 1
                new_seq = generate_new_seq_from_probabilities(seq_probability_df)
                edit_distance = get_edit_distance(new_seq, original_seq)
                # keep the sequence if we have not already filled the edit distance bin
                if (actual_counts[edit_distance] < target_counts[edit_distance]) & (new_seq not in new_seqs):
                    new_seqs.append(new_seq)
                    actual_counts[edit_distance] += 1
                else:
                    pass
                # prevent infinite loops
                if loop_count == max_tries:
                    print(f"Max tries reached. {len(new_seqs):,} out of {target_total:,} sequences made")
                    break
            except KeyError:
                pass # if edit distance not in target_counts
    return new_seqs


def get_target_edit_distance_counts(df, target_total, positives_only=True, label_col="label", edit_distance_col="edit_distance"):
    '''
    Get dictionary of target number of seqs to generate at each edit distance
    Aim to match that of observed experimental data

    :param df: df containing edit distance info from experimental data
    :param positives_only: bool True if only using positive labels
    :param label_col: str name of col with binary labels
    :param target_total: int total number of seqs to generate in artificial library
    :param edit_distance_col: str name of col with edit distances
    :returns: dict contain key values of (edit_distances, num sequences)
    '''
    if positives_only:
        df = df[df[label_col]==1]
    observed_counts = Counter(df[edit_distance_col])
    edit_target_dict = {k: round(v*target_total/df.shape[0]) for k, v in observed_counts.items()}
    # if dict values do not sum to target (due to rounding), add or subtract from edit distance with most counts
    if sum(edit_target_dict.values()) != target_total:
        max_edit_distance = max(edit_target_dict, key=edit_target_dict.get)
        edit_target_dict[max_edit_distance] += target_total - sum(edit_target_dict.values())
    return {k: round(v) for k, v in edit_target_dict.items()}
