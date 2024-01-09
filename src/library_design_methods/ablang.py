import math
import numpy as np
import pandas as pd
import ablang as abl

from src.utils import get_ordered_AA_one_letter_codes


def get_AA_ordering_in_likelihoods():
    '''
    Get order of amino acids that ablang presents likelihoods

    :returns: list of one letter capitalised amino acids chars
    '''
    model = abl.pretrained()
    # dictionary mapping tokens to AAs
    v2aa = model.tokenizer.vocab_to_aa
    # list of one letter AA codes in order that ablang presents probabilities
    return [v2aa[i] for i in range(1,21)]


def get_ablang_likelihoods(seq, chain_type="heavy", remove_start_end_tokens=True):
    '''
    Get raw AbLang likelihoods

    :param seq: str of seq
    :param chain_type: str "heavy" or "light"
    :param remove_start_end_tokens: bool True if want to remove start and end tokens
    :returns: ndarray of likelihoods
    '''
    abl_weights = abl.pretrained(chain_type)
    abl_weights.freeze()

    likelihoods = abl_weights([seq], mode="likelihood")
    if remove_start_end_tokens:
        likelihoods = np.array([likelihood[1:-1] for likelihood in likelihoods])

    return likelihoods


def reorder_array_from_ablang_AA_order_to_alphabetical(arr):
    '''
    Reorder array elements from ablang AA order to alphabetical

    :param arr: ndarray of shape (20,) - one element of ablang likelihoods
    :returns: ndarray of shape (20,) with likelihoods in alphabetical order
    '''
    ablang_AAs = get_AA_ordering_in_likelihoods()
    alphabetical_AAs = get_ordered_AA_one_letter_codes()

    reordered_arr = np.zeros_like(arr)
    for idx, value in enumerate(arr):
        reordered_arr[alphabetical_AAs.index(ablang_AAs[idx])] = value

    return reordered_arr


def get_likelihoods_for_masked_residues(seq, chain_type="heavy"):
    '''
    Get likelihoods for only masked residues
    This removes the first dimension from ablangs default output and arranges likelihoods in alphabetical order

    :param seq: str of seq with "*" masking residues
    :param chain_type: str "heavy" or "light"
    :returns: ndarray of size (num of masked residues, 20)
    '''
    likelihoods = get_ablang_likelihoods(seq, chain_type=chain_type, remove_start_end_tokens=True)

    # get likeihoods for masked residues only and remove first dim of ablang output
    likelihoods = np.array([likelihood for idx, likelihood in enumerate(likelihoods[0]) if seq[idx]=="*"])
    
    # reorder to alphabetical order
    likelihoods = np.array([reorder_array_from_ablang_AA_order_to_alphabetical(likelihood) for likelihood in likelihoods])

    return likelihoods


def get_ablang_probs_for_seq_mask_all_at_once(seq, chain_type="heavy"):
    '''
    Go AbLang probabilities masking all residues at once

    :param seq: str of seq with "*" masking residues
    :param chain_type: str "heavy" or "light"
    :returns: ndarray of size len(masked "*" residues) x 20 with normalised ablang probabilities
    '''
    likelihoods = get_likelihoods_for_masked_residues(seq, chain_type=chain_type)

    probabilities = []
    for aa_likelihoods in likelihoods:
        # use softmax to go from likelihoods to probabilities
        aa_probabilities = np.array([math.exp(l) for l in aa_likelihoods])
        sum_probabilities = aa_probabilities.sum()
        aa_probabilities = [aa_p/sum_probabilities for aa_p in aa_probabilities]
        probabilities.append(aa_probabilities)
    np.array(probabilities)

    # format to match logomaker output for plotting and general consistency
    normalised_mat = pd.DataFrame(probabilities, columns=get_ordered_AA_one_letter_codes())
    normalised_mat.index.name = 'pos'

    return normalised_mat


def get_ablang_probs_for_seq_mask_one_at_a_time(seq, unmasked_seq, chain_type="heavy"):
    '''
    Go AbLang probabilities masking residues one at a time

    :param seq: str of seq with "*" masking residues
    :param unmasked_seq: unmasked "*" residues only
    :param chain_type: str "heavy" or "light"
    :returns: ndarray of size len(masked "*" residues) x 20 with normalised ablang probabilities
    '''
    probabilities = []
    for position in range(len(unmasked_seq)):

        unmasked_seq_1star = [aa for aa in unmasked_seq]
        unmasked_seq_1star[position] = "*"
        unmasked_seq_1star = "".join(unmasked_seq_1star)

        # note this this set up to only work if masked residues are consecutive for now
        # e.g. ASYSAY****ASHHSA, not ASYSAY**SA**HHSA
        seq_1star = seq.replace("*"*len(unmasked_seq), unmasked_seq_1star)
        probabilities.append(get_ablang_probs_for_seq_mask_all_at_once(seq_1star, chain_type=chain_type).to_numpy()[0])

    # format to match logomaker output for plotting and general consistency
    normalised_mat = pd.DataFrame(probabilities, columns=get_ordered_AA_one_letter_codes())
    normalised_mat.index.name = 'pos'

    return normalised_mat
