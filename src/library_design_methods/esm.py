import numpy as np
import pandas as pd
import torch
import torch.hub as hub
import esm

from src.utils import get_ordered_AA_one_letter_codes


def get_esm_logits(seq, esm_download_dir, esm_version="esm2_650"):
    '''
    Get raw ESM logits which can be converted to probabilities later

    :param seq: str seq masked with asteriks
    :param esm_download_dir: abs path to dir to download esm model weights to
    :returns: torch tensor with raw esm output (start token omitted), dict of AA: idx
    '''
    seqs = [('seq', seq.replace("*","<mask>"))]
    hub.set_dir(esm_download_dir)

    if esm_version == "esm2_650m":
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif esm_version == "esm1b":
        esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    else:
        raise ValueError(f"esm_version provided '{esm_version}' not currently supported")

    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()

    _, _, batch_tokens = batch_converter(seqs)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33])

    return results["logits"][:,1:], alphabet


def get_esm_AA_ordering(alphabet):
    '''
    Get order of amino acids that esm presents likelihoods

    :returns: list of one letter capitalised amino acids chars
    '''
    AAs = get_ordered_AA_one_letter_codes()
    AA_to_idx_for_20_standard_AAs = {k:v for k, v in alphabet.tok_to_idx.items() if k in AAs}
    return list(AA_to_idx_for_20_standard_AAs.keys())


def reorder_array_from_esm_AA_order_to_alphabetical(arr, alphabet):
    '''
    Reorder array elements from esm AA order to alphabetical

    :param arr: ndarray of shape (20,) - one element of esm likelihoods
    :returns: ndarray of shape (20,) with likelihoods in alphabetical order
    '''
    alphabetical_AAs = get_ordered_AA_one_letter_codes()
    esm_AAs = get_esm_AA_ordering(alphabet)

    reordered_arr = np.zeros_like(arr)
    for idx, value in enumerate(arr):
        reordered_arr[alphabetical_AAs.index(esm_AAs[idx])] = value

    return reordered_arr


def get_esm_probs_for_seq_mask_all_at_once(seq, esm_download_dir, esm_version="esm2_650"):
    '''
    Get probabilities for all masked residues in seq

    :param seq: str seq masked with asteriks (must be one continuous mask)
    :param esm_download_dir: abs path to dir to download esm model weights to
    :returns: ndarry of size (len_mask, 20)
    '''
    raw_logits, alphabet = get_esm_logits(seq, esm_download_dir=esm_download_dir, esm_version=esm_version)

    # reduce esm output to masked residues only
    masked_residues_start_idx = seq.index("*")
    mask_len = seq.count("*")
    logits_for_masked_residiues = raw_logits[:,masked_residues_start_idx:masked_residues_start_idx+mask_len]

    # reduce esm output to 20 standard AAs only
    AAs = get_ordered_AA_one_letter_codes()
    AA_to_idx_for_20_standard_AAs = {k:v for k, v in alphabet.tok_to_idx.items() if k in AAs}
    logits_for_masked_residiues_20_standard_AAs = \
        logits_for_masked_residiues[:,:,min(AA_to_idx_for_20_standard_AAs.values()):max(AA_to_idx_for_20_standard_AAs.values())+1]

    # go from logits to probabiltiies between 0 and 1 that sum to 1
    m = torch.nn.Softmax(dim=2)
    probs_for_masked_residiues_20_standard_AAs = m(logits_for_masked_residiues_20_standard_AAs)

    # squeeze, torch to numpy, reorder to alphabetical
    probs_for_masked_residiues_20_standard_AAs = torch.squeeze(probs_for_masked_residiues_20_standard_AAs, 0).numpy()
    probabilities = np.array([reorder_array_from_esm_AA_order_to_alphabetical(arr, alphabet) for arr in probs_for_masked_residiues_20_standard_AAs])

    # format to match logomaker output for plotting and general consistency
    normalised_mat = pd.DataFrame(probabilities, columns=get_ordered_AA_one_letter_codes())
    normalised_mat.index.name = 'pos'

    return normalised_mat

def get_esm_probs_for_masked_residues_one_at_a_time(seq, unmasked_seq, esm_download_dir, esm_version="esm2_650"):
    '''
    Get probabilities for all masked residues in seq if only one at a time is masked

    :param seq: str seq masked with asteriks (must be one continuous mask)
    :param unmasked_seq: unmasked "*" residues only
    :param esm_download_dir: abs path to dir to download esm model weights to
    :returns: ndarry of size (len_mask, 20)
    '''
    probabilities = []
    for position in range(len(unmasked_seq)):

        unmasked_seq_1star = [aa for aa in unmasked_seq]
        unmasked_seq_1star[position] = "*"
        unmasked_seq_1star = "".join(unmasked_seq_1star)

        seq_1star = seq.replace("*"*len(unmasked_seq), unmasked_seq_1star)
        probabilities.append(get_esm_probs_for_seq_mask_all_at_once(
            seq_1star,
            esm_download_dir=esm_download_dir,
            esm_version=esm_version).to_numpy()[0]
        )

    # format to match logomaker output for plotting and general consistency
    normalised_mat = pd.DataFrame(probabilities, columns=get_ordered_AA_one_letter_codes())
    normalised_mat.index.name = 'pos'

    return normalised_mat
