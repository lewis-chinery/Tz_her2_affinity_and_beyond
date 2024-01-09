def get_H3_seqs_from_fasta(output_fasta, unique=True, H3_start_idx=106, H3_end_idx=116):
    '''
    Get H3 seqs from fasta of ProteinMPNN output
    ProteinMPNN output generated using e.g. ABodyBuilder2 model of Trastuzumab
    Output generated using example bash scripts from https://github.com/dauparas/ProteinMPNN
    
    :param output_fasta: str abs path to fasta file
    :param H3_start_idx: int of start idx of H3 seq in fasta (not IMGT numbering)
    :param H3_end_idx: int of end idx of H3 seq in fasta (not IMGT numbering)
    :returns: list of str of H3 seqs
    '''
    # read fasta
    with open(output_fasta) as f:
        lines = f.readlines()
    # get H3 seqs
    seqs = [line.strip("\n") for line in lines if line[0] != ">"]
    H3s = [seq[H3_start_idx:H3_end_idx] for seq in seqs]
    H3s = list(set(H3s)) if unique else H3s
    return H3s
