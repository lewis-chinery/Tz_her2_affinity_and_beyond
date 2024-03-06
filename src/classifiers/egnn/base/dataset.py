from pathlib import Path
import struct
import numpy as np
from scipy.spatial.distance import cdist
import torch as th
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import time
import pandas as pd
import sys
from torch_geometric.utils import remove_self_loops
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data
from collections import defaultdict
import random
from biopandas.pdb import PandasPdb

# import code needed for typing atoms
if __name__ == "__main__":
    import utils
    from atom_types import Typer
else:
    try:
        from . import utils
        from .atom_types import Typer
    except:
        from base import utils
        from base.atom_types import Typer


class dgDataSet(Dataset):

    def __init__(self,
        interaction_dist: float=4, # cutoff interaction distance for generating graphs, default 4 A
        graph_mode: str="ca_cdrh3_nbhd", # graph generation mode, default 
        typing_mode="res_type", # residue-level graph
        cache_frames: bool=False, 
        rough_search: bool=False,
        force_recalc: bool = False,
        pdb_directory: str,
        pdb_prefix: str,
        ab_chains: list,
        ag_chains: list,
        **kwargs):

        self.pdb_directory = pdb_directory
        self.pdb_prefix = pdb_prefix
        self.ab_chains = ab_chains
        self.ag_chains = ag_chains

        self.type_map = utils.get_type_map()
        if typing_mode == "res_type":
            self.node_feature_size = 22  # 20 aa, 1 for non-canonical aa, 1/0 for binding partner ID
        self.ppi_defs = pd.DataFrame() # store information about entry
        self.labels = [] # dG labels
        self.edge_dim = 2  # intra- vs inter-protein edges
        self.interaction_dist = interaction_dist
        self.typing_mode = typing_mode # typing mode
        self.graph_mode = graph_mode # graph_generation mode 
        self.cache = {}
        self.cache_frames = cache_frames
        self.speedup = rough_search

        self.graph_generation_function_dict = {
            "ca_cdrh3_nbhd": self._get_cdrh3_nbhd_graph,
        }

        self.aa_map = { # map AAs to numbers
            "ALA": 0,
            "ARG": 1,
            "ASN": 2,
            "ASP": 3,
            "CYS": 4,
            "GLU": 5,
            "GLN": 6,
            "GLY": 7,
            "HIS": 8,
            "ILE": 9,
            "LEU": 10,
            "LYS": 11,
            "MET": 12,
            "PHE": 13,
            "PRO": 14,
            "SER": 15,
            "THR": 16,
            "TRP": 17,
            "TYR": 18,
            "VAL": 19,
            "XXX": 20,
        }
        
    def _get_cdrh3_nbhd_graph(self, ab_df: pd.DataFrame, ag_df: pd.DataFrame):
        """Return graph composed of nodes in cdrh3 and surrounding neighborhood nodes (inter- and intra-protein edges)
        Pre-defined graph: node input based on residues

        Args:
            ab_df (pd.DataFrame): Protein 1 dataframe (ab)
            ag_df (pd.DataFrame): Protein 2 dataframe (ag)
        """

        # PRE-DEFINING GRAPH based on wt 1n8z pdb (foldx repaired)

        wt_pdb_df = PandasPdb().read_pdb("1n8z_Repair.pdb").df["ATOM"]
        wt_ab_df = wt_pdb_df[wt_pdb_df["chain_id"].isin(["A","B"])].copy()
        wt_ag_df = wt_pdb_df[wt_pdb_df["chain_id"].isin(["C"])].copy()

        wt_ab_df = wt_ab_df[wt_ab_df["atom_name"] == "CA"]
        wt_ag_df = wt_ag_df[wt_ag_df["atom_name"] == "CA"]

        wt_coords_ab = wt_ab_df.loc[:, ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float64)
        wt_coords_ag = wt_ag_df.loc[:, ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float64)
        wt_coords_cdrh3 = wt_ab_df.loc[(wt_ab_df["chain_id"] == "B") & (wt_ab_df["residue_number"].isin(np.arange(99,109))), ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float64)

        dist_between_wt_cdrh3_ab = cdist(wt_coords_cdrh3, wt_coords_ab)
        _, wt_intra_ab = np.where(dist_between_wt_cdrh3_ab < self.interaction_dist)
        wt_ab_node_idx = sorted(np.unique(wt_intra_ab))
        wt_ab_nodes = wt_ab_df.iloc[wt_ab_node_idx, :]
        wt_coords_ab = wt_coords_ab[wt_ab_node_idx, :]

        dist_between_wt_ab_nodes_ag = cdist(wt_coords_ab, wt_coords_ag)
        _, wt_inter_ag = np.where(dist_between_wt_ab_nodes_ag < self.interaction_dist)
        wt_ag_node_idx_init = sorted(np.unique(wt_inter_ag))
        wt_ag_nodes_init = wt_ag_df.iloc[wt_ag_node_idx_init, :]
        wt_coords_ag_init = wt_coords_ag[wt_ag_node_idx_init, :]

        dist_between_wt_ag_nodes_ag = cdist(wt_coords_ag_init, wt_coords_ag)
        _, wt_intra_ag = np.where(dist_between_wt_ag_nodes_ag < self.interaction_dist)
        wt_ag_node_idx = sorted(np.unique(wt_intra_ag))
        wt_ag_nodes = wt_ag_df.iloc[wt_ag_node_idx, :]
        wt_coords_ag = wt_coords_ag[wt_ag_node_idx, :]

        wt_ab_nodes = wt_ab_df.merge(pd.DataFrame(wt_coords_ab, columns=["x_coord", "y_coord", "z_coord"]))
        wt_ag_nodes = wt_ag_df.merge(pd.DataFrame(wt_coords_ag, columns=["x_coord", "y_coord", "z_coord"]))

        assert (wt_ab_nodes.shape[0] == wt_coords_ab.shape[0]) & (wt_ag_nodes.shape[0] == wt_coords_ag.shape[0])

        chain_A_resis = list(wt_ab_nodes[(wt_ab_nodes["chain_id"] == "A")]["residue_number"].unique())
        chain_B_resis = list(wt_ab_nodes[(wt_ab_nodes["chain_id"] == "B")]["residue_number"].unique())
        chain_C_resis = list(wt_ag_nodes["residue_number"].unique())

        # implement pre-defined graph in given (non-wt) structure

        ab_A_nodes = ab_df[(ab_df["chain_id"] == "A") & (ab_df["residue_number"].isin(chain_A_resis))]
        ab_B_nodes = ab_df[(ab_df["chain_id"] == "B") & (ab_df["residue_number"].isin(chain_B_resis))]
        ab_nodes = pd.concat([ab_A_nodes,ab_B_nodes])
        assert ab_nodes.shape[0] == ab_A_nodes.shape[0] + ab_B_nodes.shape[0]
        ag_nodes = ag_df[(ag_df["chain_id"] == "C") & (ag_df["residue_number"].isin(chain_C_resis))]

        coords_ab = ab_nodes.loc[:, ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float64)
        coords_ag = ag_nodes.loc[:, ["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float64)

        #### MAKE FEATURE VECTOR ####
        # features: one-hot encoded residue type
        node_features_ab = self._get_node_features(ab_nodes)
        node_features_ag = self._get_node_features(ag_nodes)

        # is the node on ab or ag?
        node_bool_ab = np.zeros((len(coords_ab), 1))
        node_bool_ag = np.ones((len(coords_ag), 1))

        # features: node xyz coordinates, one-hot encoded residue types, label of whether ab/ag
        feats_ab = np.concatenate([coords_ab, node_features_ab, node_bool_ab], axis=-1)
        feats_ag = np.concatenate([coords_ag, node_features_ag, node_bool_ag], axis=-1)

        feats = np.concatenate([feats_ab, feats_ag], axis=-2)

        #### MAKE EDGE LIST ####
        
        # 1 inter: interactions between ab & ag
        dist_inter = cdist(coords_ab, coords_ag)
        inter_src_ab, inter_dst_ag = np.where(dist_inter < self.interaction_dist)

        # 2 intra_ab: interactions between ab nodes
        dist_intra_ab = cdist(coords_ab, coords_ab)
        intra_src_ab, intra_dst_ab = np.where(dist_intra_ab < self.interaction_dist)

        # 3 intra_ag: interactions between ag nodes
        dist_intra_ag = cdist(coords_ag, coords_ag)
        intra_src_ag, intra_dst_ag = np.where(dist_intra_ag < self.interaction_dist)

        # don't need node lookup as only final distances (for edges) only measured for nodes in final dataset 
        ag_offset = len(ab_nodes)

        # edge source and destination nodes
        edge_src = np.concatenate([
            [i for i in inter_src_ab],
            [i for i in intra_src_ab],
            [i + ag_offset for i in intra_src_ag]
        ])
        edge_dst = np.concatenate([
            [i + ag_offset for i in inter_dst_ag],
            [i for i in intra_dst_ab],
            [i + ag_offset for i in intra_dst_ag]
        ])

        # each edge must be bidirectional
        edge_src_full = np.concatenate([edge_src, edge_dst])
        edge_dst_full = np.concatenate([edge_dst, edge_src])
        edge_indices = np.vstack([edge_src_full, edge_dst_full]).astype(np.float64)

        #### MAKE EDGE ATTRIBUTES ####

        # for now just 1/0 for inter/intra
        edge_attr = np.concatenate([
            np.ones(len(inter_dst_ag)),
            np.zeros(len(intra_dst_ab)),
            np.zeros(len(intra_dst_ag)),
        ])
        
        edge_attr_full = np.expand_dims(np.concatenate([edge_attr, edge_attr]), 1)

        return feats, edge_indices, edge_attr_full


    def _get_node_features(self, df: pd.DataFrame):
        mode = self.typing_mode

        if mode == "res_type":
            types = df["residue_name"] # get residue name
            types = types.apply(lambda x: self.aa_map[x] if x in self.aa_map.keys() else 20).astype(np.int64) # map to integer as defined in aa_map
            types = np.array(types)
            types = utils.get_one_hot(types, nb_classes=max(self.aa_map.values()) + 1) # one-hot encode
            return types
        else:
            raise NotImplementedError(mode)

            
    def _generate_graph_dict(self, ab_df: pd.DataFrame, ag_df: pd.DataFrame):
        """Generate dictionary of nodes, edge indices and edge attributes in graph

       Args:
            ab_df (pd.DataFrame): Protein 1 dataframe
            ag_df (pd.DataFrame): Protein 2 dataframe
        """
        graph_dict = {}
        nodes, edge_ind, edge_attr = self.graph_generation_function_dict[self.graph_mode](ab_df, ag_df)
        ## nodes = nodes with features: node xyz coordinates, one-hot encoded residue types, bool ab/ag
        ## edge_ind: indices of source & destination nodes defining edges
        ## edge_attr: label of whether intra- or inter-protein edge
        
        graph_dict["nodes"] = nodes
        graph_dict["edge_ind"] = edge_ind
        graph_dict["edge_attr"] = edge_attr

        return graph_dict

    
    def _parse_graph(self, graph_dict: dict, label:np.ndarray, ppi_def:dict):
        """Generate parsed graph object in correct format. This is intended to be overwritten 
            by child classes depending on requirements of the downstream models. 
            Base implementation parses the data into a pytorch geometric data instance.

        Args:
            graph_dict: output of _generate_graph_dict, contains nodes (coordinates & features), edge indices and edge attributes of graph
        """

        # remove self loops
        edge_index, edge_attr = remove_self_loops(
                edge_index=th.from_numpy(graph_dict["edge_ind"]).long(),
                edge_attr=th.from_numpy(graph_dict["edge_attr"])
            )   
        
        graph = Data( # add information to torch_geometric.data.Data ("A data object describing a homogeneous graph")
            x=th.from_numpy(graph_dict["nodes"][:,3:]), # node features: whether node is on ab/ag, one-hot encoded type
            edge_index=edge_index, # source & destination nodes defining edges
            edge_attr=edge_attr, # whether edge is intra (0) or inter (1) protein
            pos=th.from_numpy(graph_dict["nodes"][:,:3]), # node xyz coordinates
            y=th.tensor(label), # ddG label
            pdb_file = f"{self.pdb_directory}/{self.pdb_prefix}{ppi_def['seq']}.pdb" # path to pdb file
        )
        
        return graph
    
        
    def __len__(self):
        """ Get number of entries in dataset """
        return len(self.labels)


    def populate(self, input_file: Path, overwrite: bool=False):
        """Extract information from input files and save in ppi_defs list of lists

        Args:
            input_file: cvs containing columns: seq, class, label, edit_distance
        """
        
        ppi_defs = pd.read_csv(input_file)
        labels = ppi_defs["label"].to_list()

        # either overwrite or add to self.ppi_defs and self.labels
        if overwrite:
            self.ppi_defs = ppi_defs
            self.labels = labels
        else:
            if len(self.ppi_defs) > 0:
                self.ppi_defs += ppi_defs
            else:
                self.ppi_defs = ppi_defs
            if len(self.labels) != 0:
                self.labels = self.labels + labels
            else:
                self.labels = labels
    

    def __getitem__(self, idx: int, force_recalc: bool = False):
        """ Generate graph for complex in dataset
        
        Args:
            idx: index
        """
        
        ##### read in pdbs #####

        # obtain label and ppi complex info (path to pdb file, chains in ab & ag)
        label = self.labels[idx]
        ppi_def = self.ppi_defs.iloc[idx]

        # generate path to typed parquet file
        typed_pdb = Path(f"{self.pdb_directory}/{self.pdb_prefix}{ppi_def['seq']}.ca.parquet")
        
        # check if typed file in cache
        if self.cache_frames and str(typed_pdb) in self.cache and not force_recalc:
            pdb_df = self.cache[str(typed_pdb)].copy()

        # check if typed file exists
        elif typed_pdb.exists() and not force_recalc:
            pdb_df = pd.read_parquet(typed_pdb)
                            
        # if not create and save typed files
        else:
            pdb_df = utils.parse_pdb_to_parquet(f"{self.pdb_directory}/{self.pdb_prefix}{ppi_def['seq']}.pdb", typed_pdb, lmg_typed=True, ca=True)

        if self.cache_frames:
            if not str(typed_pdb) in self.cache:
                self.cache[str(typed_pdb)] = pdb_df.copy()
        
        ##### generate graphs #####
        
        # split pdb file into ab/ag
        ## identify chains in ab/ag
        chs_ab = []
        for ch in self.ab_chains:
            chs_ab.append(ch)

        chs_ag = []
        for ch in self.ag_chains:
            chs_ag.append(ch)
        
        ab_df = pdb_df[pdb_df["chain_id"].isin(chs_ab)]
        ag_df = pdb_df[pdb_df["chain_id"].isin(chs_ag)]
            
        # generate graph
        graph_dict = self._generate_graph_dict(ab_df, ag_df)
        graph = self._parse_graph(graph_dict, label, ppi_def=ppi_def)

        return graph
