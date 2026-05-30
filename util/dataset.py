import os
import torch
import pandas as pd
import dgl
import pickle
from torch.utils.data import Dataset
from functools import lru_cache

class DTIGraphDataset(Dataset):
    def __init__(self, data_split='train', root='data/biosnap', split="full_data"):
        self.data_split = data_split
        self.root = root
        self.split = split
       
        predata_root = root.replace("data", "predata").replace("full_predata", "full_data")
        raw_dti_path = os.path.join(predata_root, f"{self.data_split}.csv")
        dti_df = pd.read_csv(raw_dti_path)
        self.samples = dti_df[['DrugBank ID', 'Gene', 'Label']].values.tolist()
        
        chembert_graph_path = os.path.join(root, f"chembert_graph_{self.data_split}.pkl")
        with open(chembert_graph_path, "rb") as f:
            self.records = pickle.load(f)
        
        self.drug_id_to_idx = {record['drug_id']: i for i, record in enumerate(self.records)}
        
        self.drug_graphs = [record['graph'] for record in self.records]
    
        
        pkl_root = root.replace(f"{self.split}", "")
        self.prot_graph_dir = os.path.join(pkl_root, 'protein_150M_graphs_pkl')
        
        self.valid_samples = []
        for drug_id, gene_id, label in self.samples:
            if drug_id in self.drug_id_to_idx:
                prot_path = os.path.join(self.prot_graph_dir, f"{gene_id}.pkl")
                if os.path.exists(prot_path):
                    self.valid_samples.append((drug_id, gene_id, label))
        
        valid_df = pd.DataFrame(self.valid_samples, columns=['DrugBank ID', 'Gene', 'Label'])
        valid_df_path = os.path.join(root, f"valid_samples_{self.data_split}.csv")
        valid_df.to_csv(valid_df_path, index=False)
        print(f"Valid samples saved to {valid_df_path}")
    @staticmethod
    @lru_cache(maxsize=200)  
    def load_prot_graph_cached(path):
        with open(path, 'rb') as f:
            return pickle.load(f)["graph"]
    
            
    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        drug_id, gene_id, label = self.valid_samples[idx]

        
        drug_idx = self.drug_id_to_idx[drug_id]
        drug_graph = self.drug_graphs[drug_idx]
        drug_feat = drug_graph.ndata['bert']

        prot_path = os.path.join(self.prot_graph_dir, f"{gene_id}.pkl")
        prot_graph = self.load_prot_graph_cached(prot_path)
        prot_feat = prot_graph.ndata['h']
        
        return drug_graph, drug_feat, prot_graph, prot_feat, torch.tensor([label], dtype=torch.float)

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def collate_fn(batch):
    drug_gs, drug_hs, prot_gs, prot_hs, labels = map(list, zip(*batch))
    return (
        dgl.batch(drug_gs),
        torch.cat(drug_hs),
        dgl.batch(prot_gs),
        torch.cat(prot_hs),
        torch.tensor(labels, dtype=torch.float)
    )
