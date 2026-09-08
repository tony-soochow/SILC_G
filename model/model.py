import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class ContrastiveSampleBank(nn.Module):
    def __init__(self, feature_dim=256, bank_size=12000, cluster_update_interval = 3000, temperature=0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.cluster_update_interval = cluster_update_interval
        self.temperature = temperature
        
        self.register_buffer('positive_bank', torch.randn(bank_size, feature_dim) * 0.01)
        self.register_buffer('negative_bank', torch.randn(bank_size, feature_dim) * 0.01)
        self.register_buffer('pos_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('neg_ptr', torch.zeros(1, dtype=torch.long))
        
        self.register_buffer('pos_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('neg_count', torch.zeros(1, dtype=torch.long))
        
    def _add_noise(self, features, noise_level=None):
        if noise_level is None:
            noise_level = 0.02 * (0.5 ** (self.current_epoch // 10))
        noise = torch.randn_like(features) * noise_level
        return features + noise
        
    def update_bank(self, features, labels):
        if not self.training:
            return
            
        batch_size = features.shape[0]
        
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        if pos_mask.sum() > 0:
            pos_features = self._add_noise(features[pos_mask].detach(), noise_level=0.005)

            self._update_positive_bank(pos_features)

        if neg_mask.sum() > 0:
            neg_features = self._add_noise(features[neg_mask].detach(), noise_level=0.005)
            self._update_negative_bank(neg_features)
    
    def _update_positive_bank(self, features):
        batch_size = features.shape[0]
        ptr = int(self.pos_ptr)
        
        if ptr + batch_size <= self.bank_size:
            self.positive_bank[ptr:ptr + batch_size] = features
            ptr = (ptr + batch_size) % self.bank_size
        else:
            overflow = ptr + batch_size - self.bank_size
            self.positive_bank[ptr:] = features[:self.bank_size - ptr]
            self.positive_bank[:overflow] = features[self.bank_size - ptr:]
            ptr = overflow
            
        self.pos_ptr[0] = ptr
        self.pos_count[0] = min(self.pos_count[0] + batch_size, self.bank_size)
    def _update_negative_bank(self, features):
        batch_size = features.shape[0]
        ptr = int(self.neg_ptr)
        
        if ptr + batch_size <= self.bank_size:
            self.negative_bank[ptr:ptr + batch_size] = features
            ptr = (ptr + batch_size) % self.bank_size
        else:
            overflow = ptr + batch_size - self.bank_size
            self.negative_bank[ptr:] = features[:self.bank_size - ptr]
            self.negative_bank[:overflow] = features[self.bank_size - ptr:]
            ptr = overflow
            
        self.neg_ptr[0] = ptr
        self.neg_count[0] = min(self.neg_count[0] + batch_size, self.bank_size)
    def contrastive_loss(self, anchor_features, labels):
        min_samples_threshold = 500
        if not self.training or self.pos_count < min_samples_threshold or self.neg_count < min_samples_threshold:
            return torch.tensor(0.0, device=anchor_features.device)
        
        batch_size = anchor_features.shape[0]
        sample_size = self.cluster_update_interval
        pos_indices = torch.randperm(self.bank_size)[:sample_size]
        neg_indices = torch.randperm(self.bank_size)[:sample_size]

        pos_subset = self.positive_bank[pos_indices]
        neg_subset = self.negative_bank[neg_indices]
        
        anchor_features = self._add_noise(anchor_features, noise_level=0.01)
        pos_subset = self._add_noise(pos_subset, noise_level=0.01)
        neg_subset = self._add_noise(neg_subset, noise_level=0.01)
        
        pos_sim_matrix = F.cosine_similarity(
            anchor_features.unsqueeze(1), pos_subset.unsqueeze(0), dim=2
        ) / self.temperature
        
        neg_sim_matrix = F.cosine_similarity(
            anchor_features.unsqueeze(1), neg_subset.unsqueeze(0), dim=2
        ) / self.temperature
        
        total_loss = 0.0
        valid_count = 0
        
        for i in range(batch_size):
            label = labels[i].item()
            
            if label == 1:
                pos_sim = pos_sim_matrix[i]
                neg_sim = neg_sim_matrix[i]
                all_sim = torch.cat([pos_sim, neg_sim])
                pos_exp_sum = torch.logsumexp(pos_sim, dim=0)
                all_exp_sum = torch.logsumexp(all_sim, dim=0)
                loss = -(pos_exp_sum - all_exp_sum)
            else:
                pos_sim = pos_sim_matrix[i]
                neg_sim = neg_sim_matrix[i]
                
                all_sim = torch.cat([neg_sim, pos_sim])
                
                neg_exp_sum = torch.logsumexp(neg_sim, dim=0)
                all_exp_sum = torch.logsumexp(all_sim, dim=0)
                loss = -(neg_exp_sum - all_exp_sum)
            
            total_loss += loss
            valid_count += 1
        
        return total_loss / max(valid_count, 1)
    
class GCNEncoder(nn.Module):
    def __init__(self, layer_dims):
        super(GCNEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = nn.GELU()

        for i in range(len(layer_dims) - 1):
            self.layers.append(dglnn.GraphConv(layer_dims[i], layer_dims[i+1]))
            self.norms.append(nn.LayerNorm(layer_dims[i+1]))

    def forward(self, g, h):
        for layer, norm in zip(self.layers, self.norms):
            h = self.act(norm(layer(g, h)))

        with g.local_scope():
            g.ndata['h'] = h
            graph_repr = dgl.mean_nodes(g, 'h')

        return h, graph_repr


def pad_batched_node_features(graph, node_features):
    node_counts = graph.batch_num_nodes().tolist()
    node_sequences = torch.split(node_features, node_counts, dim=0)
    padded_features = pad_sequence(
        node_sequences,
        batch_first=True,
        padding_value=0.0,
    )

    lengths = torch.as_tensor(node_counts, device=node_features.device)
    positions = torch.arange(
        padded_features.size(1), device=node_features.device
    ).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)

    return padded_features, padding_mask

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, node_features, padding_mask):
        scores = self.score(node_features).squeeze(-1)
        scores = scores.masked_fill(padding_mask, float("-inf"))

        weights = torch.softmax(scores, dim=1)
        graph_repr = torch.sum(
            node_features * weights.unsqueeze(-1),
            dim=1
        )

        return graph_repr

class SimpleRBFEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, rbf_features):
        return self.proj(rbf_features)
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn = MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )

    def forward(
        self,
        query,
        key_value,
        query_padding_mask=None,
        key_padding_mask=None,
        need_weights=False,
    ):
        attn_out, attn_weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )

        out1 = self.norm1(query + attn_out)

        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + ffn_out)

        if query_padding_mask is not None:
            out2 = out2.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

        return out2, attn_weights


class DTIModel(nn.Module):
    def __init__(self, in_dim_drug, in_dim_prot, hidden_dim=256, out_dim=128,
                bank_size=12000, cluster_update_interval = 3000):
        super(DTIModel, self).__init__()
        self.out_dim = out_dim
        self.bank_size = bank_size
        self.cluster_update_interval = cluster_update_interval
        
        self.drug_encoder = GCNEncoder([in_dim_drug, hidden_dim, out_dim])       # 384 → 256 → 128
        self.prot_encoder = GCNEncoder([in_dim_prot, hidden_dim *2, hidden_dim, out_dim])  # 1305 → 512 → 256 → 128

        self.cross_attn_d2p = CrossAttention(out_dim)
        self.cross_attn_p2d = CrossAttention(out_dim)

        self.drug_attention_pool = AttentionPooling(out_dim)
        self.prot_attention_pool = AttentionPooling(out_dim)

        pair_feature_dim = out_dim * 2  # drug_repr + prot_repr = 128*2 = 256
        self.sample_bank = ContrastiveSampleBank(
            feature_dim=pair_feature_dim, bank_size=self.bank_size, cluster_update_interval=self.cluster_update_interval)

        self.dropout = nn.Dropout(0.3)
        
        fused_dim = (out_dim) * 8  # drug_orig + prot_orig + d2p + p2d + diff + mul + sym_avg + weighted_fusion = 128*8 = 1024
        self.main_classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),  # 1024 -> 256
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 -> 128
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # 128 -> 1
        )
        self.graph_classifier = nn.Sequential(
            nn.Linear((out_dim) * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.spatial_classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, drug_graph, drug_feat, prot_graph, prot_feat, labels=None):
        drug_node_features, drug_repr = self.drug_encoder(
            drug_graph, drug_feat
        )

        prot_node_features, prot_repr = self.prot_encoder(
            prot_graph, prot_feat
        )

        pair_features = torch.cat([drug_repr, prot_repr], dim=-1)
        if self.training and labels is not None:
            self.sample_bank.update_bank(pair_features, labels)

        drug_nodes, drug_padding_mask = pad_batched_node_features(
            drug_graph, drug_node_features
        )
        prot_nodes, prot_padding_mask = pad_batched_node_features(
            prot_graph, prot_node_features
        )

        drug_cross_nodes, _ = self.cross_attn_d2p(
            query=drug_nodes,
            key_value=prot_nodes,
            query_padding_mask=drug_padding_mask,
            key_padding_mask=prot_padding_mask,
            need_weights=False,
        )

        prot_cross_nodes, _ = self.cross_attn_p2d(
            query=prot_nodes,
            key_value=drug_nodes,
            query_padding_mask=prot_padding_mask,
            key_padding_mask=drug_padding_mask,
            need_weights=False,
        )

        d2p = self.drug_attention_pool(
            drug_cross_nodes,
            drug_padding_mask
        )

        p2d = self.prot_attention_pool(
            prot_cross_nodes,
            prot_padding_mask
        )
        
        drug_orig = drug_repr
        prot_orig = prot_repr
        
        diff = d2p - p2d
        mul = d2p * p2d
        
        sym_avg = (d2p + p2d) / 2
        
        alpha = torch.sigmoid(torch.sum(d2p * p2d, dim=-1, keepdim=True))
        weighted_fusion = alpha * d2p + (1 - alpha) * p2d

        main_fused = torch.cat([
            drug_orig, prot_orig,
            d2p, p2d,
            diff, mul,
            sym_avg, weighted_fusion
        ], dim=-1)
            
        x1 = self.main_classifier[0](main_fused)  # Linear 1024 -> 256
        x1 = self.main_classifier[1](x1)           # GELU
        x1 = self.main_classifier[2](x1)           # Dropout
        x1 = self.main_classifier[3](x1)           # Linear 256 -> 128
        main_rep = self.main_classifier[4](x1)         # GELU
        main_rep = self.main_classifier[5](main_rep)       # Dropout
        main_logits = self.main_classifier[6](main_rep)    # Linear 128 -> 1
        main_pred = main_logits.squeeze(-1)

        graph_fused = torch.cat([drug_repr, prot_repr], dim=-1)
        x2 = self.graph_classifier[0](graph_fused)
        x2 = self.graph_classifier[1](x2)
        x2 = self.graph_classifier[2](x2)
        graph_rep = x2
        graph_logits = self.graph_classifier[3](graph_rep)
        graph_pred = graph_logits.squeeze(-1)
        
        return {
            'main_pred': main_pred,
            'graph_pred': graph_pred, 
            'spatial_pred': graph_pred,
            'pair_features': pair_features
            }

class MultiLossFunction(nn.Module):
    def __init__(self, alpha=1, beta=0.3, gamma=0.3, delta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, model, predictions, targets):
        main_loss = self.bce_loss(predictions['main_pred'], targets.float())
        
        graph_loss = self.bce_loss(predictions['graph_pred'], targets.float())
                
        spatial_loss = self.bce_loss(predictions['spatial_pred'], targets.float())
        contrastive_loss = model.sample_bank.contrastive_loss(
            predictions['pair_features'], targets
        )
        total_loss = (self.alpha * main_loss + 
                     self.beta * graph_loss + 
                     self.gamma * spatial_loss +
                     self.delta * contrastive_loss)
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'graph_loss': graph_loss,
            'spatial_loss': spatial_loss,
            'contrastive_loss': contrastive_loss
        }
