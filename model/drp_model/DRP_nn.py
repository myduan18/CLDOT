import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from model.cell_model.cell_encoder import *
from model.drug_model.drug_encoder import *
from torch_geometric.nn import GINConv
import math

class DRP(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.embed_dim = model_config.get('embed_dim')
        self._hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.drug_encoder = Drug_3d_Encoder(model_config)
        self.cell_encoder = Cell_encoder_gene(model_config)
        self.fc1_drug = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc1_cell = nn.Linear(self._hidden_dim, self.embed_dim)
        self.regression = nn.Sequential(
            nn.Linear(2*self.embed_dim, 1024),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, drug_atom,drug_bond,ge):
        drug_embed = self.drug_encoder(drug_atom,drug_bond)
        drug_embed = nn.ReLU()(self.fc1_drug(drug_embed))
        cell_embed = self.cell_encoder(ge.x,ge.edge_index,ge.batch)
        cell_embed = nn.ReLU()(self.fc1_cell(cell_embed))
        x = torch.cat([drug_embed, cell_embed], -1)
        x = self.regression(x)
        x = nn.Sigmoid()(x)
        return x
    
class MultiOmicsDRP(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.embed_dim = model_config.get('embed_dim')
        self.hidden_dim = model_config.get('hidden_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.drug_encoder = Drug_3d_Encoder(model_config)
        self.cell_encoder_ge = Cell_encoder_gene(model_config)
        self.cell_encoder_mut = Cell_encoder_mut(model_config)
        self.cell_encoder_cnv = Cell_encoder_cnv(model_config)
        self.regression = nn.Sequential(
            nn.Linear(self.hidden_dim*3 + self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, drug_atom, drug_bond, ge, cell_cnv, cell_mut):
        drug_embed = self.drug_encoder(drug_atom,drug_bond)
        ge_embed = self.cell_encoder_ge(ge.x,ge.edge_index,ge.batch)
        mut_embed = self.cell_encoder_mut(cell_mut.x,cell_mut.edge_index,cell_mut.batch)
        cnv_embed = self.cell_encoder_cnv(cell_cnv.x,cell_cnv.edge_index,cell_cnv.batch) 
        cell_embed = torch.cat([ge_embed,mut_embed,cnv_embed],dim = -1)
        x = torch.cat([drug_embed, cell_embed], -1)
        x = self.regression(x)
        return x
    
    
    
    
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, attention_dim):
        super(SelfAttention,self).__init__()
        self.embed_dim = embed_dim
        self.num_head = 2
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.head_dim = self.attention_dim // self.num_head
        assert self.head_dim * self.num_head == self.attention_dim, "attention_dim must be divisible by num_head"
        self.values = nn.Linear(1, self.attention_dim)
        self.keys = nn.Linear(1, self.attention_dim)
        self.queries = nn.Linear(1, self.attention_dim)
        self.fc_out = nn.Linear(self.attention_dim, self.attention_dim)
    def forward(self, values, keys, queries): ## Cell, Cell, Drug
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        values = values.unsqueeze(dim=-1)
        keys = keys.unsqueeze(dim=-1)
        queries = queries.unsqueeze(dim = -1)
        values = self.values(values)
        keys = self.keys(keys)
        queries_out = self.queries(queries)
        ## Split embedding into self.num_head pieces
        values = values.reshape(N, self.hidden_dim, self.num_head, self.head_dim)
        keys = keys.reshape(N , self.hidden_dim, self.num_head, self.head_dim)
        queries = queries_out.reshape(N, self.embed_dim, self.num_head, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_dim ** (1/2)), dim = 3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_head * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.
        out = self.fc_out(out) 
        return out,queries_out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, attention_dim):
        super(TransformerBlock, self).__init__()
        self.attention_dim = attention_dim
        self.attention = SelfAttention(embed_dim, hidden_dim,attention_dim)
        self.embed_size = embed_dim
        self.norm1 = nn.LayerNorm(self.attention_dim)
        self.norm2 = nn.LayerNorm(self.attention_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.attention_dim, 2 * self.attention_dim),
            nn.ReLU(),
            nn.Linear(2 * self.attention_dim, self.attention_dim),
        )

        self.dropout = nn.Dropout(0.4)

    def forward(self, value, key, query):
        attention,queries = self.attention(value, key, query)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Multi_Omics_TransRegression(nn.Module):
    def __init__(self, model_config):
        super(Multi_Omics_TransRegression, self).__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.use_cnn = model_config.get('use_cnn')
        self.use_smiles = model_config.get('use_smiles')
        self.use_fp = model_config.get('use_fp')
        self.attention_dim = model_config.get('attention_dim')
        self.drug_encoder = drug_encoder(model_config)
        self.cell_encoder_ge = Cell_encoder_gene(model_config)
        self.cell_encoder_mut = Cell_encoder_mut(model_config)
        self.cell_encoder_cnv = Cell_encoder_cnv(model_config)
        self.graph_dim = self.drug_encoder.graph_dim
        self.gene_attn = TransformerBlock(self.graph_dim, self.hidden_dim, self.attention_dim)
        self.mut_attn = TransformerBlock(self.graph_dim, self.hidden_dim, self.attention_dim)
        self.cnv_attn = TransformerBlock(self.graph_dim, self.hidden_dim, self.attention_dim)
        self.regression = nn.Sequential(
            nn.Linear(3*self.graph_dim * self.attention_dim, 1024),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(1024,1) 
            # nn.ELU(),
            # nn.Dropout(p=self.dropout_rate),
            # nn.Linear(self.graph_dim, self.embed_dim),
            # nn.ELU(),
            # nn.Dropout(p=self.dropout_rate),
            # nn.Linear(self.embed_dim, 1)
        )

    def forward(self, drug_atom, drug_bond, ge, cell_cnv, cell_mut):
        drug_embed = self.drug_encoder(drug_atom,drug_bond)
        ge_embed = self.cell_encoder_ge(ge.x,ge.edge_index,ge.batch)
        mut_embed = self.cell_encoder_mut(cell_mut.x,cell_mut.edge_index,cell_mut.batch)
        cnv_embed = self.cell_encoder_cnv(cell_cnv.x,cell_cnv.edge_index,cell_cnv.batch) 
        attended_gene_feat = self.gene_attn(ge_embed, ge_embed, drug_embed)
        attended_mut_feat = self.mut_attn(mut_embed, mut_embed, drug_embed)
        attended_cnv_feat = self.cnv_attn(cnv_embed, cnv_embed, drug_embed)
        attended_gene_feat = attended_gene_feat.view(attended_gene_feat.size(0),-1)
        attended_mut_feat = attended_mut_feat.view(attended_mut_feat.size(0),-1)
        attended_cnv_feat = attended_cnv_feat.view(attended_cnv_feat.size(0),-1)
        attended_feat = torch.cat([attended_gene_feat, attended_mut_feat, attended_cnv_feat], dim=-1)
        x = self.regression(attended_feat)
        return x
 
 
 
    
def expr_recon_loss(recon_x, x):
    loss = ((recon_x - x)**2).sum()
    return loss
def kl_loss(mean, log_var):
    loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return loss   
    
class Drug_3d_VAE_resp(nn.Module):
    def __init__(self, model_config):
        super(Drug_3d_VAE_resp, self).__init__()
        self.hidden_dim = model_config.get('hidden_dim')
        self.embed_dim = model_config.get('embed_dim')
        self.dropout_rate = model_config.get('dropout_rate')
        self.attention_dim = model_config.get('attention_dim')
        self.layer_num = model_config['layer_num']
        self.cell_encoder = GE_vae(model_config)
        self.drug_encoder = Drug_3d_Encoder(model_config)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.embed_dim * (self.layer_num + 1), 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.cell_emb = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
        )

        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(512, 1)
        )
    def forward(self, drug_atom, drug_bond, ge):
        z, recon_x, mean, log_var = self.cell_encoder(ge)
        cell_embeds = self.cell_emb(mean)
        drug_embeds = self.drug_encoder(drug_atom, drug_bond)
        drug_embeds = self.drug_emb(drug_embeds)
        x = torch.cat([drug_embeds, cell_embeds], -1)
        x = self.regression(x)
        loss_vae = expr_recon_loss(recon_x, x) + kl_loss(mean, log_var)   
        return x, loss_vae
    