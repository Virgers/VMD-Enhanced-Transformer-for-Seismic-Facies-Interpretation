import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.vmd_flag = configs.is_vmd
        self.embedding_flag = configs.embedding_flag
        
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.seq_len, 
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq, 
            configs.dropout
        )
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            configs.factor, 
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ), 
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Task-specific layers
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection_cls = nn.Linear(configs.d_model * configs.enc_in, configs.num_class2)
            
        if self.task_name == 'ssl':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)

    def classification(self, x_enc, x_mark_enc, embedding_flag, vmd_flag):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, embedding_flag, vmd_flag)
        
        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
             
        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = self.projection_cls(output)
        
        # Reshape output - TODO: make this configurable
        output = output.reshape(output.shape[0], -1, 255, 1)
        
        return output, attns
    
    def ssl(self, x_enc, x_mark_enc, embedding_flag, vmd_flag):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, embedding_flag, vmd_flag)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        return dec_out    
    
    def forward(self, x_enc, x_mark_enc):
        if self.task_name == 'classification':
            dec_out, attns = self.classification(x_enc, x_mark_enc, self.embedding_flag, self.vmd_flag)
            return dec_out, attns
        
        if self.task_name == 'ssl':
            dec_out = self.ssl(x_enc, x_mark_enc, self.embedding_flag, self.vmd_flag)
            return dec_out
            
        return None