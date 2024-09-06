import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.vmd_flag = configs.is_vmd
        self.is_attn_mask = configs.is_attn_mask
        # self.att_mask_none = configs.att_mask_none
        self.embedding_flag = configs.embedding_flag

        # Embedding
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.dec_embedding = DataEmbedding(configs.seq_len, configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            # self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class2)
            self.projection_cls = nn.Linear(configs.d_model, configs.num_class2)

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]
    
    def short_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out  # [B, L, D]

    # def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
    #     # enc
    #     enc_out = self.enc_embedding(x_enc, x_mark_enc)
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)
    #     # final
    #     dec_out = self.projection(enc_out)
    #     return dec_out
    
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, embedding_flag, vmd_flag):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, embedding_flag, vmd_flag)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # projection = nn.Linear(self.d_model, self.seq_len, bias=True)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    # def classification(self, x_enc, x_mark_enc):
    #     # enc
    #     enc_out = self.enc_embedding(x_enc, None)
    #     enc_out, attns = self.encoder(enc_out, attn_mask=None)

    #     # Output
    #     output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
    #     output = self.dropout(output)
    #     output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
    #     output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
    #     output = self.projection(output)  # (batch_size, num_classes)
    #     return output

    def classification(self, x_enc, x_mark_enc, x_mark_enc_binary, embedding_flag, vmd_flag):
        # Embedding
        # convert the x_enc into enc_out by chaning the output dimension into d-model
        enc_out = self.enc_embedding(x_enc, x_mark_enc, embedding_flag, vmd_flag)
        # We may change here the attn_mask into our x_mark_enc
        # boundary = find_boundaries(x_mark_enc.cpu(), mode='thick').astype(np.uint8)
        x_mark_enc_boolean = x_mark_enc_binary.bool()
        if self.is_attn_mask:
            if self.att_mask_none:
        # question is encout has shape of 32, 255, 64 mask only has 32, 255, 1
                enc_out, attns = self.encoder(enc_out, attn_mask=None)
            else:
                enc_out, attns = self.encoder(enc_out, attn_mask=x_mark_enc_boolean)

        else:
            #  enc_out, attns = self.encoder(enc_out, is_attn_mask, att_mask_none, attn_mask=None)
             enc_out, attns = self.encoder(enc_out, attn_mask=None)
             
        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection_cls(output)  # (batch_size, num_classes)
        # output = output.reshape(output.shape[0], -1,  255, 1)
        output = output.reshape(output.shape[0], -1,  1006, 1)

        return output, attns

    def forward(self, x_enc, x_mark_enc, x_mark_enc_binary, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'short_term_forecast':
            dec_out = self.short_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask, self.embedding_flag, self.vmd_flag)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out, attns  = self.classification(x_enc, x_mark_enc, x_mark_enc_binary, self.embedding_flag, self.vmd_flag)
            return dec_out, attns  # [B, N]
        return None
