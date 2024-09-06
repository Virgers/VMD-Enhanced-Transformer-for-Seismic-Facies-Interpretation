import torch
import torch.nn as nn


# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.hidden_size = 64
#         drp = 0.1
#         embed_size = configs.embed_size
#         max_features = configs.max_features
#         embedding_matrix = torch.rand(max_features, embed_size)  # Random embedding matrix
#         self.embedding = nn.Embedding(max_features, embed_size)
#         self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
#         self.embedding.weight.requires_grad = False
#         self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
#         self.linear = nn.Linear(self.hidden_size*4 , 512)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(drp)
#         self.out = nn.Linear(512, max_features)

#     def forward(self, x):
#         h_embedding = self.embedding(x)
#         h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        
#         h_lstm, _ = self.lstm(h_embedding)
#         avg_pool = torch.mean(h_lstm, 1)
#         max_pool, _ = torch.max(h_lstm, 1)
#         #print("avg_pool", avg_pool.size())
#         #print("max_pool", max_pool.size())
#         conc = torch.cat(( avg_pool, max_pool), dim=1)
#         conc = self.relu(self.linear(conc))
#         conc = self.dropout(conc)
#         out = self.out(conc)
#         return out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # input_size is input seq len
        # hidden_size is the  d_model 
        # num_layers default
        # output_size num classes
        # input_size = configs.seq_len2
        input_size = configs.seq_len2
        hidden_size = configs.d_model
        num_layers = configs.e_layers
        output_size = configs.num_class2
        dropout_rate = configs.dropout
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 4, 64)  # *4 because we concatenate avg_pool and max_pool for both directions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(64, output_size * input_size)

    def forward(self, x):
        # No need for embedding layer, x is already numerical
        h_lstm, _ = self.lstm(x)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)

        return out
