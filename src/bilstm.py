import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention =  nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, input_dim)
        scores = self.attention(inputs)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1) # (batch_size, seq_len, 1)
        
        # return the weighted sequence 
        weighted_seq = inputs * weights  # (batch_size, seq_len, input_dim)
        return weighted_seq, weights
    
class BLSTMWithAttention(nn.Module):
    def __init__(self, 
                 input_dim,                         # bert embedding dim
                 hidden_dim,                        # lstm hidden dim
                 output_dim,                        # number of classes
                 num_layers=1,                      # lstm layers
                 bidirectional=True,                # bidirectional lstm
                 dropout=0.5,                       # dropout rate
                 use_attention=True,                # use attention mechanism
                 attention_dim=128,                 # attention layer dimension
                 autoregress = False,                # autoregressive flag  
                 n_groups=1                         # number of groups
                 ):
        super(BLSTMWithAttention, self).__init__()
        self.use_attention = use_attention
        self.autoregress = autoregress

        input_size = input_dim * n_groups
        input_size = input_size + 2 if autoregress else input_size
        self.blstm = nn.LSTM(input_size=input_size, 
                             hidden_size=hidden_dim, 
                             num_layers=num_layers, 
                             batch_first=True,
                             bidirectional=bidirectional)
        
        if self.use_attention:
            final_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.attention_layer = AttentionLayer(final_input_dim, attention_dim)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)  
    
    def forward(self, x, past_status=None, hidden=None):
        '''
        -   x: input tensor of shape (batch_size, group_number, embedding_dim) 
        -  if x is 4D, shape is (batch_size, number of text per user, group_number, embedding_dim) 
        -  past_status: tensor of shape (batch_size, past_seq_len, 2) for autoregressive mode
        -  hidden: initial hidden state for LSTM
        '''

        if x.dim() == 4:
            is_4d = True
            b, n, g, h = x.size()
            x = x.view(b, n, g * h) # reshape to (batch_size, number of text per user, group_number * embedding_dim)
             # note: if groups are a temporal sequences (e.g. sentence 1, sentence 2, ...),
            # x = x.view(b, n*g, h)  # reshape to (batch_size, number of text per user *group, embedding_dim)

        elif x.dim() == 3:
            is_4d = False
            b, n, feat = x.size()
        else:
            raise ValueError(f"Input deve essere 3D o 4D, ricevuto {x.dim()}D")

        # check if past_status is provided in autoregressive mode
        if self.autoregress :
            if past_status is None:
                raise ValueError("past_status must be provided for autoregressive mode")
            
            # past_status has to be expanded to match the batch size and number of texts if necessary
            if past_status.dim() == 2:
                past_status = past_status.unsqueeze(1).expand(-1, n, -1)
            
            # check if past_status has the correct shape
            elif past_status.dim() == 3 and past_status.size(1) != n:
                # past_status shape: (batch_size, past_seq_len, 2)
                past_status = past_status.view(b, n, -1)
                raise ValueError("past_status second dimension must match the number of texts in x")

            x = torch.cat((x, past_status), dim=-1)  # concatenate along feature dimension
        
        out, next_state = self.blstm(x, hidden)
        
        attention_weight = None
        if self.use_attention:
            out, attention_weight = self.attention_layer(out)
        
        out = self.dropout(out)
        prediction = self.fc(out)
        
        return prediction, next_state, attention_weight if self.use_attention else None
        
        