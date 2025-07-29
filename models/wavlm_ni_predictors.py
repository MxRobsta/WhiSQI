from torch import nn
import torch
from typing import Optional

from WhiSQI.models.wavLM_wrapper import WavLMfeatureExtractor_layers

class PoolAttFF(nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, dim_head_in):
        super().__init__()
        
        self.linear1 = nn.Linear(dim_head_in, 2*dim_head_in)
        self.linear2 = nn.Linear(2*dim_head_in, 1)
        
        self.linear3 = nn.Linear(dim_head_in, 1)
        
        self.activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = torch.nn.functional.softmax(att, dim=2)
      
        x = torch.bmm(att, x) 
      
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x 

class cpcWavLMLSTMLayers(nn.Module):
    def __init__(
        self, hidden_size):
        super().__init__()
        self.norm_input = nn.BatchNorm1d(512)

        self.feat_extract = WavLMfeatureExtractor_layers()
        self.feat_extract.requires_grad_(False)

        self.layer_weights = nn.Parameter(torch.ones(7))
        self.softmax = nn.Softmax(dim=0)

        self.blstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True,
        )

        self.attenPool = PoolAttFF(hidden_size * 2)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lens: Optional[list[int]] = None):

        out_feats = torch.stack(self.feat_extract(x), dim=-1) #whisper encoder returns (B, 1500, 512)
        out_feats = out_feats @ self.softmax(self.layer_weights) #weighted sum of the 13 tensors
        out_feats = self.norm_input(out_feats.permute(0,2,1)).permute(0,2,1) #normalize and permute back to (B, 1500, 512)

        pad = x.shape[0] > 1
        if pad:
            if lens is None:
                raise ValueError("Lengths should be provided for batch processing")
            lens = [min(out_feats.shape[1], l // 5) for l in lens]
            out_feats = torch.nn.utils.rnn.pack_padded_sequence(out_feats, lens, batch_first=True, enforce_sorted=False)

        out, _ = self.blstm(out_feats) # transformer returns (B, 1500, 256)

        if pad:
            out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        out = self.attenPool(out) #attenPool returns (B, 1)
        out = self.sigmoid(out) #sigmoid returns (B, 1)
        return out.squeeze(1)

if __name__ == "__main__":
    model = cpcWavLMLSTMLayers(128)

    audio = torch.rand(4, 32000)
    import numpy as np
    lens = list(np.random.randint(16000, 32000, 4))

    thing = model(audio, lens)
    print(thing)