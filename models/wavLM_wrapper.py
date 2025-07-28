from WhiSQI.WavLM.WavLM import WavLM, WavLMConfig
import torch
from torch import Tensor
import torch.nn as nn

BASE_CKPT = 'src/WhiSQI/checkpoints/wavlm_base.pt'

class WavLMfeatureExtractor(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        checkpoint = torch.load(BASE_CKPT)

        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        self.model = model.feature_extractor

    def forward(self, data: Tensor):
        return self.model(data).transpose(1,2)

class WavLMfeatureExtractor_layers(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        checkpoint = torch.load(BASE_CKPT)

        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        self.model = model.feature_extractor

    def forward(self, data: Tensor):
        conv_layers = self.model
        #print(conv_layers)
        layers_list = [data.unsqueeze(1)]
        for f in conv_layers.conv_layers:
            #print(layers_list[-1].shape)
            #print(f)
            x_f = f(layers_list[-1])
            layers_list.append(x_f)
            #print(x_f.shape)
        #pad each tensor in layers_list to the size of the longest:
        pad_to = layers_list[1].shape[-1]
        out_layers_list = []
        for l in layers_list[1:]:
            padded_l = torch.nn.functional.pad(l,(0,pad_to-l.shape[-1]))
            #print("padded_l",padded_l.shape)
            out_layers_list.append(padded_l.permute(0,2,1))

        return out_layers_list

class WavLMfeatureExtractor_singlelayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        checkpoint = torch.load(BASE_CKPT)

        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        self.model = model.feature_extractor
        self.layer = None
    
    def set_layer(self, layer):
        self.layer = layer

    def forward(self, data: Tensor):
        conv_layers = self.model
        #print(conv_layers)
        layers_list = [data.unsqueeze(1)]
        for f in conv_layers.conv_layers[:self.layer + 1]:
            #print(layers_list[-1].shape)
            #print(f)
            x_f = f(layers_list[-1])
            layers_list.append(x_f)
            #print(x_f.shape)
        return x_f

class WavLMFull(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        checkpoint = torch.load(BASE_CKPT)       
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        self.model = model

    def forward(self, data: Tensor):
        return self.model.extract_features(data)[0]

class WavLMFull_all(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        checkpoint = torch.load(BASE_CKPT)      
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        self.model = model

    def forward(self, data: Tensor):
        X = self.model.post_extract_proj(self.model.feature_extractor(data).permute(0,2,1))
        #print(X.shape)
        X = self.model.encoder.pos_conv(X.permute(0,2,1)).permute(0,2,1)
        #print(X.shape)
        layers_list = [X]
        for f in self.model.encoder.layers:
            #print(f)
            x_f = f(layers_list[-1])
            #print(x_f[0].shape)
            #print(x_f[1][1].shape)
            layers_list.append(x_f[0])
        
        #print(len(layers_list))
        #for l in layers_list:
        #    print(l.shape)
        return layers_list
    
if __name__ == "__main__":
    import numpy as np
    import torch
    #import matplotlib.pyplot as plt
    import torchaudio
    import sys
    model = WavLMfeatureExtractor_layers()


    in_tensor,fs = torch.rand(1,64000),16000
    print(in_tensor.shape)
    out = model(in_tensor)
    for l in out:
        print(l.shape)
        print(l.mean())
