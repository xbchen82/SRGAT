import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from torch import nn, Tensor
from typing import Dict, List, Tuple, NamedTuple, Any
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
# from basemodel import Gated_fuse
class Gated_fuse(nn.Module):
    def __init__(self,args):
        super(Gated_fuse, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        self.encoder_weight = nn.Sequential(
            nn.Linear(self.hidden_size*2,self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Sigmoid()
        )

    def forward(self,past_feat,dest_feat):
        fuse =torch.cat((past_feat,dest_feat),dim=-1)
        weight = self.encoder_weight(fuse)
        fused = past_feat*weight + dest_feat*(1-weight)

        return fused

class GRUDecoder(nn.Module):

    def __init__(self, args) -> None:
        super(GRUDecoder, self).__init__()
        min_scale: float = 1e-3
        self.args = args
        self.input_size = self.args.hidden_size
        self.hidden_size = self.args.hidden_size
        self.future_steps = args.pred_length
        self.num_modes = args.final_mode
        self.min_scale = min_scale
        self.args = args
        self.lstm_for = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.lstm_bac = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        self.multihead_proj_global = nn.Sequential(
                                    nn.Linear(self.input_size , self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes * self.hidden_size),
                                    nn.ReLU(inplace=True))
        self.fuse = Gated_fuse(self.args)   
        self.apply(init_weights)

    def forward(self, global_embed: torch.Tensor, hidden_state, dest_feat) -> Tuple[torch.Tensor, torch.Tensor]:
        global_embed = self.multihead_proj_global(global_embed).view(-1, self.num_modes, self.hidden_size)  # [N, F, D]
     
        global_embed = global_embed.transpose(0, 1)  # [F, N, D]
        goal_embed = dest_feat.transpose(0,1)
        local_embed = hidden_state#.repeat(self.num_modes, 1, 1)  # [F, N, D]
        
        global_embed = global_embed.reshape(-1, self.hidden_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        #  反向输入
        goal_embed = goal_embed.reshape(-1, self.hidden_size)  # [F x N, D]
        goal_embed = goal_embed.expand(self.future_steps, *goal_embed.shape)  #
        local_embed = local_embed.reshape(-1, self.input_size).unsqueeze(0)  # [1, F x N, D]
        # cn = cn.reshape(-1, self.input_size).unsqueeze(0)  # [1, F x N, D]
        out1, _ = self.lstm_for(global_embed, local_embed)
        goal_embed_bac =goal_embed
        out2, _ = self.lstm_bac(goal_embed_bac, local_embed)
        out1 = out1.transpose(0, 1)  # [F x N, H, D]
        out2 = out2.transpose(0, 1).flip(1)
        # dest_feat =dest_feat.view(-1,self.hidden_size).unsqueeze(1).repeat(1,11,1)
        out = self.fuse(out1,out2)
        loc = self.loc(out)  # [F x N, H, 2]
        # scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
        loc = loc.view(self.num_modes, -1, self.future_steps, 2) # [F, N, H, 2]
        # scale = scale.view(self.num_modes, -1, self.future_steps, 2) # [F, N, H, 2]
        return loc # [F, N, H, 2], [F, N, H, 2], [N, F]



   
   
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)