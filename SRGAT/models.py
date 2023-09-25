from utils import *
from basemodel import *
from bigru_decoder import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SRGAT(nn.Module):
    def __init__(self, args):
        super(SRGAT, self).__init__()
        self.args = args
        self.hidden_size = self.args.hidden_size
        self.Temperal_Encoder=Temperal_Encoder(self.args)
        self.BiGru_Decoder=BiGru_Decoder(self.args)
       
        if self.args.SR:
            message_passing = []
            message_passing_mult = []
            for i in range(self.args.pass_time):
                message_passing.append(Global_interaction(args))
            for i in range(self.args.gs_pass_time):
                message_passing_mult.append(Global_interaction_mult(args))
            self.Global_interaction = nn.ModuleList(message_passing)
            self.Global_interaction_mult = nn.ModuleList(message_passing_mult)
        
        
        self.past_MLP = self.dest_embeding = nn.Sequential(
            MLP(14,self.hidden_size),
            MLP(self.hidden_size,self.hidden_size)
        )

        
        self.dest_embeding = nn.Sequential(
            MLP(2,self.hidden_size),
            MLP(self.hidden_size,self.hidden_size)
        )
        self.dest_latent = nn.Sequential(
            MLP(self.hidden_size*2,self.hidden_size),
            MLP(self.hidden_size,self.hidden_size*2)
        )
        
        self.num_modes =20
       
        self.multihead_proj_past_feat = nn.Sequential(
                                    nn.Linear(self.hidden_size, self.num_modes * self.hidden_size),
                                    nn.LayerNorm(self.num_modes * self.hidden_size),
                                    nn.ReLU(inplace=True)) 
       
        self.decoder_dest = nn.Sequential(
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.Linear(self.hidden_size,2)
        )

    
        self.fuse1 = Gated_fuse(self.args)
        self.dest_h = MLP(self.hidden_size)
        self.dest_c = MLP(self.hidden_size)
        self.fuse_h = Gated_fuse(self.args)
        self.fuse_c = Gated_fuse(self.args)
    def forward(self, inputs, epoch, iftest=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split = inputs # #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
        # frames = batch_abs_gt.transpose(0,1)[:,7,3]
        self.batch_norm_gt = batch_norm_gt
        if self.args.input_offset:
            train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
        elif self.args.input_mix:
            offset = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
            position = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
            pad_offset = torch.zeros_like(position).to(device)
            pad_offset[1:, :, :] = offset
            train_x = torch.cat((position, pad_offset), dim=2)
        elif self.args.input_position:
            train_x = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
        train_x = train_x.permute(1, 2, 0) #[N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0) #[N, 2, H]

        dest_true = train_y[:,:,-1]
        # dest_feat = self.dest_embeding(dest_true)

        self.pre_obs=batch_norm_gt[1:self.args.obs_length]
        self.x_encoded_dense, self.hidden_state_unsplited=self.Temperal_Encoder.forward(train_x)  #[N, D], [N, D]
       
        self.hidden_state_global = torch.ones_like(self.hidden_state_unsplited, device=device)
        # cn_global = torch.ones_like(cn, device=device)
        if self.args.SR:
            for b in range(len(nei_list_batch)):
                left, right = batch_split[b][0], batch_split[b][1]
                element_states = self.hidden_state_unsplited[left: right] #[N, D]
                agent_v = train_x[left: right,:,-1]
    
                # cn_state = cn[left: right] #[N, D]
                if element_states.shape[0] != 1:
                    corr = batch_abs_gt[self.args.obs_length-1, left: right, :2].repeat(element_states.shape[0], 1, 1) #[N, N, D]
                    corr_index = corr.transpose(0,1)-corr  #[N, N, D]  nn之间的距离
                    nei_num = nei_num_batch[left:right, self.args.obs_length-1] #[N]
                    nei_index = torch.tensor(nei_list_batch[b][self.args.obs_length-1], device=device) #[N, N]
                    for i in range(self.args.pass_time):
                        element_states = self.Global_interaction[i](corr_index, nei_index, nei_num, element_states,agent_v)
                    self.hidden_state_global[left: right] = element_states
                    # cn_global[left: right] = cn_state
                else:
                    self.hidden_state_global[left: right] = element_states
                    # cn_global[left: right] = cn_state
        else:
            self.hidden_state_global = self.hidden_state_unsplited
            # cn_global = cn

        fuse_social = self.fuse1(self.x_encoded_dense,self.hidden_state_global)
        past_feat = self.multihead_proj_past_feat(fuse_social).view(-1, self.num_modes, self.hidden_size) #[n,20,c]
        
        dest_pred = self.decoder_dest(past_feat.permute(1,0,2))   # goal
        dest_norm = torch.norm(dest_pred-dest_true,p=2,dim=-1)    # goal_loss
        index = dest_norm.argmin(dim=0)
        best_dest = dest_pred[index,torch.arange(dest_true.size(0))]
        dest_loss = torch.norm(best_dest-dest_true,p=2,dim=-1).mean()
        # dest_feat = self.dest_embeding(dest_pred)
        hidden_state_unsplited = self.hidden_state_unsplited.repeat(20,1,1)
        # cn = cn.repeat(20,1,1)
        # dest_feat = self.dest_embeding(best_dest)
        dest_pred_feat = self.dest_embeding(dest_pred)
        dest_h = self.dest_h(dest_pred_feat)
        # dest_c = self.dest_c(dest_pred_feat)
        fuse_h = self.fuse_h(dest_h,hidden_state_unsplited)
        # fuse_c = self.fuse_c(dest_c,cn)
        dest_abs_pred = batch_abs_gt[self.args.obs_length-1,:, :2].repeat(20,1,1)+dest_pred
        self.hidden_state_global_goal = torch.ones_like(hidden_state_unsplited, device=device)
        if self.args.GS:
            for b in range(len(nei_list_batch)):
                left, right = batch_split[b][0], batch_split[b][1]
                element_states = fuse_h[:,left: right] #[N, D]
                agent_v = train_x[left: right,:,-1]
                if element_states.shape[1] != 1:
                    corr = batch_abs_gt[self.args.obs_length-1, left: right, :2].repeat(element_states.shape[1], 1, 1) #[N, N, D]
                    corr_dest = dest_abs_pred[:,left: right].unsqueeze(1).repeat(1,element_states.shape[1], 1, 1)
                    corr_dest_index = corr_dest.transpose(1,2)-corr_dest
                    past_dest = dest_abs_pred[:,left: right] - batch_abs_gt[self.args.obs_length-1, left: right, :2].repeat(20,1,1)
                    corr_index = corr.transpose(0,1)-corr  #[N, N, D]
                    nei_num = nei_num_batch[left:right, self.args.obs_length-1] #[N]
                    nei_index = torch.tensor(nei_list_batch[b][self.args.obs_length-1], device=device) #[N, N]
                    for i in range(self.args.gs_pass_time):
                        element_states = self.Global_interaction_mult[i](corr_index, nei_index, nei_num, element_states,corr_dest_index,past_dest,agent_v)
                    self.hidden_state_global_goal[:,left: right] = element_states
                else:
                    self.hidden_state_global_goal[:,left: right] = element_states
        else:
            self.hidden_state_global_goal = hidden_state_unsplited
        out = self.BiGru_Decoder.forward(self.x_encoded_dense, self.hidden_state_global_goal, dest_pred_feat, epoch)
        SRGAT_loss, full_pre_tra = self.l2_loss(train_y.permute(2, 0, 1), out, dest_pred, iftest)  #[K, H, N, 2]
        return dest_loss +SRGAT_loss, full_pre_tra
        

    def l2_loss(self, y, y_prime, dest_pred, iftest):
        batch_size=y.shape[1]
        y = y.permute(1, 0, 2)  #[N, H, 2] 
        pred_y = y_prime#torch.cat((y_prime,dest_pred.unsqueeze(2)),dim=-2)
        full_pre_tra = []
        l2_norm = (torch.norm(pred_y - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = pred_y[best_mode, torch.arange(batch_size)]
        
        l2_loss = torch.norm(y_hat_best-y,p=2,dim=-1).mean(-1).mean(-1)
        #best ADE
        sample_k = pred_y[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        # sample_k = out_mu.permute(1, 0, 2)
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(pred_y[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = pred_y[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        return l2_loss, full_pre_tra


    
