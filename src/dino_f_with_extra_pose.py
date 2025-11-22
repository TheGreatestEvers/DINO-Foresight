import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import einops
from time import time
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np
from torchmetrics import JaccardIndex
from torchmetrics.aggregation import MeanMetric
from src.attention_masked import MaskTransformer, MaskTransformerWithPose
import math
from dpt import DPTHead
import timm
from src.pose_eval_helpers import *

def _lad2_scale_shift(p, g, s_init=None, lr=1e-4, max_iters=1000, tol=1e-6):
    """
    Robust L1 fit for scale+shift: minimize sum |s * p + t - g|.
    Works even if called from a no_grad() context.
    """
    p = p.detach()
    g = g.detach()

    if s_init is None:
        s_init = (torch.median(g) / (torch.median(p) + 1e-8)).item()

    s = torch.tensor([s_init], dtype=g.dtype, device=g.device, requires_grad=True)
    t = torch.tensor([0.0],    dtype=g.dtype, device=g.device, requires_grad=True)
    opt = torch.optim.Adam([s, t], lr=lr)

    last = None
    for _ in range(max_iters):
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = (s * p + t - g).abs().sum()
        loss.backward()
        opt.step()

        cur = float(loss.detach())
        if last is not None and abs(last - cur) < tol:
            break
        last = cur

    return s.detach(), t.detach()

def _scale_shift_l2_closed_form(p, g):
    """
    Minimize || s*p + t - g ||_2^2 in closed form (works in no_grad / inference_mode).
    Returns s, t as 0-D tensors on the same device/dtype.
    """
    p = p.detach().to(dtype=g.dtype)
    g = g.detach()
    pm, gm = p.mean(), g.mean()
    varp = (p - pm).pow(2).sum().clamp_min(1e-8)
    cov  = ((p - pm) * (g - gm)).sum()
    s = cov / varp
    t = gm - s * pm
    return s, t

@torch.no_grad()
def _align_depth(pred, gt, mode="scale&shift", post_clip_max=None):
    if pred.numel() == 0:
        return pred

    if mode == "metric":
        pa = pred
    elif mode == "scale":
        s = (gt.mean() / (pred.mean() + 1e-8))
        pa = s * pred
    elif mode == "scale&shift":
        # If grads are disabled (Lightning validation), use closed-form L2.
        # If for some reason grads are enabled (e.g., debug), keep your robust L1.
        if torch.is_grad_enabled():
            s, t = _lad2_scale_shift(pred, gt)   # your robust L1 fit (uses autograd)
        else:
            s, t = _scale_shift_l2_closed_form(pred, gt)
        pa = s * pred + t
    else:  # "median"
        s = torch.median(gt) / (torch.median(pred) + 1e-8)
        pa = s * pred

    if post_clip_max is not None:
        pa = torch.clamp(pa, max=post_clip_max)
    return pa

def update_depth_metrics(pred, gt, d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    valid_pixels = gt > 0
    pred = pred[valid_pixels]
    gt = gt[valid_pixels]
    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()
    d1_m.update(d1)
    d2_m.update(d2)
    d3_m.update(d3)
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.float().mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100
    log_10 = (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()
    abs_rel_m.update(abs_rel)
    rmse_m.update(rmse)
    log_10_m.update(log_10)
    rmse_log_m.update(rmse_log)
    silog_m.update(silog)
    sq_rel_m.update(sq_rel)

def compute_depth_metrics(d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    d1 = d1_m.compute()
    d2 = d2_m.compute()
    d3 = d3_m.compute()
    abs_rel = abs_rel_m.compute()
    rmse = rmse_m.compute()
    log_10 = log_10_m.compute()
    rmse_log = rmse_log_m.compute()
    silog = silog_m.compute()
    sq_rel = sq_rel_m.compute()
    return d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel
    
def reset_depth_metrics(d1_m, d2_m, d3_m, abs_rel_m, rmse_m, log_10_m, rmse_log_m, silog_m, sq_rel_m):
    d1_m.reset()
    d2_m.reset()
    d3_m.reset()
    abs_rel_m.reset()
    rmse_m.reset()
    log_10_m.reset()
    rmse_log_m.reset()
    silog_m.reset()
    sq_rel_m.reset()

def update_normal_metrics(pred, gt, mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred, gt, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    mean_ae = pred_error.mean()
    median_ae = pred_error.median()
    rmse = torch.sqrt((pred_error ** 2).mean())
    a1 = 100*(pred_error < 5).float().mean()
    a2 = 100*(pred_error < 7.5).float().mean()
    a3 = 100*(pred_error < 11.25).float().mean()
    a4 = 100*(pred_error < 22.5).float().mean()
    a5 = 100*(pred_error < 30).float().mean()
    mean_ae_m.update(mean_ae)
    median_ae_m.update(median_ae)
    rmse_m.update(rmse)
    a1_m.update(a1)
    a2_m.update(a2)
    a3_m.update(a3)
    a4_m.update(a4)
    a5_m.update(a5)

def compute_normal_metrics(mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    mean_ae = mean_ae_m.compute()
    median_ae = median_ae_m.compute()
    rmse = rmse_m.compute()
    a1 = a1_m.compute()
    a2 = a2_m.compute()
    a3 = a3_m.compute()
    a4 = a4_m.compute()
    a5 = a5_m.compute()
    return mean_ae, median_ae, rmse, a1, a2, a3, a4, a5

def reset_normal_metrics(mean_ae_m, median_ae_m, rmse_m, a1_m, a2_m, a3_m, a4_m, a5_m):
    mean_ae_m.reset()
    median_ae_m.reset()
    rmse_m.reset()
    a1_m.reset()
    a2_m.reset()
    a3_m.reset()
    a4_m.reset()
    a5_m.reset()


class Dino_f(pl.LightningModule):
    def __init__(self,args):
        super(Dino_f,self).__init__()
        self.args = args
        self.sequence_length = args.sequence_length # 4
        self.batch_size = args.batch_size 
        self.hidden_dim = args.hidden_dim 
        self.heads = args.heads
        self.layers = args.layers 
        self.dropout = args.dropout
        self.loss_type = args.loss_type
        self.img_size  = args.img_size
        self.d_layers = args.d_layers
        self.patch_size = 14 if self.args.feature_extractor in ['dino', 'eva2-clip'] else 16
        self.d_num_layers = len(self.d_layers) if isinstance(self.d_layers, list) else self.d_layers
        if not self.args.crop_feats and not self.args.sliding_window_inference:
            self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size), self.img_size[1]//(self.patch_size))
        else:
            self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size*2), self.img_size[1]//(self.patch_size*2))
        
        if self.args.feature_extractor == "external":
            # Provide features already concatenated over layers -> treat as single block
            self.d_num_layers = 1
            self.feature_dim = self.args.cached_feature_dim
            Hf, Wf = self.args.feat_hw
            self.shape = (self.sequence_length, Hf, Wf)
            self.embedding_dim = self.feature_dim

            # 3d head for eval
            from src.cut3r_head_loader import load_exported_head
            self.head, head_meta = load_exported_head("/workspace/cut3r-forecasting/cut3r/src/exported_cut3r_head")
            self.head_img_hw = (224,224)
            self._do_eval3d_this_epoch = False

            self.pose_loss_weight = getattr(self.args, "pose_loss_weight", 0.1)
        else:
            pass

        if self.args.pca_ckpt:
            self.pca_dict = torch.load(self.args.pca_ckpt, weights_only=False)
            self.pca = self.pca_dict['pca_model']
            self.pca_mean = torch.nn.Parameter(torch.tensor(self.pca.mean_), requires_grad=False)
            self.pca_components = torch.nn.Parameter(torch.tensor(self.pca.components_), requires_grad=False)
            self.mean = torch.nn.Parameter(torch.tensor(self.pca_dict['mean']), requires_grad=False)
            self.std = torch.nn.Parameter(torch.tensor(self.pca_dict['std']),requires_grad=False)
            self.embedding_dim = self.pca_components.shape[0]
        #self.maskvit = MaskTransformer(shape=self.shape, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, depth=self.layers,
        #                               heads=self.heads, mlp_dim=4*self.hidden_dim, dropout=self.dropout,use_fc_bias=args.use_fc_bias,
        #                               seperable_attention=args.seperable_attention,seperable_window_size=args.seperable_window_size, use_first_last=args.use_first_last)
        self.maskvit = MaskTransformerWithPose(shape=self.shape, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, depth=self.layers,
                                       heads=self.heads, mlp_dim=4*self.hidden_dim, dropout=self.dropout,use_fc_bias=args.use_fc_bias,
                                       seperable_attention=args.seperable_attention,seperable_window_size=args.seperable_window_size, use_first_last=args.use_first_last)
        self.train_mask_frames = args.train_mask_frames
        self.train_mask_mode = args.train_mask_mode
        self.masking = args.masking
        assert self.masking in ("half_half", "simple_replace", "half_half_previous")
        if self.masking in ("half_half", "half_half_previous"): # default 
            self.mask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.unmask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            if self.masking=="half_half":
                self.replace_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim//2)
        elif self.masking == "simple_replace":
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)
            self.replace_vector = nn.Parameter(torch.zeros(1, 1, 1, 1, self.hidden_dim))
            torch.nn.init.normal_(self.replace_vector, std=.02)
            self.maskvit.fc_in = nn.Identity()
        self.activation = nn.Sigmoid() if args.output_activation == "sigmoid" else nn.Identity()
        
        self.train_pose_loss_m = MeanMetric()
        self.train_grid_loss_m = MeanMetric()
        
        # Necessary for evaluation
        self.mean_metric = MeanMetric()

        # feature recon metrics
        self.ff_mse = MeanMetric()
        self.ff_mae = MeanMetric()
        self.ff_cos = MeanMetric()

        # pose token recon metrics
        self.pose_ff_mse = MeanMetric()
        self.pose_ff_mae = MeanMetric()
        self.pose_ff_cos = MeanMetric()

        # 3d metrics
        self.depth_absrel_m = MeanMetric()
        self.depth_delta1_m = MeanMetric()

        # pose metrics
        self.pose_ate_m = MeanMetric()
        self.pose_rpe_t_m = MeanMetric()
        self.pose_rpe_r_m = MeanMetric()

        self.batch_crops = []
        self.random_crop = T.RandomCrop(16,32)
        self.save_hyperparameters()
       

    def on_load_checkpoint(self, checkpoint):
        if self.args.high_res_adapt:
            # Interpolate positional embeddings
            pos_emb_h = checkpoint['state_dict']['maskvit.pos_embd.emb.d_1']
            pos_emb_h = F.interpolate(pos_emb_h.unsqueeze(0).unsqueeze(0), size=(self.img_size[0]//(self.patch_size), pos_emb_h.shape[1]), mode="bilinear")
            checkpoint['state_dict']['maskvit.pos_embd.emb.d_1'] = pos_emb_h.squeeze(0).squeeze(0)
            pos_emb_w = checkpoint['state_dict']['maskvit.pos_embd.emb.d_2']
            pos_emb_w = F.interpolate(pos_emb_w.unsqueeze(0).unsqueeze(0), size=(self.img_size[1]//(self.patch_size), pos_emb_w.shape[1]), mode="bilinear")
            checkpoint['state_dict']['maskvit.pos_embd.emb.d_2'] = pos_emb_w.squeeze(0).squeeze(0)
  
    def sliding_window(self, im, window_size, stride):
        B, SL, H, W, C = im.shape
        ws_h, ws_w = window_size
        s_h, s_w = stride
        windows = []
        for i in range(0, H-ws_h+1, s_h):
            for j in range(0, W-ws_w+1, s_w):
                windows.append(im[:,:,i:i+ws_h,j:j+ws_w])
        return torch.stack(windows)

    def merge_windows(self, windows_res, original_shape, window_size, stride):
        B, SL, H, W, C = original_shape
        ws_h, ws_w = window_size
        s_h, s_w = stride
        merged = torch.zeros(B, SL, H, W, C, dtype=windows_res.dtype)
        count = torch.zeros(B, SL, H, W, C, dtype=windows_res.dtype)
        idx = 0
        for i in range(0, H-ws_h+1, s_h):
            for j in range(0, W-ws_w+1, s_w):
                merged[:,:,i:i+ws_h,j:j+ws_w] += windows_res[idx].to(merged.device)
                count[:,:,i:i+ws_h,j:j+ws_w] += 1
                idx += 1
        merged /= count
        return merged


    def crop_feats(self, x, use_crop_params=False):
        B, SL, H, W, c= x.shape
        x = x.permute(0, 1, 4, 2, 3)
        if not use_crop_params:
            self.batch_crops = [self.random_crop.get_params(torch.zeros(H, W),(16,32)) for _ in range(B)]
        cropped_tensor = torch.stack([torch.stack([TF.crop(x[b, s], *self.batch_crops[b]) for s in range(SL)]) for b in range(B)])
        cropped_tensor = cropped_tensor.permute(0, 1, 3, 4, 2)
        return cropped_tensor

    def extract_features(self, x, reshape=False):
        with torch.no_grad():
            if self.args.feature_extractor == 'dino':
                x = self.dino_v2.get_intermediate_layers(x,n=self.d_layers, reshape=reshape)
            elif self.args.feature_extractor == 'eva2-clip':
                x = self.eva2clip.forward_intermediates(x, indices=self.d_layers, output_fmt  = 'NLC', norm=True, intermediates_only=True)
            elif self.args.feature_extractor == 'sam':
                x = self.sam.forward_intermediates(x, indices=self.d_layers, norm=False, intermediates_only=True) # Norm is False to avoide neck layer that reduces feature_dim to 256. Also output is in NCHW format
                x = [einops.rearrange(f, 'b c h w -> b (h w) c') for f in x]
            if self.d_num_layers > 1:
                x = torch.cat(x,dim=-1)
            else:
                x = x[0]
        return x
            
    def get_mask_tokens(self, x, mode="arccos", mask_frames=1):
        B, sl, h, w, c = x.shape # x.shape [B,T,H,W,C]
        assert mask_frames <= sl
        if mode == "full_mask":
            if self.sequence_length == 7:
                assert mask_frames <= 3
            else:
                assert mask_frames == 1
            mask = torch.ones(B,mask_frames, h,w, dtype=torch.bool)
        else:
            r = torch.rand(B) # Batch size
            if mode == "linear":                # linear scheduler
                val_to_mask = r
            elif mode == "square":              # square scheduler
                val_to_mask = (r ** 2)
            elif mode == "cosine":              # cosine scheduler
                val_to_mask = torch.cos(r * math.pi * 0.5)
            elif mode == "arccos":              # arc cosine scheduler
                val_to_mask = torch.arccos(r) / (math.pi * 0.5)
            else:
                val_to_mask = None
            # Create a mask of size [Batch,1,Height, Width] for the last frame of each sequence
            mask = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)

        # Create the mask for all frames, by concatenating the mask for the last frame to a tensor of zeros(no mask) for first T-1 frames
        mask = torch.cat([torch.zeros(B,sl-mask_frames,h,w).bool(), mask], dim=1).to(x.device)

        if self.masking in ("half_half", "half_half_previous"):
            # Create the mask_tokens tensor
            mask_tokens = mask.unsqueeze(-1).float()*self.mask_vector.expand(B,sl,h,w,-1) + (~mask.unsqueeze(-1)).float()*self.unmask_vector.expand(B,sl,h,w,-1)
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            if self.masking == "half_half_previous":
                # Replace the embedded tokens at masked locations with the embedded tokens from the previous frames
                replace = torch.cat((torch.zeros((B,1,h,w,embedded_tokens.shape[-1]), dtype=embedded_tokens.dtype, device=embedded_tokens.device), embedded_tokens[:,:-1]), dim=1)
            else:
                # Replace the embedded tokens at masked locations with the replace vector
                replace = self.replace_vector.expand(B,sl,h,w,-1)
            embedded_tokens = torch.where(mask.unsqueeze(-1), replace, embedded_tokens)
            # Concatenate the masked tokens to the embedded tokens. Only take half of each to get the right hidden size
            final_tokens = torch.cat((embedded_tokens,mask_tokens), dim=-1)
        elif self.masking=="simple_replace":
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            # Replace the embedded tokens at masked locations with the replace vector
            final_tokens = torch.where(mask.unsqueeze(-1), self.replace_vector.expand(B,sl,h,w,-1), embedded_tokens)

        return final_tokens, mask
    
    def mask_pose_tokens_like_frames(self, pose_tokens, frame_mask):
        """
        pose_tokens: [B,T,1,C_in] (same feature space as x / PCA space if enabled)
        frame_mask:  [B,T] bool, True => frame is masked
        Returns:
            masked_pose_tokens: [B,T,1,H] in the SAME hidden space as masked_x fed to the transformer.
        """
        if pose_tokens is None:
            return None

        B, T, _, Cin = pose_tokens.shape
        assert frame_mask.dtype == torch.bool, "frame_mask must be boolean"

        if self.masking in ("half_half", "half_half_previous"):
            # embed half
            embedded = self.embed(pose_tokens)  # [B,T,1,H/2]
            m = frame_mask.unsqueeze(-1)        # [B,T,1]

            # --- build replacements/bias with correct rank [1,1,1,H/2] ---
            # vectors were defined as [1,1,1,1,H/2] for grid; reshape to [1,1,1,H/2]
            mask_vec   = self.mask_vector.reshape(1, 1, 1, -1)       # [1,1,1,H/2]
            unmask_vec = self.unmask_vector.reshape(1, 1, 1, -1)     # [1,1,1,H/2]

            if self.masking == "half_half_previous":
                # previous embedded (0 at t=0)
                prev = torch.cat(
                    (torch.zeros((B,1,1,embedded.shape[-1]), dtype=embedded.dtype, device=embedded.device),
                    embedded[:, :-1]),
                    dim=1
                )                                                    # [B,T,1,H/2]
                replace = prev
            else:  # "half_half"
                replace_full = self.replace_vector.reshape(1, 1, 1, -1)     # [1,1,1,H]
                replace = replace_full[..., :embedded.shape[-1]]            # [1,1,1,H/2]
                replace = replace.expand(B, T, 1, -1)                       # [B,T,1,H/2]

            # apply replacement on masked frames for the embedded half
            embedded = torch.where(m.unsqueeze(-1), replace, embedded)      # [B,T,1,H/2]

            # companion half constructed exactly like grid path
            mask_tokens_half = m.float().unsqueeze(-1) * mask_vec + (~m).float().unsqueeze(-1) * unmask_vec  # [B,T,1,H/2]

            masked_pose_tokens = torch.cat((embedded, mask_tokens_half), dim=-1)  # [B,T,1,H]
            return masked_pose_tokens

        elif self.masking == "simple_replace":
            embedded = self.embed(pose_tokens)                                # [B,T,1,H]
            m = frame_mask.unsqueeze(-1)                                      # [B,T,1]

            # reshape replace_vector from [1,1,1,1,H] -> [1,1,1,H]
            replace = self.replace_vector.reshape(1, 1, 1, -1).expand(B, T, 1, -1)  # [B,T,1,H]

            masked_pose_tokens = torch.where(m.unsqueeze(-1), replace, embedded)    # [B,T,1,H]
            return masked_pose_tokens

        else:
            raise ValueError(f"Unknown masking mode: {self.masking}")

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
        :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
        :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step+1)[1:]
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return
        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.shape[1]*self.shape[2])
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.shape[1] * self.shape[2]) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def downsample_feats(self, x):
        # Accepts shape [B*T,H,W,C]
        BT,H,W,C = x.shape # shape [8*5, 16, 32, 3072]
        emb_dim = self.feature_dim
        x = torch.movedim(x, -1, 1) # [B*T,C,H,W]
        features_list = [x[:,i*emb_dim:(i+1)*emb_dim] for i in range(self.d_num_layers)]
        features_list_down = [self.downsample_conv[i](x) for i, x in enumerate(features_list)]
        return torch.movedim(torch.cat(features_list_down, dim=1), 1, -1) # [B*T,H//2,W//2,C]

    def upsample_feats(self, x):
        # Accepts shape [B*T,H//2,W//2,C]
        BT,H,W,C = x.shape
        emb_dim = self.feature_dim
        x = torch.movedim(x, -1, 1) # [B*T,C//2,H//2,W]
        features_list = [x[:,i*emb_dim:(i+1)*emb_dim] for i in range(self.d_num_layers)]
        features_list_up = [self.upsample_conv[i](x) for i, x in enumerate(features_list)]
        return torch.movedim(torch.cat(features_list_up, dim=1), 1, -1) # [B*T,H,W,C]

    def pca_transform(self, x):
        BT, HW, C = x.shape
        x = (x - self.mean) / self.std
        x = x - self.pca_mean
        x_pca = torch.matmul(x, self.pca_components.T)
        return x_pca

    def pca_inverse_transform(self, x):
        B, T, H, W, C = x.shape
        x = torch.matmul(x, self.pca_components) + self.pca_mean
        x = x * self.std + self.mean
        return x

    def preprocess(self, x):
        """
        External mode:
        Accepts any of:
            [B, T, H, W, C]  (preferred)
            [B, T, C, H, W]  (will permute to [B,T,H,W,C])
            [B, T, HW, C]    (will reshape to [B,T,H,W,C] using args.feat_hw)
        Returns [B, T, H, W, C] in the (optional) PCA space if pca_ckpt is provided.
        """
        pose_tokens = None
        if self.args.feature_extractor == 'external':
            if x.dim() == 4 and x.shape[-1] == self.args.cached_feature_dim:
                # [B,T,HW,C] -> [B,T,H,W,C]
                B, T, HW, C = x.shape

                # Extract pose token
                if HW > self.args.feat_hw[0] * self.args.feat_hw[1]:
                    pose_tokens = x[:,:, -1, :].unsqueeze(2)  # [B,T,1,C]
                    x = x[:,:, :-1, :]  # [B,T,HW-1,C]
                    HW -= 1

                Hf, Wf = self.args.feat_hw
                x = x.view(B, T, Hf, Wf, C)
            else:
                raise ValueError(f"Unsupported external feature shape: {tuple(x.shape)}")

            if self.args.pca_ckpt:
                B, T, H, W, C = x.shape
                x = x.view(B*T, H*W, C)
                x = self.pca_transform(x)  # token-wise PCA
                x = x.view(B, T, H, W, -1)
            return x, pose_tokens

        # -------- original image-encoder path (kept intact if you still want it) --------
        B, T, C, H, W = x.shape
        x = x.flatten(end_dim=1)  # [B*T,C,H,W]
        x = self.extract_features(x)  # [B*T, H*W, C]
        if self.args.pca_ckpt:
            x = self.pca_transform(x)
        x = einops.rearrange(x, 'b (h w) c -> b h w c', h=H//self.patch_size, w=W//self.patch_size)
        x = x.unflatten(dim=0, sizes=(B, self.sequence_length))  # [B,T,H,W,C]
        return x

    def postprocess(self, x):
        if self.args.pca_ckpt:
            x = self.pca_inverse_transform(x)
        return x
    
    def calculate_loss(self, x_pred, x_target):
        if self.args.loss_type == "MSE":
            loss = F.mse_loss(x_pred, x_target)
        elif self.args.loss_type == "SmoothL1":
            loss = F.smooth_l1_loss(x_pred, x_target, beta=self.args.beta_smoothl1)
        elif self.args.loss_type == "L1":
            loss = F.l1_loss(x_pred, x_target)
        return loss

    def forward_loss(self, x_pred, x_target, mask):
        B, T, H, W, C = x_pred.shape
        x_target = x_target[mask] 
        x_pred = x_pred[mask] 
        return self.calculate_loss(x_pred, x_target)
        
    """def sample(self, x, sched_mode="arccos", step=15, mask_frames=1):
        self.maskvit.eval()
        with torch.no_grad():
            x, _ = self.preprocess(x)
            B, SL, H, W, C = x.shape
            if not self.args.sliding_window_inference:
                if self.args.crop_feats:
                    x = self.crop_feats(x)
                masked_soft_tokens, mask = self.get_mask_tokens(x, mode="full_mask",mask_frames=mask_frames)
                mask = mask.to(x.device)
                if self.args.single_step_sample_train or step==1:
                    if self.args.vis_attn:
                        _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                    else:
                        loss, final_tokens = self.forward(x, masked_soft_tokens, mask)
                else:
                    assert "Not implemented"
                prediction = self.postprocess(final_tokens)
            else:
                window_size = (16,32)
                stride = (16,32)
                x = self.sliding_window(x, window_size, stride)
                wins = []
                for i in range(x.shape[0]):
                    win = x[i]
                    masked_soft_tokens, mask = self.get_mask_tokens(win, mode="full_mask",mask_frames=mask_frames)
                    mask = mask.to(x.device)
                    if self.args.single_step_sample_train or step==1:
                        if self.args.vis_attn:
                            _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                        else:
                            loss, final_tokens = self.forward(win, masked_soft_tokens, mask)
                    else:
                        # Instantiate scheduler
                        if isinstance(sched_mode, str):  # Standard ones
                            scheduler = self.adap_sche(step, mode=sched_mode)
                        else:  # Custom one
                            scheduler = sched_mode
                        final_tokens, loss = self.oracle_sample(x, masked_soft_tokens, mask, scheduler, step, mask_frames)
                    prediction = self.postprocess(final_tokens)
                    wins.append(prediction)
                prediction = self.merge_windows(torch.stack(wins), (B, SL, H, W, 3328), window_size, stride).to(x.device)
            return prediction, loss
            # return prediction, loss, final_tokens"""
    def sample(self, x, sched_mode="arccos", step=15, mask_frames=1):
        self.maskvit.eval()
        with torch.no_grad():
            x, pose_tokens = self.preprocess(x)  # pose_tokens: [B,T,1,C] or None
            B, SL, H, W, C = x.shape

            if not self.args.sliding_window_inference:
                if self.args.crop_feats:
                    x = self.crop_feats(x)
                masked_soft_tokens, mask = self.get_mask_tokens(x, mode="full_mask", mask_frames=mask_frames)
                mask = mask.to(x.device)
                frame_mask = mask.any(dim=(2,3))
                masked_pose_tokens = self.mask_pose_tokens_like_frames(pose_tokens, frame_mask)

                if self.args.single_step_sample_train or step==1:
                    if self.args.vis_attn:
                        _, final_tokens, pose_pred, _ = self.forward(
                            x, masked_soft_tokens, mask,
                            pose_tokens_masked=masked_pose_tokens,
                            pose_tokens_target=pose_tokens
                        )
                    else:
                        loss, final_tokens, pose_pred = self.forward(
                            x, masked_soft_tokens, mask,
                            pose_tokens_masked=masked_pose_tokens,
                            pose_tokens_target=pose_tokens
                        )
                else:
                    assert "Not implemented"

                prediction = self.postprocess(final_tokens)
                return prediction, pose_pred, loss

            else:
                # (Keep pose None for now in sliding-window to keep it simple, or thread per-window similarly.)
                window_size = (16,32)
                stride = (16,32)
                x_w = self.sliding_window(x, window_size, stride)
                wins = []
                for i in range(x_w.shape[0]):
                    win = x_w[i]
                    masked_soft_tokens, mask = self.get_mask_tokens(win, mode="full_mask", mask_frames=mask_frames)
                    mask = mask.to(x.device)
                    if self.args.single_step_sample_train or step==1:
                        if self.args.vis_attn:
                            _, final_tokens_win, _, _ = self.forward(win, masked_soft_tokens, mask,
                                                                    pose_tokens_masked=None, pose_tokens_target=None)
                        else:
                            loss, final_tokens_win, _ = self.forward(win, masked_soft_tokens, mask,
                                                                    pose_tokens_masked=None, pose_tokens_target=None)
                    else:
                        assert "Not implemented"
                    wins.append(self.postprocess(final_tokens_win))
                prediction = self.merge_windows(torch.stack(wins), (B, SL, H, W, C), window_size, stride).to(x.device)
                return prediction, None, loss

    """def sample_unroll(self, x, gt_feats, sched_mode="arccos", step=15, mask_frames=1, unroll_steps=3, ):
        self.maskvit.eval()
        with torch.no_grad():
            x, _ = self.preprocess(x)
        B, SL, H, W, C = x.shape
        for i in range(unroll_steps):
            if not self.args.sliding_window_inference:
                masked_soft_tokens, mask = self.get_mask_tokens(x, mode="full_mask",mask_frames=mask_frames)
                mask = mask.to(x.device)
                if self.args.single_step_sample_train or step==1:
                    if self.args.vis_attn:
                        _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                    else:
                        loss, final_tokens= self.forward(x, masked_soft_tokens, mask)
                else:
                    assert "Not implemented"
                # x = self.postprocess(final_tokens)
            else:
                window_size = (16,32)
                stride = (16,32)
                x_s = self.sliding_window(x, window_size, stride)
                wins = []
                for i in range(x_s.shape[0]):
                    win = x_s[i]
                    masked_soft_tokens, mask = self.get_mask_tokens(win, mode="full_mask",mask_frames=mask_frames)
                    mask = mask.to(x.device)
                    if self.args.single_step_sample_train or step==1:
                        if self.args.vis_attn:
                            _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                        else:
                            loss, final_tokens_win = self.forward(win, masked_soft_tokens, mask)
                    wins.append(final_tokens_win)
                final_tokens = self.merge_windows(torch.stack(wins), (B, SL, H, W, 1152), window_size, stride).to(x.device)
            x[:,-1] = final_tokens[:,-1]
            x = torch.cat((x[:,1:], x[:,-1].unsqueeze(1)), dim=1) # Mayve also try torch.zeros instead of x[:,-1]
        prediction = self.postprocess(x)
        loss = self.calculate_loss(prediction[:,-1].flatten(end_dim=-2), gt_feats.flatten(end_dim=-2))
        # return prediction, loss, x
        return prediction, loss"""
    def sample_unroll(self, x, gt_feats, sched_mode="arccos", step=1, mask_frames=1, unroll_steps=3):
        """
        Autoregressive unroll:
        - keeps a running context for BOTH grid features and pose tokens
        - at each step masks the last mask_frames frames
        - predicts those frames, writes predictions back (grid + pose), shifts window
        - computes loss vs GT on the final last frame (grid + pose)

        Args:
        x:          input seq in the same format as training (images or cached feats)
        gt_feats:   GT for the last frame; accepts:
                    * [B, Hf*Wf+1, C]  (pose token at index 0 + grid tokens)
                    * [B, Hf, Wf, C]   (grid only)
                    * or a dict: {'grid': [B,Hf,Wf,C], 'pose': [B,1,C]}
        step:       leave at 1 (single-step sampler)
        Returns:
        pred_grid:  [B,T,Hf,Wf,C] after postprocess()
        pred_pose:  [B,T,1,C_pca] (pose tokens in the same feature space as model IO)
        loss:       grid loss (+ pose loss * weight) on final last frame
        """
        self.maskvit.eval()
        with torch.no_grad():
            grid_seq, pose_seq = self.preprocess(x)   # grid_seq: [B,T,Hf,Wf,Cp], pose_seq: [B,T,1,Cp] or None

        B, T, Hf, Wf, Cp = grid_seq.shape
        device = grid_seq.device

        # --- unroll autoregressively ---
        pose_last = None
        for _ in range(unroll_steps):
            # Build a mask that covers the last mask_frames frames spatially
            mask = torch.zeros(B, T, Hf, Wf, dtype=torch.bool, device=device)
            mask[:, -mask_frames:] = True

            # Grid masking (re-using your masking scheme to construct transformer inputs)
            masked_grid, _ = self.get_mask_tokens(grid_seq, mode="full_mask", mask_frames=mask_frames)  # [B,T,Hf,Wf,HID]

            # Pose masking mirrors the frame mask
            frame_mask = mask.any(dim=(2,3))  # [B,T]
            masked_pose = self.mask_pose_tokens_like_frames(pose_seq, frame_mask) if pose_seq is not None else None

            # Forward with NO teacher forcing on pose (targets=None), we just want predictions
            if self.args.vis_attn:
                _, x_pred, pose_pred, _ = self.forward(
                    x=grid_seq,
                    masked_x=masked_grid,
                    mask=mask,
                    pose_tokens_masked=masked_pose,
                    pose_tokens_target=None
                )
            else:
                _, x_pred, pose_pred = self.forward(
                    x=grid_seq,
                    masked_x=masked_grid,
                    mask=mask,
                    pose_tokens_masked=masked_pose,
                    pose_tokens_target=None
                )

            # Write back predictions for the masked tail (only the last frame is needed for AR)
            grid_seq[:, -1] = x_pred[:, -1]
            if pose_pred is not None and pose_seq is not None:
                pose_seq[:, -1] = pose_pred[:, -1]
                pose_last = pose_pred

            # Shift the window (drop first, append the newly predicted last)
            grid_seq = torch.cat((grid_seq[:, 1:], grid_seq[:, -1:].clone()), dim=1)
            if pose_seq is not None:
                pose_seq = torch.cat((pose_seq[:, 1:], pose_seq[:, -1:].clone()), dim=1)

        # --- finalize predictions in the head's original feature space ---
        pred_seq_grid = self.postprocess(grid_seq)  # [B,T,Hf,Wf,C]
        pred_last_grid = pred_seq_grid[:, -1]       # [B,Hf,Wf,C]
        pred_last_pose = None
        if pose_seq is not None:
            pred_last_pose = pose_seq[:, -1]        # [B,1,Cp]  (still in PCA/model space by design)

        # --- prepare GT for loss (flexibly accept shapes) ---
        # Unified: target_grid [B,Hf,Wf,C], target_pose [B,1,C] or None
        target_grid, target_pose = None, None
        if isinstance(gt_feats, dict):
            target_grid = gt_feats.get("grid", None)
            target_pose = gt_feats.get("pose", None)
        else:
            tgt = gt_feats
            if tgt.dim() == 3:  # [B, S(+1), C]
                S = tgt.shape[1]
                if S == Hf*Wf + 1:
                    target_pose = tgt[:, :1, :]                       # [B,1,C]
                    target_grid = tgt[:, 1:, :].view(B, Hf, Wf, -1)   # [B,Hf,Wf,C]
                elif S == Hf*Wf:
                    target_grid = tgt.view(B, Hf, Wf, -1)
                else:
                    raise ValueError(f"Unexpected gt_feats length {S}; expected {Hf*Wf} or {Hf*Wf+1}.")
            elif tgt.dim() == 4:  # [B,Hf,Wf,C]
                target_grid = tgt
            else:
                raise ValueError(f"Unsupported gt_feats shape: {tuple(tgt.shape)}")

        # --- compute loss on final frame (grid + optional pose) ---
        # Use the same criterion as training, but no masking (we want full-frame error)
        grid_loss = self.calculate_loss(
            pred_last_grid.reshape(B, -1, pred_last_grid.shape[-1]),
            target_grid.reshape(B, -1, target_grid.shape[-1])
        )

        pose_loss = torch.tensor(0.0, device=device)
        if (pred_last_pose is not None) and (target_pose is not None):
            pose_loss = self.calculate_loss(
                pred_last_pose.squeeze(1),   # [B,C]
                target_pose.squeeze(1)       # [B,C]
            )

        loss = grid_loss + self.pose_loss_weight * pose_loss
        return pred_seq_grid, pose_seq, loss

    def forward(self, x, masked_x, mask=None, pose_tokens_masked=None, pose_tokens_target=None, return_both_losses=False):
        if self.args.vis_attn:
            (x_pred, pose_pred), attn = self.maskvit(masked_x, pose_token=pose_tokens_masked, return_attn=True)
        else:
            x_pred, pose_pred = self.maskvit(masked_x, pose_token=pose_tokens_masked)

        # x_pred comes from MaskTransformerWithPose as either 3D [B, SL*H*W, C] (old path) or 5D [B, SL, H, W, C] (new path).
        if x_pred.dim() == 3:
            x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c',
                                    sl=self.shape[0], h=self.shape[1], w=self.shape[2])
        elif x_pred.dim() == 5:
            pass  # already [B, SL, H, W, C]
        else:
            raise RuntimeError(f"Unexpected x_pred dims: {x_pred.shape}")

        x_pred = self.activation(x_pred)

        # ---- grid loss on masked spatial tokens ----
        grid_loss = self.forward_loss(x_pred=x_pred, x_target=x, mask=mask)

        # ---- pose loss on masked frames only ----
        pose_loss = torch.tensor(0.0, device=x_pred.device)
        if (pose_pred is not None) and (pose_tokens_target is not None) and (mask is not None):
            frame_mask = mask.any(dim=(2,3))  # [B,T]
            if frame_mask.any():
                pred_m = pose_pred[frame_mask].squeeze(1)         # [Nm, C]
                targ_m = pose_tokens_target[frame_mask].squeeze(1) # [Nm, C]
                pose_loss = self.calculate_loss(pred_m, targ_m)

        loss_total = grid_loss + self.pose_loss_weight * pose_loss
        if return_both_losses:
            if self.args.vis_attn:
                return loss_total, grid_loss, pose_loss, x_pred, pose_pred, attn
            else:
                return loss_total, grid_loss, pose_loss, x_pred, pose_pred
        else:
            if self.args.vis_attn:
                return loss_total, x_pred, pose_pred, attn
            else:
                return loss_total, x_pred, pose_pred
    def training_step(self, x, batch_idx):
        B = x.shape[0]
        x, pose_tokens = self.preprocess(x)  # pose_tokens: [B,T,1,C] or None

        # Mask grid
        if self.args.crop_feats:
            x = self.crop_feats(x)
        if self.sequence_length == 7:
            train_mask_frames = torch.randint(1, 4, (1,)) if self.training else 3
        else:
            train_mask_frames = self.train_mask_frames if self.training else 1
        masked_x, mask = self.get_mask_tokens(x, mode=self.train_mask_mode, mask_frames=train_mask_frames)
        mask = mask.to(x.device)  # [B,T,H,W] True where masked

        # Build pose masked tokens using frame mask (any masked pixel => masked frame)
        frame_mask = mask.any(dim=(2,3))  # [B,T] bool
        masked_pose_tokens = self.mask_pose_tokens_like_frames(pose_tokens, frame_mask)  # [B,T,1,HID] or None

        # Forward (pose participates in model + loss)
        if self.args.vis_attn:
            loss, grid_loss, pose_loss, _, _, _ = self.forward(
                x, masked_x, mask,
                pose_tokens_masked=masked_pose_tokens,
                pose_tokens_target=pose_tokens,
                return_both_losses=True
            )
        else:
            loss, grid_loss, pose_loss, _, _ = self.forward(
                x, masked_x, mask,
                pose_tokens_masked=masked_pose_tokens,
                pose_tokens_target=pose_tokens,
                return_both_losses=True
            )

        # Logs
        self.log("Train/loss", loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        self.log("Train/grid_loss", grid_loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        self.log("Train/pose_loss", pose_loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("Train/lr", lr, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        mask_ratio = mask[:,-1].float().mean().item() * 100
        self.log("Train/mask_ratio", mask_ratio, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss

    @torch.no_grad()
    def _feature_fit_metrics(self, pred, target, mask):
        # pred/target: [B,T,H,W,C]; mask: [B,T,H,W] True where masked
        pred_m = pred[mask]
        targ_m = target[mask]
        mse = F.mse_loss(pred_m, targ_m)
        mae = F.l1_loss(pred_m, targ_m)
        cos = F.cosine_similarity(pred_m, targ_m, dim=-1).mean()
        self.ff_mse.update(mse)
        self.ff_mae.update(mae)
        self.ff_cos.update(cos)
        return mse, mae, cos
    
    @torch.no_grad()
    def _feature_fit_metrics_pose(self, pred_pose, target_pose, mask):
        # pred/target: [B,T,1,C]; mask: [B,T] True where masked
        pred_m = pred_pose[mask]
        targ_m = target_pose[mask]
        mse = F.mse_loss(pred_m, targ_m)
        mae = F.l1_loss(pred_m, targ_m)
        cos = F.cosine_similarity(pred_m, targ_m, dim=-1).mean()
        self.pose_ff_mse.update(mse)
        self.pose_ff_mae.update(mae)
        self.pose_ff_cos.update(cos)
        return mse, mae, cos
    
    @torch.no_grad()
    def _run_head_on_seq(self, feats_bt_hw_c: torch.Tensor, pose_tokens_bt_1_c: torch.Tensor | None,
                        per_layer_dims: tuple[int,int,int,int] | None = (1024, 768, 768, 768)):
        """
        Run CUT3R head on EVERY frame. Inputs must already be in the head's feature space
        (i.e., AFTER self.postprocess()).

        Args:
            feats_bt_hw_c: [B, T, Hf, Wf, C]
            pose_tokens_bt_1_c: [B, T, 1, C] or None
            per_layer_dims: channel split for the 4 layer-blobs concatenated in C

        Returns:
            outs: list of length T; each item has 'camera_pose': [B,7]
        """
        assert hasattr(self, "head") and (self.head is not None), "CUT3R head not loaded."
        self.head.eval()

        B, T, Hf, Wf, C = feats_bt_hw_c.shape
        if pose_tokens_bt_1_c is None:
            pose_tokens_bt_1_c = torch.zeros(B, T, 1, C, device=feats_bt_hw_c.device, dtype=feats_bt_hw_c.dtype)

        # fall back to equal quarters if the provided split doesn't match C
        if (per_layer_dims is None) or (sum(per_layer_dims) != C):
            q = C // 4
            per_layer_dims = (q, q, q, C - 3*q)

        outs = []
        for t in range(T):
            x_t = feats_bt_hw_c[:, t]      # [B,Hf,Wf,C]
            p_t = pose_tokens_bt_1_c[:, t] # [B,1,C]
            head_in_t = self._create_head_input(x_t, p_t, per_layer_dims=per_layer_dims)
            out_t = self.head(head_in_t, img_info=self.head_img_hw)
            outs.append(out_t)
        return outs
    
    def validation_step(self, batch, batch_idx):
        if self.args.eval_mode: #or self._do_eval3d_this_epoch:
            return self.evaluation_step(batch, batch_idx)
        
        if isinstance(batch, torch.Tensor):
            feats = batch
            depth = None
            pose  = None
        else:
            feats, depth, pose = batch

        B = feats.shape[0]
        Hf, Wf = self.args.feat_hw

        # --- Feature-fit metrics on masked tokens (unchanged except pose threading) ---
        x, pose_tokens = self.preprocess(feats)
        if self.args.crop_feats:
            x = self.crop_feats(x)
        mask_frames = self.train_mask_frames if self.sequence_length != 7 else 1
        masked_x, mask = self.get_mask_tokens(x, mode=self.train_mask_mode, mask_frames=mask_frames)
        frame_mask = mask.any(dim=(2,3))
        masked_pose_tokens = self.mask_pose_tokens_like_frames(pose_tokens, frame_mask)

        total_loss, x_pred, pose_pred = self.forward(
            x, masked_x, mask,
            pose_tokens_masked=masked_pose_tokens,
            pose_tokens_target=pose_tokens
        )
        self._feature_fit_metrics(x_pred, x, mask)
        self._feature_fit_metrics_pose(pose_pred, pose_tokens, frame_mask)

        # per-batch loss for epoch aggregation
        self.mean_metric.update(total_loss)
        self.log('val/loss', total_loss, prog_bar=True, batch_size=B, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def sample_baseline_copy_last(self, feats, gt_feats, last_context_idx=0):
        # feats: [B, T, H*W+1, C]
        # gt_feats: [B, 1, H*W+1, C]
        feats_pred = feats[:, last_context_idx].squeeze(1)
        x_pred = feats_pred[:, 1:]
        pose_token_pred = feats_pred[:, :1]

        x_gt = gt_feats[:, -1, 1:]
        pose_token_gt = gt_feats[:, -1, :1]

        grid_loss = self.calculate_loss(x_pred, x_gt)
        pose_loss = self.calculate_loss(pose_token_pred, pose_token_gt)
        loss = grid_loss + self.pose_loss_weight * pose_loss

        # Return sequence with the last feature/pose being predicted with copy last
        gt_feats[:, -1, 1:] = x_pred
        gt_feats[:, -1, :1] = pose_token_pred

        return gt_feats[:,:, 1:], gt_feats[:, :, :1], loss  # Return grid, pose token, loss

    def evaluation_step(self, batch, batch_idx):
        feats, depth, pose = batch  # [B,T,H*W+1,C], [B,T,H,W], [B,T,4,4]

        # --- Forecast samples (unchanged) ---
        if self.args.evaluate_baseline:
            samples_grid, samples_pose, loss = self.sample_baseline_copy_last(feats, gt_feats=feats.clone())
        else:
            if self.args.eval_midterm:
                samples_grid, samples_pose, loss = self.sample_unroll(
                    feats, gt_feats=feats[:, -1],
                    sched_mode=self.train_mask_mode, step=self.args.step, unroll_steps=3
                )
            else:
                samples_grid, samples_pose, loss = self.sample(
                    feats, sched_mode=self.train_mask_mode, step=self.args.step
                )

        #print("SHAPES:", samples_grid.shape, samples_pose.shape)

        # after obtaining loss from sample/sample_unroll/baseline
        B = feats.shape[0]
        # keep a scalar tensor
        loss_val = loss if torch.is_tensor(loss) else torch.tensor(loss, device=self.device, dtype=torch.float32)

        # log per-epoch aggregation and update running mean so val/mean_loss is finite
        self.log('val/loss', loss_val, prog_bar=True, batch_size=B, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.mean_metric.update(loss_val.detach())

        # ------------ Depth on TARGET ------------
        pred_last = samples_grid[:, -1]       # [B,Hf,Wf,4D]
        pose_pred_last = samples_pose[:, -1]  # [B,1,4D]
        per_layer_dims = (1024, 768, 768, 768)

        head_in_tgt = self._create_head_input(pred_last, pose_pred_last, per_layer_dims=per_layer_dims)
        with torch.no_grad():
            head_out_tgt = self.head(head_in_tgt, img_info=self.head_img_hw)
            #torch.save(head_in_tgt, "/workspace/demo_head_input.pt")
            #assert False, "Saved head input for demo."
            

        gt_depth_t = depth[:, -1].to(self.device)
        self._update_depth_metrics_from_head(head_out_tgt, gt_depth_t)

        # ------------ Pose metrics (GLOBAL Sim(3) over many frames) ------------
        # Use the GT features for ALL frames, run head per frame, then swap-in the forecast for the last.
        x_gt_hw, pose_tokens = self.preprocess(feats)   # [B,T,Hf,Wf,C_pca], [B,T,1,C_pca] or None
        x_gt_hw = self.postprocess(x_gt_hw)             # back to head feature space (for the head)

        # run head on every frame to get per-frame camera_pose predictions
        outs_all = self._run_head_on_seq(x_gt_hw, pose_tokens, per_layer_dims=per_layer_dims)  # list length T

        # replace the last (target) with your actual forecast head output
        outs_all[-1] = head_out_tgt

        # use frames 0..T-1 for alignment (or pass use_until_idx=t for causal-at-t)
        self._update_pose_metrics_global(
            head_out_seq=outs_all,
            gt_pose_seq=pose.to(self.device),  # [B,T,4,4]
            bi=0,
            use_until_idx=3,                # None => full sequence; set to T-1 or t for causal eval
            with_scale=True,
            align_rot=True,
        )

    def on_validation_epoch_start(self):
        n = getattr(self.args, "eval3d_every_n_epochs", 0) or 0
        self._do_eval3d_this_epoch = (n > 0 and (self.current_epoch % n == 0))
    
    def on_validation_epoch_end(self):
        mean_loss = self.mean_metric.compute()
        logs = {
            'val/mean_loss': mean_loss,
            'val/loss': mean_loss,
            'val/ff_mse': self.ff_mse.compute(),
            'val/ff_mae': self.ff_mae.compute(),
            'val/ff_cos': self.ff_cos.compute(),
        }

        # If you evaluated 3D, include those:
        if self.args.eval_mode or self._do_eval3d_this_epoch:
            logs['val/depth_absrel'] = self.depth_absrel_m.compute()
            logs['val/depth_delta1'] = self.depth_delta1_m.compute()
       
            logs['val/pose_ATE_m']     = self.pose_ate_m.compute()
            logs['val/pose_RPE_t_m']   = self.pose_rpe_t_m.compute()
            logs['val/pose_RPE_r_deg'] = self.pose_rpe_r_m.compute()

            self.depth_absrel_m.reset(); self.depth_delta1_m.reset()
            self.pose_ate_m.reset(); self.pose_rpe_t_m.reset(); self.pose_rpe_r_m.reset()

        self.log_dict(logs, prog_bar=True, logger=True)

        # Reset
        self.mean_metric.reset()
        self.ff_mse.reset(); self.ff_mae.reset(); self.ff_cos.reset()
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]

    def _create_head_input(self, x_last: torch.Tensor, pose_last:torch.Tensor, per_layer_dims=None, add_dummy_pose_token_on_last=False):
        """
        x_last: [B, Hf, Wf, 4D] or [B, S, 4D]
        per_layer_dims: optional (d0, d1, d2, d3). If None, we infer equal quarters.
        Returns: [l0, l1, l2, l3] each [B, S, d_i] (and we can append a CLS to l3).
        """
        if x_last.dim() == 4:  # [B,Hf,Wf,C]
            B, Hf, Wf, C = x_last.shape
            S = Hf * Wf
            x_flat = x_last.view(B, S, C)
        elif x_last.dim() == 3:  # [B,S,C]
            B, S, C = x_last.shape
            x_flat = x_last
        else:
            raise ValueError(f"Unexpected feat shape {tuple(x_last.shape)}")

        
        d0, d1, d2, d3 = per_layer_dims
        assert d0 + d1 + d2 + d3 == C, f"Channel split mismatch: {C=} vs {d0,d1,d2,d3}"

        # Layer features
        l0 = x_flat[:, :, 0:d0]
        l1 = x_flat[:, :, d0:d0+d1]
        l2 = x_flat[:, :, d0+d1:d0+d1+d2]
        l3 = x_flat[:, :, d0+d1+d2:d0+d1+d2+d3]

        # Pose token
        p3 = pose_last[:, :, d0+d1+d2:d0+d1+d2+d3]

        if add_dummy_pose_token_on_last:
            # prepend a dummy pose token to the last layer -> [B, S+1, d3]
            dummy_pose_token = torch.zeros(l3.size(0), 1, l3.size(2), device=l3.device, dtype=l3.dtype)
            l3 = torch.cat([dummy_pose_token, l3], dim=1)
        else:
            l3 = torch.cat([p3, l3], dim=1)
        return [l0, l1, l2, l3]

    @torch.no_grad()
    def _update_depth_metrics_from_head(self, head_out, gt_depth_t):
        if head_out is None or gt_depth_t is None:
            return

        pts3d = head_out.get("pts3d_in_self_view", None)  # [B,H,W,3]
        if pts3d is None:
            return

        pred_depth = pts3d[..., 2]

        # Resize prediction to GT resolution if needed
        if pred_depth.shape[-2:] != gt_depth_t.shape[-2:]:
            pred_depth = torch.nn.functional.interpolate(
                pred_depth.unsqueeze(1), size=gt_depth_t.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)

        # Build mask
        valid = gt_depth_t > 0
        max_d = getattr(self.args, "depth_eval_max_depth", None)
        if max_d is not None:
            valid = valid & (gt_depth_t < float(max_d))

        if valid.sum() == 0:
            return

        pd = pred_depth[valid].float()
        gt = gt_depth_t[valid].float()

        # ---- alignment (match your authors-style) ----
        align_mode     = getattr(self.args, "depth_eval_align_mode", "scale&shift")
        post_clip_max  = getattr(self.args, "depth_post_clip_max", None)
        pd_aligned = _align_depth(pd, gt, mode=align_mode, post_clip_max=post_clip_max)

        # ---- metrics ----
        absrel = torch.mean(torch.abs(gt - pd_aligned) / gt)
        ratio  = torch.maximum(
            pd_aligned / gt,
            gt / pd_aligned.clamp_min(1e-8)
        )
        delta1 = (ratio < 1.25).float().mean()

        self.depth_absrel_m.update(absrel)
        self.depth_delta1_m.update(delta1)

    @torch.no_grad()
    def _update_pose_metrics_from_head(
        self,
        head_out_tgt,
        gt_pose_last: torch.Tensor | None = None,
        gt_pose_t: torch.Tensor | None = None,
        head_out_ctx=None,
        bi: int = 0,
        align_rot: bool = True,
    ):
        """
        Two-frame pose metrics using CUT3R head outputs.
        - Runs ATE on target translation (after 2-frame Sim(3) alignment)
        - Runs 1-step RPE (translation in m, rotation in deg) from last->target

        Compatibility:
        - If only head_out_tgt is given (old path), this is a no-op for pose metrics.
        - If any required piece is missing, gracefully skip.

        Args:
            head_out_tgt: dict from CUT3R head for target frame (must contain 'camera_pose': [B,7])
            head_out_ctx: dict from CUT3R head for last context frame (must contain 'camera_pose': [B,7])
            gt_pose_last: [B,4,4] GT last context pose
            gt_pose_t:    [B,4,4] GT target pose
            bi:           int sample index (avoid mixing Sim(3) across items)
            align_rot:    if False, Sim(3) alignment won't rotate local camera frames (rotation error is "model-only")
        """
        key = "camera_pose"
        if (head_out_tgt is None) or (key not in head_out_tgt) or (head_out_tgt[key] is None):
            return
        if (head_out_ctx is None) or (key not in head_out_ctx) or (head_out_ctx[key] is None):
            return
        if gt_pose_last is None or gt_pose_t is None:
            return

        # slice sample bi
        pred_tgt_q7 = head_out_tgt[key][bi:bi+1]  # [1,7]
        pred_ctx_q7 = head_out_ctx[key][bi:bi+1]  # [1,7]
        if pred_tgt_q7.numel() == 0 or pred_ctx_q7.numel() == 0:
            return

        # to 4x4
        T_pred_last = quat7_to_mat44(pred_ctx_q7).detach().cpu().numpy()[0]
        T_pred_tgt  = quat7_to_mat44(pred_tgt_q7).detach().cpu().numpy()[0]

        # GT 4x4
        T_gt_last = ensure_4x4(gt_pose_last[bi])
        T_gt_tgt  = ensure_4x4(gt_pose_t[bi])

        # metrics
        ate, rpe_t, rpe_r = one_step_pose_metrics(
            T_gt_last, T_gt_tgt, T_pred_last, T_pred_tgt, align_rot=align_rot
        )

        # log
        self.pose_ate_m.update(torch.tensor(ate, device=self.device))
        self.pose_rpe_t_m.update(torch.tensor(rpe_t, device=self.device))
        self.pose_rpe_r_m.update(torch.tensor(rpe_r, device=self.device))
    
    @torch.no_grad()
    def _update_pose_metrics_global(
        self,
        head_out_seq: list,           # list length T, each has 'camera_pose': [B,7]
        gt_pose_seq: torch.Tensor,    # [B,T,4,4] GT c2w
        bi: int = 0,
        use_until_idx: int | None = None,  # last index to use (None => last)
        with_scale: bool = True,
        align_rot: bool = True,
    ):
        """
        Estimate ONE Sim(3) from MANY frames (0..t), apply it to predictions, then:
        - ATE at time t (translation-only)
        - 1-step RPE at t-1 -> t

        Classic ATE: translation error only; rotation influences only the alignment.
        """
        key = "camera_pose"
        Ttot = len(head_out_seq)
        assert Ttot >= 2, "Need at least two frames for meaningful metrics."
        t_last = (Ttot - 1) if (use_until_idx is None) else int(use_until_idx)
        assert 1 <= t_last < Ttot, "use_until_idx must be in [1..T-1]"

        # ---------- predicted 4x4 for frames [0..t_last] ----------
        T_pred_seq = []
        for t in range(t_last + 1):
            q7 = head_out_seq[t].get(key, None)
            if q7 is None or q7.shape[0] <= bi:
                return
            # If your head outputs w2c instead of c2w, flip this line to np.linalg.inv(...).
            T_pred_seq.append(quat7_to_mat44(q7[bi:bi+1]).detach().cpu().numpy()[0])

        # ---------- GT 4x4 for frames [0..t_last] ----------
        T_gt_seq = [ensure_4x4(gt_pose_seq[bi, t]) for t in range(t_last + 1)]

        # ---------- Sim(3) from MANY frames (positions only) ----------
        P_pred = np.stack([T[:3, 3] for T in T_pred_seq], 0)
        P_gt   = np.stack([T[:3, 3] for T in T_gt_seq],   0)

        if P_pred.shape[0] >= 3:
            Rsim, ssim, tsim = umeyama_sim3(P_pred, P_gt, with_scale=with_scale)
        else:
            Rsim, ssim, tsim = sim3_from_two_frames(T_gt_seq[0], T_gt_seq[-1], T_pred_seq[0], T_pred_seq[-1], align_rot=align_rot)

        # ---------- apply Sim(3) ----------
        T_pred_al = [apply_sim3_pose(T, Rsim, ssim, tsim, align_rot=align_rot) for T in T_pred_seq]

        # ---------- ATE at t ----------
        T_gt_t = ensure_4x4(T_gt_seq[t_last])
        T_pr_t = ensure_4x4(T_pred_al[t_last])
        ate = float(np.linalg.norm(T_pr_t[:3, 3] - T_gt_t[:3, 3]))

        # ---------- 1-step RPE at t-1 -> t ----------
        T_gt_prev = ensure_4x4(T_gt_seq[t_last - 1])
        dT_gt   = np.linalg.inv(T_gt_prev) @ T_gt_t
        dT_pred = np.linalg.inv(ensure_4x4(T_pred_al[t_last - 1])) @ T_pr_t
        dT_err  = np.linalg.inv(dT_gt) @ dT_pred
        rpe_t = trans_norm(dT_err)
        rpe_r = rot_angle_deg(dT_err)

        # ---------- log ----------
        self.pose_ate_m.update(torch.tensor(ate, device=self.device))
        self.pose_rpe_t_m.update(torch.tensor(rpe_t, device=self.device))
        self.pose_rpe_r_m.update(torch.tensor(rpe_r, device=self.device))

        