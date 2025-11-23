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
from src.attention_masked import MaskTransformer
import math
from dpt import DPTHead
import timm
import torch.nn.functional as F

sys.path.append("/workspace/CUT3R")
from eval.video_depth.tools import depth_evaluation


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
            self.head, head_meta = load_exported_head("/workspace/CUT3R/exported_cut3r_head")
            self.head_img_hw = (224,224)
            self._do_eval3d_this_epoch = False

            self.context_len = 4                        # frames 0..3
            self.target_t    = self.sequence_length - 1 # 4 (5th frame)
            self.unroll_steps = getattr(args, "unroll_steps", 3)

            # horizon frame index (in the original 7-frame GT sequence) for midterm eval
            self.horizon_idx_unroll = self.context_len - 1 + self.unroll_steps  # 3 + 3 = 6

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
            
        self.maskvit = MaskTransformer(shape=self.shape, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, depth=self.layers,
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
        
        # Necessary for evaluation
        self.mean_metric = MeanMetric()

        # feature recon metrics
        self.ff_mse = MeanMetric()
        self.ff_mae = MeanMetric()
        self.ff_cos = MeanMetric()

        # 3d metrics
        self.depth_absrel_m = MeanMetric()
        self.depth_delta1_m = MeanMetric()

        # pose metrics
        #self.pose_ate_m = MeanMetric()
        #self.pose_rpe_t_m = MeanMetric()
        #self.pose_rpe_r_m = MeanMetric()

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
            x: [B, T, Hf*Wf+1, C]  (pose token + grid tokens)
        Non-external mode:
            x: [B, T, C, H, W]

        Always returns: x of shape [B, sequence_length, Hf, Wf, C_pca?]
        """
        if self.args.feature_extractor == 'external':
            # x: [B, T, S, C] with S = Hf*Wf+1 (pose + grid)
            if x.dim() != 4:
                raise ValueError(f"Unsupported external feature shape: {tuple(x.shape)}")
            B, T, S, C = x.shape
            Hf, Wf = self.args.feat_hw
            assert S == Hf * Wf + 1, f"Expected S = Hf*Wf+1 = {Hf*Wf+1}, got {S}"

            # --- correct pose-token filtering ---
            # pose_tokens = x[:, :, 0:1, :]   # keep if you ever need them
            x = x[:, :, 1:, :]               # drop pose token -> [B, T, Hf*Wf, C]

            # enforce model sequence length (5) on time dimension
            assert T >= self.sequence_length, f"Got T={T}, need at least {self.sequence_length}"
            x = x[:, :self.sequence_length]  # [B, seq_len, Hf*Wf, C]

            # reshape tokens to grid
            x = x.view(B, self.sequence_length, Hf, Wf, C)  # [B, seq_len, Hf, Wf, C]

            # optional PCA
            if self.args.pca_ckpt:
                B_, T_, H, W, C_ = x.shape
                x = x.view(B_ * T_, H * W, C_)
                x = self.pca_transform(x)
                x = x.view(B_, T_, H, W, -1)

            return x

        # ---------------- non-external path unchanged (just enforce seq_len) ----------------
        B, T, C, H, W = x.shape
        assert T >= self.sequence_length
        x = x[:, :self.sequence_length]

        x = x.flatten(end_dim=1)  # [B*seq_len, C, H, W]
        x = self.extract_features(x)  # [B*seq_len, Hf*Wf, C_raw]
        if self.args.pca_ckpt:
            x = self.pca_transform(x)
        x = einops.rearrange(
            x, 'b (h w) c -> b h w c',
            h=H // self.patch_size, w=W // self.patch_size
        )
        x = x.unflatten(dim=0, sizes=(B, self.sequence_length))
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
        
    def sample(self, x, sched_mode="arccos", step=15, mask_frames=1):
        self.maskvit.eval()
        with torch.no_grad():
            x = self.preprocess(x)
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
                prediction = self.merge_windows(torch.stack(wins), (B, SL, H, W, 3072), window_size, stride).to(x.device)
            return prediction, loss
            # return prediction, loss, final_tokens

    def sample_unroll(self, x, gt_feats, sched_mode="arccos", step=15, mask_frames=1, unroll_steps=3):
        """
        Autoregressive rollout:

        x:      raw input features (same as training_step), with at least sequence_length frames.
        gt_feats: GT features for the horizon frame you care about (e.g. feats[:, 6]).

        We:
        - preprocess to a sequence of length sequence_length (5),
        - run unroll_steps iterations:
            each iteration masks the last frame, predicts it, writes prediction back in,
            then shifts the sequence so the prediction becomes part of the context,
        - postprocess and compute loss on the final last frame vs gt_feats.
        """
        self.maskvit.eval()
        with torch.no_grad():
            # This uses your new preprocess which slices to sequence_length
            x = self.preprocess(x)  # [B, seq_len, H, W, C]
        B, SL, H, W, C = x.shape
        assert SL == self.sequence_length, f"Expected seq_len {self.sequence_length}, got {SL}"

        for i in range(unroll_steps):
            if not self.args.sliding_window_inference:
                # Mask ONLY the last frame (full_mask on that frame)
                masked_soft_tokens, mask = self.get_mask_tokens(
                    x, mode="full_mask", mask_frames=mask_frames
                )
                mask = mask.to(x.device)

                if self.args.single_step_sample_train or step == 1:
                    if self.args.vis_attn:
                        _, final_tokens, attn_weights = self.forward(x, masked_soft_tokens, mask)
                    else:
                        loss_step, final_tokens = self.forward(x, masked_soft_tokens, mask)
                else:
                    assert False, "Multi-step sched_mode not implemented for unroll"

                # Overwrite the last frame in-place with the prediction
                x[:, -1] = final_tokens[:, -1]

                # Slide window: drop first, append predicted last as new last context+target
                x = torch.cat((x[:, 1:], x[:, -1:].clone()), dim=1)

            else:
                # Only if you actually need sliding_window_inference here.
                window_size = (16, 32)
                stride = (16, 32)
                x_s = self.sliding_window(x, window_size, stride)
                wins = []
                for j in range(x_s.shape[0]):
                    win = x_s[j]
                    masked_soft_tokens, mask = self.get_mask_tokens(
                        win, mode="full_mask", mask_frames=mask_frames
                    )
                    mask = mask.to(x.device)
                    if self.args.single_step_sample_train or step == 1:
                        if self.args.vis_attn:
                            _, final_tokens, attn_weights = self.forward(win, masked_soft_tokens, mask)
                        else:
                            loss_step, final_tokens_win = self.forward(win, masked_soft_tokens, mask)
                    wins.append(final_tokens_win)

                final_tokens = self.merge_windows(
                    torch.stack(wins), (B, SL, H, W, final_tokens_win.shape[-1]),
                    window_size, stride
                ).to(x.device)

                x[:, -1] = final_tokens[:, -1]
                x = torch.cat((x[:, 1:], x[:, -1:].clone()), dim=1)

        # After unroll, interpret the LAST frame as the horizon prediction
        prediction = self.postprocess(x)      # back to original feature space
        pred_horizon = prediction[:, -1]      # [B, H, W, C] at horizon

        # gt_feats: [B, H, W, C] for the horizon frame (e.g. feats[:, 6] after reshape)
        loss = self.calculate_loss(
            pred_horizon.flatten(end_dim=-2),
            gt_feats.flatten(end_dim=-2)
        )
        return prediction, loss

    def forward(self, x, masked_x, mask=None):
        if self.args.vis_attn:
            x_pred, attn = self.maskvit(masked_x,return_attn=True)
        else:
            x_pred, = self.maskvit(masked_x)
        x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c',sl=self.shape[0], h=self.shape[1], w=self.shape[2])
        x_pred = self.activation(x_pred)
        # if self.args.predict_residuals:
        #     x_prev = torch.cat([torch.zeros(B,1,H,W,C).to(x.device), x[:,:-1]], dim=1)
        #     x_pred = x_pred + x_prev
        loss = self.forward_loss(x_pred=x_pred, x_target=x, mask=mask)
        if self.args.vis_attn:
            return loss, x_pred, attn
        else:
            return loss, x_pred

    def training_step(self, x, batch_idx):
        B = x.shape[0]

        x = self.preprocess(x)

        # Mask grid
        if self.args.crop_feats:
            x = self.crop_feats(x)
        if self.sequence_length == 7:
            train_mask_frames = torch.randint(1, 4, (1,)) if self.training else 3
        else:
            train_mask_frames = self.train_mask_frames if self.training else 1
        masked_x, mask = self.get_mask_tokens(x, mode=self.train_mask_mode, mask_frames=train_mask_frames)
        mask = mask.to(x.device)  # [B,T,H,W] True where masked

        # Forward (pose participates in model + loss)
        if self.args.vis_attn:
            loss, _, _ = self.forward(x, masked_x, mask)
        else:
            loss, _ = self.forward(x, masked_x, mask)

        # Logs
        self.log("Train/loss", loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        # epoch-aggregated train loss (nice and smooth curve)
        self.log("Train/epoch_loss", loss, batch_size=B, logger=True, on_step=False, on_epoch=True, rank_zero_only=True, sync_dist=True)
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
    
    def validation_step(self, batch, batch_idx):
        if self.args.eval_mode or self._do_eval3d_this_epoch:
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
        x = self.preprocess(feats)
        if self.args.crop_feats:
            x = self.crop_feats(x)
        mask_frames = self.train_mask_frames if self.sequence_length != 7 else 1
        masked_x, mask = self.get_mask_tokens(x, mode=self.train_mask_mode, mask_frames=mask_frames)
        
        total_loss, x_pred = self.forward(x, masked_x, mask)
        self._feature_fit_metrics(x_pred, x, mask)

        # per-batch loss for epoch aggregation
        self.mean_metric.update(total_loss)
        self.log('val/loss', total_loss, prog_bar=True, batch_size=B, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def sample_baseline_copy_last(self, feats, last_context_idx=None):
        """
        feats: [B, T, H*W+1, C] with T >= sequence_length (e.g. 7)
        Returns:
            samples: [B, sequence_length, H*W+1, C] with frame 4 predicted as copy-last
            loss   : baseline loss at target_t
        """
        feats = self.preprocess(feats)
        B, T, H, W, C = feats.shape
        target_t = self.target_t               # 4
        if last_context_idx is None:
            last_context_idx = self.context_len - 1  # 3

        # copy last context frame
        x_pred = feats[:, last_context_idx]          # [B, H, W, C]
        x_gt   = feats[:, target_t]                  # [B, H, W, C]

        loss = self.calculate_loss(x_pred, x_gt)

        # construct a sequence [0..4] with predicted frame at target_t
        samples = feats[:, :self.sequence_length].clone()
        samples[:, target_t] = x_pred

        return samples, loss

    def evaluation_step(self, batch, batch_idx):
        feats, depth, pose = batch  # feats: [B, 7, H*W+1, C] (for external), depth: [B, 7, H, W]

        if self.args.evaluate_baseline:
            samples, loss = self.sample_baseline_copy_last(feats)
        else:
            if self.args.eval_midterm:
                unroll_steps = getattr(self.args, "unroll_steps", self.unroll_steps)

                # This is your midterm horizon index in the original 7-frame sequence
                horizon_idx = self.horizon_idx_unroll  # 6 for context_len=4, unroll_steps=3

                # You may need to reshape feats[:, horizon_idx] to [B, H, W, C] if it is [B, HW+1, C]
                # For external features, that might look like:
                B, T, HW_plus1, C = feats.shape
                Hf, Wf = self.args.feat_hw
                gt_feats_horizon = feats[:, horizon_idx, 1:]           # drop pose token
                gt_feats_horizon = gt_feats_horizon.view(B, Hf, Wf, C) # [B,Hf,Wf,C]

                samples, loss = self.sample_unroll(
                    feats,                   # raw features sequence
                    gt_feats=gt_feats_horizon,
                    sched_mode=self.train_mask_mode,
                    step=self.args.step,
                    mask_frames=1,           # only last frame masked
                    unroll_steps=unroll_steps,
                )
            else:
                samples, loss = self.sample(
                    feats, sched_mode=self.train_mask_mode, step=self.args.step
                )

        B = feats.shape[0]
        loss_val = loss if torch.is_tensor(loss) else torch.tensor(loss, device=self.device, dtype=torch.float32)
        self.log('val/loss', loss_val, prog_bar=True, batch_size=B, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.mean_metric.update(loss_val.detach())

        # -------- Depth eval alignment --------
        if self.args.eval_midterm:
            horizon_idx = self.horizon_idx_unroll  # 6
        else:
            horizon_idx = self.target_t            # 4

        pred_last = samples[:, -1]  # [B,Hf,Wf,C]
        per_layer_dims = (1024, 768, 768, 768)
        head_in_tgt = self._create_head_input(pred_last, per_layer_dims=per_layer_dims, add_dummy_pose_token_on_last=True)
        with torch.no_grad():
            head_out_tgt = self.head(head_in_tgt, img_info=self.head_img_hw)

        gt_depth_t = depth[:, horizon_idx].to(self.device)
        self._update_depth_metrics_from_head(head_out_tgt, gt_depth_t)

    def on_validation_epoch_start(self):
        n = getattr(self.args, "eval3d_every_n_epochs", 0) or 0
        self._do_eval3d_this_epoch = (n > 0 and (self.current_epoch % n == 0))
    
    def on_validation_epoch_end(self):
        mean_loss = self.mean_metric.compute()

        # Always log loss
        logs = {
            'val/loss': mean_loss,
        }

        # 3D-eval epoch (either eval-only or periodic 3D eval)
        if self.args.eval_mode or getattr(self, "_do_eval3d_this_epoch", False):
            # log ONLY 3D depth metrics (if they were updated)
            logs['val/depth_absrel'] = self.depth_absrel_m.compute()
            logs['val/depth_delta1'] = self.depth_delta1_m.compute()

            # reset 3D metrics
            self.depth_absrel_m.reset()
            self.depth_delta1_m.reset()

        else:
            # normal feature-only validation epoch:
            # log ONLY feature-fit metrics
            logs['val/ff_mse'] = self.ff_mse.compute()
            logs['val/ff_mae'] = self.ff_mae.compute()
            logs['val/ff_cos'] = self.ff_cos.compute()

            # reset feature metrics
            self.ff_mse.reset()
            self.ff_mae.reset()
            self.ff_cos.reset()

        # common reset
        self.mean_metric.reset()

        self.log_dict(logs, prog_bar=True, logger=True)
            
    """def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]"""

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999)
        )

        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, \
            "Must set max_steps argument via scale_and_set_lr_args"

        warmup_steps = getattr(self.args, "warmup_steps", 0)
        max_steps   = self.args.max_steps

        if warmup_steps > 0:
            def lr_lambda(current_step: int):
                # 1) warmup phase: linear from 0 -> 1
                if current_step < warmup_steps:
                    return float(current_step + 1) / float(max(1, warmup_steps))

                # 2) cosine decay phase: 1 -> 0
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                # standard cosine from 1 to 0
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            # pure cosine over all steps
            def lr_lambda(current_step: int):
                progress = float(current_step) / float(max(1, max_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return [optimizer], [dict(scheduler=scheduler,
                                interval='step',
                                frequency=1)]

    def _create_head_input(self, x_last: torch.Tensor, per_layer_dims=None, add_dummy_pose_token_on_last=False):
        """
        x_last: [B, Hf, Wf, C] or [B, S, C]
        Returns: [l0, l1, l2, l3], each [B, S, d_i]
        Optionally prepends a dummy pose token on the last layer.
        """
        if x_last.dim() == 4:  # [B,Hf,Wf,C]
            B, Hf, Wf, C = x_last.shape
            S = Hf * Wf
            x_flat = x_last.view(B, S, C)
        elif x_last.dim() == 3:  # [B,S,C]
            x_flat = x_last
            B, S, C = x_flat.shape
        else:
            raise ValueError(f"Unexpected feat shape {tuple(x_last.shape)}")

        if per_layer_dims is None:
            q = C // 4
            per_layer_dims = (q, q, q, C - 3*q)

        d0, d1, d2, d3 = per_layer_dims
        assert d0 + d1 + d2 + d3 == C

        l0 = x_flat[:, :, 0:d0]
        l1 = x_flat[:, :, d0:d0+d1]
        l2 = x_flat[:, :, d0+d1:d0+d1+d2]
        l3 = x_flat[:, :, d0+d1+d2:d0+d1+d2+d3]

        if add_dummy_pose_token_on_last:
            dummy_pose_token = torch.zeros(B, 1, d3, device=l3.device, dtype=l3.dtype)
            l3 = torch.cat([dummy_pose_token, l3], dim=1)

        return [l0, l1, l2, l3]

    def _update_depth_metrics_from_head(self, head_out, gt_depth_t):
        """
        Per-sequence depth eval using the original depth_evaluation(), with:
        - per-sequence alignment (scale&shift, scale, or metric)
        - pixel-weighted aggregation over sequences
        - local grad enable for LAD2 optimization (s, t) only
        """
        #torch.set_grad_enable(True)

        if head_out is None or gt_depth_t is None:
            return

        pts3d = head_out.get("pts3d_in_self_view", None)  # [B,H_pred,W_pred,3]
        if pts3d is None:
            return

        # 1) Pred depth from Z
        pred_depth = pts3d[..., 2]                        # [B,H_pred,W_pred]
        gt_depth   = gt_depth_t.to(pred_depth.device)     # [B,H_gt,W_gt]

        # 2) Resize prediction to GT resolution (match original eval scripts)
        if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
            pred_depth = F.interpolate(
                pred_depth.unsqueeze(1),
                size=gt_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)                                   # [B,H_gt,W_gt]

        B = pred_depth.shape[0]

        # 3) Config
        align_mode    = getattr(self.args, "depth_eval_align_mode", "scale&shift")
        max_depth     = getattr(self.args, "depth_eval_max_depth",  None)
        post_clip_max = getattr(self.args, "depth_post_clip_max",   None)

        align_with_lad2  = (align_mode == "scale&shift")  # robust scale+shift
        align_with_scale = (align_mode == "scale")        # robust scale-only
        metric_scale     = (align_mode == "metric")       # no alignment

        # 4) Loop over batch elements, treat each as one "sequence"
        for b in range(B):
            pr_b = pred_depth[b:b+1].detach()   # [1,H,W], detached from net
            gt_b = gt_depth[b:b+1].detach()     # [1,H,W]

            if align_with_lad2:
                # We NEED grads for (s,t) inside absolute_value_scaling2,
                # so temporarily re-enable grad here.
                with torch.enable_grad():
                    depth_results, _, _, _ = depth_evaluation(
                        predicted_depth_original    = pr_b,
                        ground_truth_depth_original = gt_b,
                        max_depth       = max_depth,
                        post_clip_max   = post_clip_max,
                        align_with_lad2 = True,
                        align_with_scale= False,
                        metric_scale    = False,
                        use_gpu         = (self.device.type == "cuda"),
                    )
            else:
                # Other modes don't use autograd-based optimization;
                # they are safe under no_grad but detach anyway for safety.
                depth_results, _, _, _ = depth_evaluation(
                    predicted_depth_original    = pr_b,
                    ground_truth_depth_original = gt_b,
                    max_depth       = max_depth,
                    post_clip_max   = post_clip_max,
                    align_with_lad2 = False,
                    align_with_scale= align_with_scale,
                    metric_scale    = metric_scale,
                    use_gpu         = (self.device.type == "cuda"),
                )

            valid_pixels = depth_results["valid_pixels"]
            if valid_pixels == 0:
                continue

            absrel = depth_results["Abs Rel"]      # float
            delta1 = depth_results["δ < 1.25"]     # float

            absrel_t = torch.tensor(absrel, device=self.device, dtype=torch.float32)
            delta1_t = torch.tensor(delta1, device=self.device, dtype=torch.float32)
            weight_t = torch.tensor(valid_pixels, device=self.device, dtype=torch.float32)

            # Pixel-weighted aggregation (matches original np.average(..., weights=valid_pixels))
            self.depth_absrel_m.update(absrel_t, weight=weight_t)
            self.depth_delta1_m.update(delta1_t, weight=weight_t)
        
    @torch.no_grad()
    def _update_pose_metrics_from_head(
        self
    ):
        pass
        # TODO

        