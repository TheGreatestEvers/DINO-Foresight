# BERT architecture for the Masked Bidirectional Encoder Transformer
import torch
import einops
from torch import nn
import torch.nn.functional as F


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        """ PreNorm module to apply layer normalization before a given function
            :param:
                dim  -> int: Dimension of the input
                fn   -> nn.Module: The function to apply after layer normalization
            """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after layer normalization
        """
        return self.fn(self.norm(x), **kwargs)
        

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """ Initialize the Multi-Layer Perceptron (MLP).
            :param:
                dim        -> int : Dimension of the input
                dim        -> int : Dimension of the hidden layer
                dim        -> float : Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        """ Initialize the Attention module.
            :param:
                embed_dim     -> int : Dimension of the embedding
                num_heads     -> int : Number of heads
                dropout       -> float : Dropout rate
        """
        super(Attention, self).__init__()
        self.dim = embed_dim
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True, bias=True)

    def forward(self, x, mask=None):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                attention_value  -> torch.Tensor: Output the value of the attention
                attention_weight -> torch.Tensor: Output the weight of the attention
        """
        if mask is not None:
            attention_value, attention_weight = self.mha(x, x, x, attn_mask=mask)
        else:
            attention_value, attention_weight = self.mha(x, x, x)
        return attention_value, attention_weight


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, full_shape=None):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        l_attn = []
        for attn, ff in self.layers:
            attention_value, attention_weight = attn(x)
            x = attention_value + x
            x = ff(x) + x
            l_attn.append(attention_weight)
        return x, l_attn


class TransformerEncoderSeperableAttention(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., window_size=1):
        """ Initialize the Attention module.
            :param:
                dim       -> int : number of hidden dimension of attention
                depth     -> int : number of layer for the transformer
                heads     -> int : Number of heads
                mlp_dim   -> int : number of hidden dimension for mlp
                dropout   -> float : Dropout rate
        """
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, full_shape):
        """ Forward pass through the Attention module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                x -> torch.Tensor: Output of the Transformer
                l_attn -> list(torch.Tensor): list of the attention
        """
        b, t, h, w, d = full_shape
        l_attn = []
        for attn_temporal, attn_spatial, ff in self.layers:
            if self.window_size == 1:
                x = einops.rearrange(x, 'b (t h w) c -> (b h w) t c', b=b, t=t, h=h, w=w)
            else:
                x = einops.rearrange(x, 'b (t h w) c -> b t h w c', b=b, t=t, h=h, w=w)
                x = einops.rearrange(
                    x, 'b t (h k1) (w k2) c -> (b h w) (t k1 k2) c',
                    k1=self.window_size, k2=self.window_size)
            attention_value, attention_weight = attn_temporal(x)
            x = x + attention_value
            l_attn.append(attention_weight)

            if self.window_size == 1:
                x = einops.rearrange(x, '(b h w) t c -> (b t) (h w) c', b=b, t=t, h=h, w=w)
            else:
                x = einops.rearrange(
                    x, '(b h w) (t k1 k2) c -> (b t) (h k1) (w k2) c',
                    b=b, t=t, h=h//self.window_size, w=w//self.window_size, k1=self.window_size, k2=self.window_size)
                x = x.flatten(1,2)
            attention_value, attention_weight = attn_spatial(x)
            x = x + attention_value
            l_attn.append(attention_weight)

            x = einops.rearrange(x, '(b t) (h w) c -> b (t h w) c', b=b, t=t, h=h, w=w)
            x = x + ff(x)

        return x, l_attn


class MaskTransformer(nn.Module):
    def __init__(self, shape, img_size=256, embedding_dim = 768, hidden_dim=768, codebook_size=1024, depth=24, heads=8, mlp_dim=3072, 
                dropout=0.1, one_var=False, use_fc_bias=False, use_first_last=False, 
                seperable_attention=False, seperable_window_size=1, train_aux=False):
        """ Initialize the Transformer model.
            :param:
                #img_size       -> int:     Input image size (default: 256)
                hidden_dim     -> int:     Hidden dimension for the transformer (default: 768)
                codebook_size  -> int:     Size of the codebook (default: 1024)
                depth          -> int:     Depth of the transformer (default: 24)
                heads          -> int:     Number of attention heads (default: 8)
                mlp_dim        -> int:     MLP dimension (default: 3072)
                dropout        -> float:   Dropout rate (default: 0.1)
                # nclass         -> int:     Number of classes (default: 1000)
        """
        super().__init__()
        self.pos_embd = AddBroadcastPosEmbed(shape=shape, embd_dim=hidden_dim)
        
        # First layer before the Transformer block
        self.first_layer = nn.Identity() 
        if use_first_last:
            self.first_layer = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim), # hidden_dim = 768
            )

        self.seperable_attention = seperable_attention
        Transformer = TransformerEncoderSeperableAttention if seperable_attention else TransformerEncoder
        self.transformer = Transformer(dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        if seperable_attention:
            self.transformer = TransformerEncoderSeperableAttention(
                dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout,
                window_size=seperable_window_size)
        else:
            self.transformer = TransformerEncoder(
                dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # Last layer after the Transformer block
        self.last_layer = nn.Identity()
        if use_first_last:
            self.last_layer = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim, eps=1e-12),
            )
        self.fc_in = nn.Linear(hidden_dim, hidden_dim, bias=use_fc_bias)
        self.fc_in.weight.data.normal_(std=0.02) 

        self.fc_out = nn.Linear(hidden_dim, embedding_dim, bias=use_fc_bias)
        self.fc_out.weight.data.copy_(torch.zeros(embedding_dim, hidden_dim))


    def forward(self, vid_token, y=None, drop_label=None, return_attn=False):
        """ Forward.
            :param:
                img_token      -> torch.LongTensor: bsize x 16 x 16, the encoded image tokens
                y              -> torch.LongTensor: condition class to generate
                drop_label     -> torch.BoolTensor: either or not to drop the condition
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: bsize x path_size*path_size * 1024, the predicted logit
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, t, h, w, c = vid_token.size()

        # Position embedding
        vid_token = self.fc_in(vid_token)
        vid_token_embeddings = self.pos_embd(vid_token) 
        x = einops.rearrange(vid_token_embeddings, 'b t h w c -> b (t h w) c')

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x, full_shape=[b, t, h, w, c])
        x = self.last_layer(x)
        x_out = self.fc_out(x)
        if return_attn:
            return x_out,  attn
        else:
            return x_out,

################ Spatiotemporal broadcasted positional embeddings ###############
class AddBroadcastPosEmbed(nn.Module):
    def __init__(self, shape, embd_dim, dim=-1):
        super().__init__()
        assert dim in [-1, 1] # only first or last dim supported
        self.shape = shape 
        self.n_dim = n_dim = len(shape) 
        self.embd_dim = embd_dim
        self.dim = dim

        assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
        self.emb = nn.ParameterDict({
            f'd_{i}': nn.init.trunc_normal_(nn.Parameter(torch.randn(self.shape[i], embd_dim // n_dim)),0.,0.02)
                                    if dim == -1 else
                                    nn.init.trunc_normal_(torch.randn(embd_dim // n_dim, self.shape[i]),0.,0.02)
            for i in range(n_dim)
        })
        # print("self.emb",self.emb)

    def forward(self, x, decode_step=None, decode_idx=None):
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f'd_{i}']
            if self.dim == -1:
                # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
                e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
                # print("e.shape",e.shape) # [1,5, 1, 1, 384] , [1,1,16,1,384], [1,1,1,32,384]
                e = e.expand(1, *self.shape, -1)
                # print("e.expand.shape",e.shape)
            else:
                e = e.view(1, -1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)))
                e = e.expand(1, -1, *self.shape)
            
            embs.append(e)

        embs = torch.cat(embs, dim=self.dim)
        return x + embs
    


# === New, additive classes (do not override your originals) ===================
import torch
import einops
from torch import nn

# --- simple temporal PE for the pose token (time-aware + token-type bias) ---
class PoseTemporalPE(nn.Module):
    def __init__(self, T, dim, init_std=0.02):
        super().__init__()
        self.temb = nn.Embedding(T, dim)
        nn.init.trunc_normal_(self.temb.weight, std=init_std)
        self.type_bias = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.type_bias, std=init_std)

    def forward(self, B, T, device):
        t_ids = torch.arange(T, device=device)                   # (T,)
        pe = self.temb(t_ids)[None].expand(B, T, -1)             # (B,T,C)
        pe = pe + self.type_bias                                 # (B,T,C)
        return pe.unsqueeze(2)                                   # (B,T,1,C)


# --- separable encoder that knows about extra per-frame tokens (pose slots) ---
class TransformerEncoderSeparableWithPose(nn.Module):
    """
    Drop-in variant of your separable encoder that also takes pose_n per frame.
    window_size=1 path is implemented for simplicity (as requested).
    """
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., window_size=1):
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),  # temporal
                PreNorm(dim, Attention(dim, heads, dropout=dropout)),  # spatial
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, full_shape, pose_n=0):
        """
        x:          (B, T*(H*W + pose_n), C) when window_size == 1
        full_shape: [B, T, H, W, C_in]
        pose_n:     int, number of per-frame extra tokens (0 or 1)
        """
        b, t, h, w, _ = full_shape
        assert self.window_size == 1, "Pose path currently supports window_size=1."
        p = h * w + pose_n  # spatial length per frame

        l_attn = []
        for attn_temporal, attn_spatial, ff in self.layers:
            # Temporal attention: for each spatial slot, attend over time
            xt = einops.rearrange(x, 'b (t p) c -> (b p) t c', b=b, t=t, p=p)
            a_val_t, a_w_t = attn_temporal(xt)
            xt = xt + a_val_t
            l_attn.append(a_w_t)

            # Spatial attention: for each time step, attend over spatial positions (incl. pose slot)
            xs = einops.rearrange(xt, '(b p) t c -> (b t) p c', b=b, t=t, p=p)
            a_val_s, a_w_s = attn_spatial(xs)
            xs = xs + a_val_s
            l_attn.append(a_w_s)

            # MLP + residual, back to (B, T*P, C)
            x = einops.rearrange(xs, '(b t) p c -> b (t p) c', b=b, t=t, p=p)
            x = x + ff(x)

        return x, l_attn


# --- main model that accepts an optional pose token and returns both outputs ---
class MaskTransformerWithPose(nn.Module):
    """
    New class that mirrors your MaskTransformer but can also take a per-frame pose token
    (B x T x 1 x C_in). It returns (grid_pred, pose_pred) when pose is provided,
    otherwise it behaves like your original (returns only grid_pred).
    """
    def __init__(self, shape, img_size=256, embedding_dim=768, hidden_dim=768, codebook_size=1024,
                 depth=24, heads=8, mlp_dim=3072, dropout=0.1, one_var=False, use_fc_bias=False,
                 use_first_last=False, seperable_attention=True, seperable_window_size=1, train_aux=False):
        super().__init__()
        self.shape = shape  # [T, H, W]
        T, H, W = shape
        self.seperable_attention = seperable_attention

        # Positional embeddings for the geometry grid (reuse your module)
        self.pos_embd = AddBroadcastPosEmbed(shape=shape, embd_dim=hidden_dim)

        # Simple temporal PE for the pose token
        self.pose_pe = PoseTemporalPE(T=T, dim=hidden_dim)

        # Optional first/last layers (mirror your setup)
        self.first_layer = nn.Identity()
        if use_first_last:
            self.first_layer = nn.Sequential(
                nn.LayerNorm(hidden_dim, eps=1e-12),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim, eps=1e-12),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            )

        self.last_layer = nn.Identity()
        if use_first_last:
            self.last_layer = nn.Sequential(
                nn.LayerNorm(hidden_dim, eps=1e-12),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim, eps=1e-12),
            )

        # In/out projections
        self.fc_in  = nn.Linear(hidden_dim, hidden_dim, bias=use_fc_bias)
        self.fc_out = nn.Linear(hidden_dim, embedding_dim, bias=use_fc_bias)
        self.fc_in.weight.data.normal_(std=0.02)
        self.fc_out.weight.data.copy_(torch.zeros(embedding_dim, hidden_dim))

        # Transformer backbone
        if seperable_attention:
            self.transformer = TransformerEncoderSeparableWithPose(
                dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout,
                window_size=seperable_window_size
            )
        else:
            # falls back to your vanilla encoder (no pose specialization needed)
            self.transformer = TransformerEncoder(
                dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout
            )

    @torch.no_grad()
    def _check_shapes(self, vid_token, pose_token):
        b, t, h, w, c_in = vid_token.size()
        if pose_token is not None:
            assert pose_token.shape[:3] == (b, t, 1), \
                f"pose_token must be (B,T,1,C), got {pose_token.shape}"
            assert pose_token.shape[-1] == c_in, \
                f"pose_token C must match vid_token C ({c_in}), got {pose_token.shape[-1]}"
        return b, t, h, w, c_in

    def forward(self, vid_token, pose_token=None, y=None, drop_label=None, return_attn=False):
        """
        vid_token:  B x T x H x W x C_in (geometry features)
        pose_token: (optional) B x T x 1 x C_in (per-frame pose features)
        Returns:
            - If pose provided:  grid_pred (B,T,H,W,C_out), pose_pred (B,T,1,C_out)
            - Else:              grid_pred (B,T,H,W,C_out)
        """
        b, t, h, w, c_in = self._check_shapes(vid_token, pose_token)

        # Project + add broadcasted (T,H,W) PE to the grid
        grid = self.fc_in(vid_token)               # (B,T,H,W,C)
        grid = self.pos_embd(grid)                 # (B,T,H,W,C)
        grid = einops.rearrange(grid, 'b t h w c -> b t (h w) c')  # (B,T,Pg,C), Pg=H*W

        pose_n = 0
        if pose_token is not None:
            pose_n = 1
            pose = self.fc_in(pose_token)                  # (B,T,1,C)
            pose = pose + self.pose_pe(B=b, T=t, device=pose.device)  # (B,T,1,C)
            x = torch.cat([grid, pose], dim=2)             # (B,T,Pg+1,C)
        else:
            x = grid                                       # (B,T,Pg,C)

        # pack to sequence (B, T*P, C)
        x = einops.rearrange(x, 'b t p c -> b (t p) c')

        # transformer
        x = self.first_layer(x)
        if self.seperable_attention:
            x, attn = self.transformer(x, full_shape=[b, t, h, w, c_in], pose_n=pose_n)
        else:
            x, attn = self.transformer(x, full_shape=[b, t, h, w, c_in])

        x = self.last_layer(x)
        x = self.fc_out(x)  # (B, T*P, C_out)

        # unpack + split
        p = h * w + pose_n
        x = einops.rearrange(x, 'b (t p) c -> b t p c', t=t, p=p)
        if pose_n == 1:
            grid_pred = x[:, :, :h*w, :]
            pose_pred = x[:, :, h*w:, :]    # (B,T,1,C_out)
        else:
            grid_pred = x
            pose_pred = None

        grid_pred = einops.rearrange(grid_pred, 'b t (h w) c -> b t h w c', h=h, w=w)

        if return_attn:
            return (grid_pred, pose_pred), attn
        else:
            return grid_pred, pose_pred