import json, torch, torch.nn as nn
from pathlib import Path

# from your ported file:
# from your_heads_file import DPTPts3dPose, PixelwiseTaskWithDPT
import sys
sys.path.append("/workspace/CUT3R/src")
from dust3r.heads.dpt_head import DPTPts3dPose, PixelwiseTaskWithDPT

class _NetLike:
    """Tiny object that exposes the attributes the head constructor reads."""
    def __init__(self, enc_embed_dim, dec_embed_dim, dec_num_heads, depth_mode, conf_mode, pose_mode):
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.dec_num_heads = dec_num_heads
        self.depth_mode = tuple(depth_mode) if depth_mode else None
        self.conf_mode  = tuple(conf_mode)  if conf_mode  else None
        self.pose_mode  = tuple(pose_mode)  if pose_mode  else None
        self.rope = None  # not used by the DPT adapters in your pasted code

def load_exported_head(export_dir):
    export_dir = Path(export_dir)
    meta = json.loads((export_dir / "cut3r_head_meta.json").read_text())

    net_like = _NetLike(
        meta["enc_embed_dim"], meta["dec_embed_dim"], meta["dec_num_heads"],
        meta["depth_mode"], meta["conf_mode"], meta["pose_mode"],
    )

    # Pick the correct head class
    if meta["has_pose"] or meta["has_rgb"]:
        head = DPTPts3dPose(
            net_like,
            has_conf=bool(meta["has_conf"]),
            has_rgb=bool(meta["has_rgb"]),
            has_pose=bool(meta["has_pose"]),
        )
    else:
        # plain pixelwise pts3d head
        head = PixelwiseTaskWithDPT(
            num_channels=3 + int(bool(meta["has_conf"])),
            feature_dim=256, last_dim=128,
            hooks_idx=[0, meta["dec_depth"]*2//4, meta["dec_depth"]*3//4, meta["dec_depth"]],
            dim_tokens=[meta["enc_embed_dim"], meta["dec_embed_dim"], meta["dec_embed_dim"], meta["dec_embed_dim"]],
            postprocess=postprocess,  # import from your utils
            depth_mode=meta["depth_mode"],
            conf_mode=meta["conf_mode"],
            head_type="regression",
        )

    # Load weights (strict=True to catch mismatches)
    sd = torch.load(export_dir / "cut3r_head_only.pth", map_location="cpu")
    head.load_state_dict(sd, strict=True)
    head.eval()
    return head, meta


if __name__ == "__main__":
    head, meta = load_exported_head("/workspace/CUT3R/exported_cut3r_head")

    # Test if forward pass runs through
    l0 = torch.randn(1, 196, 1024)
    l1 = torch.randn(1, 196, 768)
    l2 = torch.randn(1, 196, 768)
    l3 = torch.randn(1, 197, 768)
    head_input = [l0, l1, l2, l3]
    with torch.no_grad():
        out = head(head_input, img_info=(224,224))
    for k,v in out.items():
        print(f"{k}: {v.shape}")