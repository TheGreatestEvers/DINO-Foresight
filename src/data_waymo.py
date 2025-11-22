from pathlib import Path
from tqdm import tqdm
import torch
import os
import re
from torch.utils.data import Dataset

class SelectedFusedImgFeatConcatDataset(Dataset):
    """
    Loads tXX.pt files and returns:
        feats_with_pose: [T, S+1, 4*D]

    - the first token (index 0) is the concatenated pose token
      from the four layers [l0, l2q, l3q, lfin]
    - the remaining S tokens are the spatial features
    - if return_view=True, also returns (depth_T, pose_view_T)

    Notes:
    - The pose tokens are no longer discarded or separated;
      they are prepended to the spatial features for each timestep.
    """
    _t_pat = re.compile(r"[tT](\d+)\.pt$")

    def __init__(self, root_dir: str, return_view: bool = False):
        super().__init__()
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(root_dir)
        self.root_dir = root_dir
        self.return_view = return_view

        items = []
        for seq in sorted(os.listdir(root_dir)):
            seq_path = os.path.join(root_dir, seq)
            if not os.path.isdir(seq_path):
                continue
            for sample in sorted(os.listdir(seq_path)):
                sample_path = os.path.join(seq_path, sample)
                if not os.path.isdir(sample_path):
                    continue
                pt_files = sorted(
                    (f for f in os.listdir(sample_path) if f.endswith(".pt")),
                    key=lambda n: int(self._t_pat.search(n).group(1)) if self._t_pat.search(n) else n,
                )
                if pt_files:
                    items.append({
                        "sequence": seq,
                        "sample": sample,
                        "paths": [os.path.join(sample_path, f) for f in pt_files],
                    })
        if not items:
            raise FileNotFoundError(f"No .pt files found under {root_dir}")
        self._items = items

    def __len__(self):
        return len(self._items)

    @staticmethod
    def _split_pose_and_spatial(x: torch.Tensor, target_S: int):
        """
        Split a [S or S+1, D] tensor into (spatial, pose).
        If x has one extra token, x[0] is pose, x[1:] spatial.
        If x has no extra token, pose=None.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected [S,D], got {tuple(x.shape)}")
        Sx, D = x.shape
        if Sx == target_S + 1:
            return x[1:], x[0]           # spatial, pose
        if Sx == target_S:
            return x, None                # spatial, no pose
        raise ValueError(f"Incompatible S: got {Sx}, expected {target_S} or {target_S+1}")

    def __getitem__(self, idx):
        it = self._items[idx]
        feats_T = []
        depth_list = []
        pose_view_list = []

        # Determine target_S (spatial length) from first payload
        first_payload = torch.load(it["paths"][0], map_location="cpu")
        L0 = first_payload["layers"]
        target_S = min([L0[k].shape[0] for k in L0])  # smallest spatial len (ignoring pose token)

        layer_order = ["l0", "l2q", "l3q", "lfin"]

        for p in it["paths"]:
            payload = torch.load(p, map_location="cpu")
            L = payload["layers"]

            spatial_parts = []
            pose_parts = []

            for k in layer_order:
                x = L[k]
                spatial_k, pose_k = self._split_pose_and_spatial(x, target_S)
                spatial_parts.append(spatial_k)
                # if missing pose, fill zeros to preserve dim
                if pose_k is None:
                    pose_k = torch.zeros(spatial_k.shape[-1], dtype=spatial_k.dtype)
                pose_parts.append(pose_k)

            # concat spatial along feature dim -> [S, 4*D]
            spatial_concat = torch.cat(spatial_parts, dim=-1)
            # concat pose tokens along feature dim -> [4*D]
            pose_concat = torch.cat(pose_parts, dim=-1).unsqueeze(0)  # [1, 4*D]

            # prepend pose token → [S+1, 4*D]
            feats = torch.cat([pose_concat, spatial_concat], dim=0)
            feats_T.append(feats)

            if self.return_view:
                V = payload.get("view", {}) or {}
                depth_list.append(V.get("depthmap", None))
                pose_view_list.append(V.get("camera_pose", None))

        # stack over time → [T, S+1, 4*D]
        out_feats = torch.stack(feats_T, dim=0).float()

        if not self.return_view:
            return out_feats

        def _try_stack(lst):
            if all((x is not None) for x in lst):
                try:
                    return torch.stack([x.float() for x in lst], dim=0)
                except Exception:
                    return lst
            if all((x is None) for x in lst):
                return None
            return lst

        depth_T = _try_stack(depth_list)
        pose_view_T = _try_stack(pose_view_list)
        return out_feats, depth_T, pose_view_T


if __name__ == "__main__":
    # Features + view data
    ds_view = SelectedFusedImgFeatConcatDataset("/workspace/raid/jevers/cut3r_features/waymo/fused_img_tokens_224/train", return_view=True)
    print("Length feature set: ,", len(ds_view))
    sample = ds_view[0]
    if isinstance(sample, tuple):
        feats, depth, pose = sample
        print("Feats:", None if feats is None else feats.shape)
        if isinstance(depth, torch.Tensor):
            print("Depth:", depth.shape)
        else:
            print("Depth:", type(depth))
        if isinstance(pose, torch.Tensor):
            print("Pose:", pose.shape)
        else:
            print("Pose:", type(pose))