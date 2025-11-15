# src/pose_eval_helpers.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# --- evo (optional, still supported) ---
from evo.core.trajectory import PoseTrajectory3D
from evo.core.geometry import GeometryException
from scipy.spatial.transform import Rotation as R

# optional if you sometimes get Dust3r-encoded poses instead of quat7
try:
    import sys
    sys.path.append("/workspace/cut3r-forecasting/cut3r/src")
    from dust3r.utils.camera import pose_encoding_to_camera  # noqa: F401
except Exception:
    pose_encoding_to_camera = None


# ---------------------------------------------------------
# Conversions
# ---------------------------------------------------------
def quat7_to_mat44(q7: torch.Tensor) -> torch.Tensor:
    """
    q7: [B,7] -> [B,4,4] c2w (float32)
      layout auto-detected for quaternion part:
        either [tx,ty,tz, qw,qx,qy,qz]  or  [tx,ty,tz, qx,qy,qz,qw]
    """
    assert q7.ndim == 2 and q7.shape[-1] == 7, f"Expected [B,7], got {tuple(q7.shape)}"
    t = q7[:, :3]
    q = q7[:, 3:]

    # candidate A: [qw,qx,qy,qz]
    qA = torch.stack([q[:, 0], q[:, 1], q[:, 2], q[:, 3]], dim=-1)
    # candidate B: [qx,qy,qz,qw] -> to [qw,qx,qy,qz]
    qB = torch.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], dim=-1)

    # pick by unit-norm proximity
    errA = (qA.norm(dim=-1) - 1.0).abs()
    errB = (qB.norm(dim=-1) - 1.0).abs()
    useA = (errA <= errB).view(-1, 1).float()
    qWXYZ = F.normalize(useA * qA + (1.0 - useA) * qB, dim=-1)

    # enforce shortest arc (optional but helps consistency)
    neg = qWXYZ[:, 0] < 0
    qWXYZ[neg] = -qWXYZ[neg]

    # scipy expects [x,y,z,w]
    q_xyzw = torch.stack([qWXYZ[:, 1], qWXYZ[:, 2], qWXYZ[:, 3], qWXYZ[:, 0]], dim=-1).cpu().numpy()
    Rmats = torch.from_numpy(R.from_quat(q_xyzw).as_matrix()).to(q7.device, q7.dtype)

    B = q7.shape[0]
    T = torch.eye(4, dtype=q7.dtype, device=q7.device).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = Rmats
    T[:, :3,  3] = t
    return T

def ensure_4x4(x) -> np.ndarray:
    """Accepts (4,4), (3,4), (1,4,4), (1,3,4) or torch tensors. Returns (4,4) float64."""
    A = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    A = A.astype(np.float64)
    if A.ndim == 3 and A.shape[0] == 1:
        A = A[0]
    if A.shape == (4, 4):
        return A
    if A.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = A[:, :3]
        T[:3,  3] = A[:,  3]
        return T
    raise ValueError(f"Bad pose shape {A.shape}")


# ---------------------------------------------------------
# evo helpers  (kept for multi-context alignment use-cases)
# ---------------------------------------------------------
def traj_from_c2w_list(c2w_list: List[np.ndarray]) -> PoseTrajectory3D:
    """GT c2w list -> evo PoseTrajectory3D (wxyz convention)."""
    mats = [ensure_4x4(T) for T in c2w_list]
    pos = np.stack([T[:3, 3] for T in mats], 0)          # (N,3)
    Rm  = np.stack([T[:3, :3] for T in mats], 0)         # (N,3,3)
    q_xyzw = R.from_matrix(Rm).as_quat()                 # (N,4) [x,y,z,w]
    q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], 1)  # evo wants [w,x,y,z]
    ts = np.arange(len(mats), dtype=float)
    return PoseTrajectory3D(positions_xyz=pos, orientations_quat_wxyz=q_wxyz, timestamps=ts)

def traj_positions_quats(traj: PoseTrajectory3D) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions(N,3), quats_wxyz(N,4)) as np.float64."""
    P = np.asarray(traj.positions_xyz, dtype=np.float64)
    Q = np.asarray(traj.orientations_quat_wxyz, dtype=np.float64)
    return P, Q


# ---------------------------------------------------------
# Small Umeyama (Sim(3) on positions only)
# ---------------------------------------------------------
def umeyama_sim3(P: np.ndarray, Q: np.ndarray, with_scale=True) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Solve s,R,t s.t. s*R*P + t ≈ Q  (P,Q: Nx3). Returns (R(3,3), s, t(3,))
    """
    assert P.shape == Q.shape and P.ndim == 2 and P.shape[1] == 3 and P.shape[0] >= 2
    muP = P.mean(0); muQ = Q.mean(0)
    X = P - muP; Y = Q - muQ
    Sigma = (Y.T @ X) / P.shape[0]
    U, D, Vt = np.linalg.svd(Sigma)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = U @ Vt
    varP = (X * X).sum() / P.shape[0]
    s = (D.sum() / (varP + 1e-12)) if with_scale else 1.0
    t = muQ - s * (Rm @ muP)
    return Rm, float(s), t

def apply_sim3(T44: np.ndarray, Rm: np.ndarray, s: float, t: np.ndarray) -> np.ndarray:
    """Apply Sim(3) (R,s,t) to a 4×4 c2w (rot left-multiplied, position mapped by s*R*p + t)."""
    T = ensure_4x4(T44).copy()
    T[:3, :3] = Rm @ T[:3, :3]
    T[:3,  3] = s * (Rm @ T[:3, 3]) + t
    return T


# ---------------------------------------------------------
# Metrics utils
# ---------------------------------------------------------
def rel_T(Ti: np.ndarray, Tj: np.ndarray) -> np.ndarray:
    return np.linalg.inv(ensure_4x4(Ti)) @ ensure_4x4(Tj)

def rot_angle_deg(T_err: np.ndarray) -> float:
    Rm = ensure_4x4(T_err)[:3, :3]
    tr = np.trace(Rm)
    val = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))

def trans_norm(T_err: np.ndarray) -> float:
    return float(np.linalg.norm(ensure_4x4(T_err)[:3, 3]))

def shortest_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Ensure scalar part >= 0 for consistency (optional)."""
    q = q_wxyz.copy()
    flip = q[:, 0] < 0
    q[flip] *= -1.0
    return q


# ---------------------------------------------------------
# Forecast metrics using evo (context-only alignment; kept for reference)
# ---------------------------------------------------------
def _filter_distinct_rows(P: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Keep rows that are distinct up to a tolerance.
    Returns a mask of valid (kept) rows.
    """
    if P.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    key = np.round(P / max(tol, 1e-12), decimals=0)
    key_view = np.ascontiguousarray(key).view(
        np.dtype((np.void, key.dtype.itemsize * key.shape[1]))
    )
    _, idx = np.unique(key_view, return_index=True)
    mask = np.zeros(P.shape[0], dtype=bool)
    mask[idx] = True
    return mask

@torch.no_grad()
def forecast_pose_metrics_evo(
    gt_ctx_c2w: list,                 # list len Tctx of (4,4)
    gt_tgt_c2w: np.ndarray,           # (4,4)
    pred_ctx_q7: torch.Tensor | None, # [Tctx,7] or None
    pred_tgt_q7: torch.Tensor         # [1,7]
):
    """
    Context-only Sim(3) alignment (robust):
      - filter degenerate/duplicate context points
      - try evo.align(correct_scale=True)
      - on failure, fallback to Umeyama(P_pred_ctx -> P_gt_ctx)
      - otherwise identity Sim(3)
    Metrics are computed ONLY on the forecast target frame:
      - ATE_target (m)
      - 1-step RPE: translation (m) and rotation (deg) wrt last GT context
    """
    # ----- GT target & last context -----
    T_gt_last = ensure_4x4(gt_ctx_c2w[-1])
    T_gt_tgt  = ensure_4x4(gt_tgt_c2w)

    # default: identity Sim(3)
    Rsim = np.eye(3); ssim = 1.0; tsim = np.zeros(3)

    # ----- Context alignment path -----
    if pred_ctx_q7 is not None and pred_ctx_q7.shape[0] >= 2:
        # Predicted context -> 4x4 and positions
        T_pred_ctx = quat7_to_mat44(pred_ctx_q7).detach().cpu().numpy()  # (Nc,4,4)
        P_pred_raw = T_pred_ctx[:, :3, 3]

        # GT context positions
        T_gt_ctx = [ensure_4x4(T) for T in gt_ctx_c2w]
        P_gt_raw = np.stack([T[:3, 3] for T in T_gt_ctx], 0)

        # Filter duplicates (independently on pred/gt but keep lengths consistent)
        m_pred = _filter_distinct_rows(P_pred_raw, tol=1e-6)
        m_gt   = _filter_distinct_rows(P_gt_raw,   tol=1e-6)

        # Use indices present in both (conservative)
        keep = np.where(m_pred & m_gt)[0]
        P_pred = P_pred_raw[keep] if keep.size > 0 else P_pred_raw
        P_gt   = P_gt_raw[keep]   if keep.size > 0 else P_gt_raw

        # Need at least 2 distinct points to estimate Sim(3)
        if P_pred.shape[0] >= 2 and P_gt.shape[0] >= 2:
            try:
                traj_gt   = traj_from_c2w_list([T_gt_ctx[i] for i in range(len(T_gt_ctx))])
                traj_pred = traj_from_c2w_list([T_pred_ctx[i] for i in range(T_pred_ctx.shape[0])])
                traj_pred.align(traj_gt, correct_scale=True)   # may raise GeometryException
                P_aligned, _ = traj_positions_quats(traj_pred)
                # derive Sim(3) mapping original P_pred_raw -> P_aligned
                Rsim, ssim, tsim = umeyama_sim3(P_pred_raw, P_aligned, with_scale=True)
            except GeometryException:
                try:
                    Rsim, ssim, tsim = umeyama_sim3(P_pred, P_gt, with_scale=True)
                except Exception:
                    pass

    # ----- Apply Sim(3) to predicted TARGET -----
    T_pred_tgt = quat7_to_mat44(pred_tgt_q7).detach().cpu().numpy()[0]
    T_pred_tgt_aligned = apply_sim3(T_pred_tgt, Rsim, ssim, tsim)

    # ----- Metrics on TARGET ONLY -----
    ate = float(np.linalg.norm(T_pred_tgt_aligned[:3, 3] - T_gt_tgt[:3, 3]))

    dT_gt   = rel_T(T_gt_last, T_gt_tgt)
    dT_pred = rel_T(T_gt_last, T_pred_tgt_aligned)
    dT_err  = np.linalg.inv(dT_gt) @ dT_pred

    rpe_t = trans_norm(dT_err)
    rpe_r = rot_angle_deg(dT_err)
    return ate, rpe_t, rpe_r


# ---------------------------------------------------------
# Two-frame Sim(3) alignment (simple & minimal)
# ---------------------------------------------------------
def sim3_from_two_frames(
    T_gt_last: np.ndarray, T_gt_tgt: np.ndarray,
    T_pr_last: np.ndarray, T_pr_tgt: np.ndarray,
    eps: float = 1e-8, align_rot: bool = True
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Build a Sim(3) using only last-context & target frames.

    - Rotation: align last-context orientations (optional).
    - Scale   : ratio of step lengths ||p_gt_tgt - p_gt_last|| / ||p_pr_tgt - p_pr_last||.
    - Translation: pin last-context positions.

    Returns (Rsim(3x3), ssim(float), tsim(3,))
    """
    TgL = ensure_4x4(T_gt_last);  TgT = ensure_4x4(T_gt_tgt)
    TpL = ensure_4x4(T_pr_last);  TpT = ensure_4x4(T_pr_tgt)

    p_gL, R_gL = TgL[:3, 3], TgL[:3, :3]
    p_gT       = TgT[:3, 3]
    p_pL, R_pL = TpL[:3, 3], TpL[:3, :3]
    p_pT       = TpT[:3, 3]

    Rsim = (R_gL @ R_pL.T) if align_rot else np.eye(3)

    dp_gt   = p_gT - p_gL
    dp_pred = p_pT - p_pL
    n_gt    = np.linalg.norm(dp_gt)
    n_pr    = np.linalg.norm(dp_pred)
    if n_pr < eps:
        ssim = 1.0
    else:
        ssim = float(n_gt / n_pr)

    tsim = p_gL - ssim * (Rsim @ p_pL)
    return Rsim, ssim, tsim

def apply_sim3_pose(T44: np.ndarray, Rm: np.ndarray, s: float, t: np.ndarray, align_rot: bool = True) -> np.ndarray:
    """Apply Sim(3) to a 4x4; optionally do NOT touch the local orientation."""
    T = ensure_4x4(T44).copy()
    if align_rot:
        T[:3, :3] = Rm @ T[:3, :3]
    T[:3,  3] = s * (Rm @ T[:3, 3]) + t
    return T

def one_step_pose_metrics(
    T_gt_last: np.ndarray, T_gt_tgt: np.ndarray,
    T_pred_last: np.ndarray, T_pred_tgt: np.ndarray,
    align_rot: bool = True
) -> tuple[float, float, float]:
    """
    Compute ATE (target only) and 1-step RPE (trans[m], rot[deg]) using a two-frame Sim(3).
    """
    Rsim, ssim, tsim = sim3_from_two_frames(T_gt_last, T_gt_tgt, T_pred_last, T_pred_tgt, align_rot=align_rot)

    T_pl_al = apply_sim3_pose(T_pred_last, Rsim, ssim, tsim, align_rot=align_rot)
    T_pt_al = apply_sim3_pose(T_pred_tgt,  Rsim, ssim, tsim, align_rot=align_rot)

    # ATE on target translation
    ate = float(np.linalg.norm(T_pt_al[:3, 3] - ensure_4x4(T_gt_tgt)[:3, 3]))

    # 1-step RPE (within each trajectory)
    dT_gt   = np.linalg.inv(ensure_4x4(T_gt_last)) @ ensure_4x4(T_gt_tgt)
    dT_pred = np.linalg.inv(T_pl_al) @ T_pt_al
    dT_err  = np.linalg.inv(dT_gt) @ dT_pred
    rpe_t = trans_norm(dT_err)
    rpe_r = rot_angle_deg(dT_err)
    return ate, rpe_t, rpe_r