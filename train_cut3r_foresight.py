import argparse, os, torch, pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger          # <-- ADDED
from pytorch_lightning.callbacks import LearningRateMonitor      # <-- ADDED
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from torch.utils.data import DataLoader, Subset
import torch

from src.dino_f import Dino_f  
from src.data_waymo import SelectedFusedImgFeatConcatDataset

DEBUG = False

def _stack_or_list(xs, dim=0):
    """Try to stack a list of Tensors. If shapes differ or any element is None, return the list unchanged."""
    if len(xs) == 0:
        return xs
    if all((x is None) for x in xs):
        return None
    if any((x is None) for x in xs):
        # mixed availability -> keep per-sample list to preserve alignment
        return xs
    try:
        return torch.stack(xs, dim=dim)
    except Exception:
        return xs

def collate_fused_view(batch):
    """
    batch is a list of:
      - feats                    : [T, S, 4D]           (always)
      - depth (optional)         : [T, H, W] or list    (when return_view=True)
      - pose  (optional)         : [T, 7]   or list     (when return_view=True)
    Returns:
      feats_b   : [B, T, S, 4D]
      depth_b   : [B, T, H, W] or list/None
      pose_b    : [B, T, 7] or list/None
    """
    if isinstance(batch[0], torch.Tensor):
        # Old path: dataset returns only feats
        feats_b = torch.stack(batch, dim=0)
        return feats_b

    # New path: dataset returns (feats, depth, pose)
    feats_list, depth_list, pose_list = [], [], []
    for item in batch:
        feats_list.append(item[0])
        depth_list.append(item[1] if len(item) > 1 else None)
        pose_list.append(item[2] if len(item) > 2 else None)

    feats_b = torch.stack(feats_list, dim=0)  # [B, T, S, 4D]
    depth_b = _stack_or_list(depth_list, dim=0)   # [B, T, H, W] or list/None
    pose_b  = _stack_or_list(pose_list,  dim=0)   # [B, T, 7]    or list/None
    return feats_b, depth_b, pose_b


if not DEBUG:
    def make_loaders(args):

        g = torch.Generator().manual_seed(42)

        train_ds = SelectedFusedImgFeatConcatDataset(args.train_features_path)
        val_ds   = SelectedFusedImgFeatConcatDataset(args.val_features_path, return_view=True)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fused_view
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=getattr(args, "num_workers_val", None) or args.num_workers,
            pin_memory=True, collate_fn=collate_fused_view
        )

        return train_loader, val_loader
else:
    # For debugging
    def make_loaders(args):

            g = torch.Generator().manual_seed(42)

            # -------------------------------
            # DEBUG MODE: use the SAME items for train and val
            # - Train: feats only (return_view=False)
            # - Val  : feats + (depth, pose) (return_view=True)
            # We pick items from a single root (val path by default) so
            # indices match exactly across the two dataset instances.
            # -------------------------------

            # choose which root to pull from for both
            root = args.val_features_path  # or args.train_features_path

            # one dataset instance to derive the indices deterministically
            base_ds = SelectedFusedImgFeatConcatDataset(root, return_view=False)
            n = min(len(base_ds), int(getattr(args, "debug_num_samples", 4)))
            idx = torch.arange(n).tolist()  # first N samples; or torch.randperm for random

            # build train/val with identical indices but different return_view
            train_ds = Subset(
                SelectedFusedImgFeatConcatDataset(root, return_view=False), idx
            )
            val_ds = Subset(
                SelectedFusedImgFeatConcatDataset(root, return_view=True), idx
            )

            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=False,
                collate_fn=collate_fused_view
            )
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=getattr(args, "num_workers_val", None) or args.num_workers,
                pin_memory=True, drop_last=False,       # keep complete eval set
                collate_fn=collate_fused_view
            )
            return train_loader, val_loader


# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------
def parse_tuple(x): return tuple(map(int, x.split(',')))

def build_argparser():
    p = argparse.ArgumentParser()

    # Data / features
    p.add_argument('--train_features_path', type=str, default="/workspace/raid/jevers/cut3r_features/waymo/fused_img_tokens_224/train")
    p.add_argument('--val_features_path',   type=str, default="/workspace/raid/jevers/cut3r_features/waymo/fused_img_tokens_224/validation")
    p.add_argument('--feat_hw', type=parse_tuple, default=(14,14))
    p.add_argument('--cached_feature_dim', type=int, default=3328)
    p.add_argument('--pose_token_mode', action='store_true', default=False)
    

    # Sequence / loader
    p.add_argument('--sequence_length', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--num_workers_val', type=int, default=None)

    # Masking / loss
    p.add_argument('--train_mask_frames', type=int, default=1)
    p.add_argument('--train_mask_mode', type=str, default='full_mask',
                   choices=['full_mask','arccos','linear','cosine','square'])
    p.add_argument('--masking', type=str, default='simple_replace',
                   choices=['half_half','simple_replace','half_half_previous'])
    p.add_argument('--loss_type', type=str, default='SmoothL1', choices=['SmoothL1','MSE','L1'])
    p.add_argument('--beta_smoothl1', type=float, default=0.1)
    p.add_argument('--output_activation', type=str, default='none', choices=['none','sigmoid'])

    # Model capacity
    p.add_argument('--hidden_dim', type=int, default=1152)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--layers', type=int, default=12)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--attn_dropout', type=float, default=0.3)
    p.add_argument('--seperable_attention', action='store_true', default=True)
    p.add_argument('--seperable_window_size', type=int, default=1)
    p.add_argument('--use_first_last', action='store_true', default=False)
    p.add_argument('--use_fc_bias', action='store_true', default=False)

    # PCA (off by default)
    p.add_argument('--pca_ckpt', type=str, default=None)

    # Train loop
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--accum_iter', type=int, default=1)
    p.add_argument('--num_gpus', type=int, default=1)
    p.add_argument('--precision', type=str, default='32-true', choices=['16-true','16-mixed','32-true','32'])
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--lr_base', type=float, default=2e-5, help='LR at effective batch size 8')
    p.add_argument('--gclip', type=float, default=1.0)
    p.add_argument('--dst_path', type=str, default=None)
    p.add_argument('--eval_freq', type=int, default=1)
    p.add_argument('--vis_attn', action='store_true', default=False)
    p.add_argument('--warmup_p', type=float, default=0.0,
               help='Fraction of max_steps used for LR warmup')

    # Evaluation toggles
    p.add_argument('--evaluate', action='store_true', default=False,
                   help='Run validation-only using a checkpoint (best/last/--ckpt).')
    p.add_argument('--eval_last', action='store_true', default=False,
                   help='Use the "last.ckpt" if true; otherwise use best.')
    p.add_argument('--ckpt', type=str, default=None,
                   help='Explicit checkpoint path to load for eval or resume fit.')
    p.add_argument('--eval_ckpt_only', action='store_true', default=False,
                   help='Skip training and only evaluate the given/best/last checkpoint.')

    # Depth eval settings
    p.add_argument('--depth_eval_align_mode', type=str, default="scale&shift",
                   choices=["metric", "scale", "scale&shift", "median"])
    p.add_argument('--depth_eval_max_depth', type=float, default=70.0)
    p.add_argument('--depth_post_clip_max', type=float, default=70.0)

    # 3D / pose eval cadence
    p.add_argument('--eval3d_every_n_epochs', type=int, default=5 if not DEBUG else 1)
    p.add_argument('--step', type=int, default=1)
    p.add_argument('--eval_midterm', action='store_true', default=False)
    p.add_argument('--evaluate_baseline', action='store_true', default=False)
    p.add_argument('--evaluate_oracle', action='store_true', default=False)

    return p

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def normalize_precision_arg(p):
    # lightning accepts "32-true" etc; keep your special-case "32" as int
    return 32 if p == '32' else p

def setup_args_defaults(args):
    # Fixed flags for external features
    args.eval_modality = None
    args.feature_extractor = 'external'
    args.d_layers = [0]
    args.img_size = (0,0)
    args.crop_feats = False
    args.sliding_window_inference = False
    args.high_res_adapt = False
    args.single_step_sample_train = True
    # depth cfg already parsed above

    args.pose_token_mode = True

    args.eval_ckpt_only = False
    args.ckpt = None
    args.warmup_p = 0.04
    
    args.eval_mode = False  # will flip to True when evaluating
    return args

def setup_dist_env(args):
    pl.seed_everything(args.seed, workers=True)
    if args.num_gpus > 1:
        import torch
        env = os.environ
        if 'RANK' in env and 'WORLD_SIZE' in env:
            args.rank = int(env['RANK'])
            args.world_size = int(env['WORLD_SIZE'])
            args.gpu = int(env['LOCAL_RANK'])
        elif 'SLURM_PROCID' in env:
            args.rank = int(env['SLURM_PROCID'])
            args.world_size = int(env['SLURM_NTASKS'])
            args.gpu = int(env['SLURM_LOCALID'])
            gpn = int(env['SLURM_GPUS_ON_NODE'])
            assert gpn == torch.cuda.device_count()
            args.node = args.rank // gpn
        else:
            args.rank = 0
            args.world_size = args.num_gpus
            args.gpu = 0
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        import torch
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.node = 0
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'rank={args.rank} - world_size={args.world_size} - gpu={args.gpu} - device={args.device}')
    return args

def scale_and_set_lr_args(args, steps_per_epoch):
    """
    Mimic the original logic:

      max_steps = max_epochs * (len(train_loader) // (num_gpus * accum_iter))
      effective_batch_size = batch_size * world_size * accum_iter
      lr = lr_base * (effective_batch_size / 8), where lr_base is defined for EBS=8
    """
    # Steps per global optimizer update
    # (len(train_loader) is per-rank, but your original formula divided by num_gpus anyway)
    global_steps_per_epoch = max(
        1,
        steps_per_epoch // max(1, args.num_gpus * args.accum_iter)
    )
    args.max_steps = args.max_epochs * global_steps_per_epoch

    # Effective global batch size (this is what you had originally)
    args.effective_batch_size = args.batch_size * args.world_size * args.accum_iter

    # Linear LR scaling w.r.t. effective batch size (ref = 8)
    #args.lr = (args.lr_base * args.effective_batch_size) / 8.0
    args.lr = 1.0e-4

    # Warmup steps (if you use warmup_p)
    args.warmup_steps = int(getattr(args, "warmup_p", 0.0) * args.max_steps)

    print(
        f'Effective batch size: {args.effective_batch_size} '
        f'lr_base={args.lr_base} lr={args.lr} '
        f'max_epochs={args.max_epochs} max_steps={args.max_steps}'
    )
    return args

def build_trainer(args):
    if args.dst_path is None:
        args.dst_path = os.getcwd()

    tb_logger = TensorBoardLogger(
        save_dir=args.dst_path,
        name="lightning_logs",
        default_hp_metric=True
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        monitor='val/loss', mode='min', save_top_k=1, save_last=True, filename='{epoch}-{step}-{val_loss:.5f}'
    )
    if args.max_epochs < args.eval_freq:
        args.eval_freq = 1

    trainer = pl.Trainer(
        accelerator='gpu',
        strategy=(DDPStrategy(find_unused_parameters=True) if args.num_gpus > 1 else 'auto'),
        devices=args.num_gpus,
        callbacks=[ckpt_cb, lr_cb],
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gclip,
        default_root_dir=args.dst_path,
        precision=normalize_precision_arg(args.precision),
        log_every_n_steps=1,
        check_val_every_n_epoch=args.eval_freq,
        accumulate_grad_batches=args.accum_iter,
        logger=tb_logger,
        inference_mode=False
    )
    print(f"TensorBoard logdir: {tb_logger.log_dir}")
    return trainer, tb_logger, ckpt_cb

def resolve_checkpoint_for_eval(args, ckpt_cb, trainer_logdir):
    """
    Priority:
      1) explicit --ckpt
      2) best (ModelCheckpoint.best_model_path)
      3) last (ModelCheckpoint.last_model_path or *last.ckpt in checkpoints/)
    """
    if args.ckpt and os.path.isfile(args.ckpt):
        return args.ckpt

    # Try to get from callback (works if same Trainer/dir)
    best = getattr(ckpt_cb, "best_model_path", "") or ""
    last = getattr(ckpt_cb, "last_model_path", "") or ""

    # If we’re resuming evaluation in a fresh process, scan logdir/checkpoints
    ckpt_dir = None
    # pl places ckpts under default_root_dir/lightning_logs/version_x/checkpoints
    if trainer_logdir and os.path.isdir(trainer_logdir):
        ckpt_dir = os.path.join(trainer_logdir, "checkpoints")

    if not best and ckpt_dir:
        # find best by filename pattern (optional)
        # fallback to newest file if no monitored metric in filename
        files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
        if files:
            best = files[0]

    if args.eval_last:
        # prefer explicit "last.ckpt" if present in dir
        if ckpt_dir:
            candidates = glob.glob(os.path.join(ckpt_dir, "last.ckpt"))
            if candidates:
                last = candidates[0]
        return last or best

    # default: best
    return best or last

def write_results(log_dir, metrics_dict):
    path = os.path.join(log_dir, 'results.txt')
    with open(path, 'w') as f:
        for k, v in metrics_dict.items():
            f.write(f'{k}: {v}\n')
    print(f'Results saved at: {path}')

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    from src.dino_f import Dino_f  # use your modified class with external features + 3D/pose eval

    parser = build_argparser()
    args = parser.parse_args()
    args.precision = normalize_precision_arg(args.precision)
    args = setup_args_defaults(args)
    args = setup_dist_env(args)

    # Data
    train_loader, val_loader = make_loaders(args)
    steps_per_epoch = len(train_loader)
    args = scale_and_set_lr_args(args, steps_per_epoch)

    # Model
    model = Dino_f(args)

    # Logger/Trainer
    trainer, tb_logger, ckpt_cb = build_trainer(args)

    # Log a couple of hparams
    tb_logger.log_hyperparams({"batch_size": args.batch_size})

    # -------------------- Train or Skip --------------------
    if not args.eval_ckpt_only:
        # resume-from-ckpt training if --ckpt provided; else start fresh
        if args.ckpt:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt)
        else:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        # Eval-only mode; do not fit
        args.evaluate = True
        args.eval_last = bool(args.eval_last)  # no-op, just explicit

    # -------------------- Evaluation --------------------
    if args.evaluate or args.eval_ckpt_only:
        # Turn on the 3D+pose eval route inside your LightningModule
        args.eval_mode = True

        # Decide which checkpoint to evaluate
        ckpt_path = resolve_checkpoint_for_eval(args, ckpt_cb, trainer.logger.log_dir if trainer.logger else None)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            # If nothing found but user gave --ckpt earlier and we just trained, try best_model_path now
            ckpt_path = getattr(ckpt_cb, "best_model_path", "") or getattr(ckpt_cb, "last_model_path", "")
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError("Could not find a checkpoint to evaluate. "
                                    "Pass --ckpt, or ensure training saved checkpoints.")

        print(f'[Eval] Loading checkpoint: {ckpt_path}')
        model = Dino_f.load_from_checkpoint(ckpt_path, args=args, strict=False, map_location="cpu")
        model.to(args.device).eval()

        # validate() returns a list of dicts (one per dataloader)
        out_metrics = trainer.validate(model=model, dataloaders=val_loader, verbose=False) or [{}]
        m0 = out_metrics[0] if out_metrics else {}

        # Gather the key results you care about
        results = {}
        for k in ("val/mean_loss", "val/loss", "val/ff_mse", "val/ff_mae", "val/ff_cos",
                  "val/depth_absrel", "val/depth_delta1",
                  "val/pose_ATE_m", "val/pose_RPE_t_m", "val/pose_RPE_r_deg"):
            if k in m0:
                results[k] = m0[k]

        # Also keep a plain 'Mean Loss' line for legacy readers
        if "val/mean_loss" in m0:
            results["Mean Loss"] = m0["val/mean_loss"]

        write_results(trainer.log_dir, results)

if __name__ == "__main__":
    main()