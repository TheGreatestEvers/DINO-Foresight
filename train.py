import argparse
import os
import glob
import torch
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from src.dino_f_original import Dino_f
from src.data_waymo import WaymoSimpleVideoDepthDataset  # <-- your dataset


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def parse_tuple(x): 
    return tuple(map(int, x.split(',')))

def parse_list(x): 
    return list(map(int, x.split(',')))


# --------------------------------------------------------------------------
# Argparser
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser()

    # ---------------- Data ----------------
    p.add_argument('--train_data_path', type=str, default="/workspace/raid/jevers/cut3r_processed_waymo/train/",
                   help='Root directory for Waymo TRAIN scenes (jpg/exr/npz).')
    p.add_argument('--val_data_path', type=str, default="/workspace/raid/jevers/cut3r_processed_waymo/validation/",
                   help='Root directory for Waymo VAL scenes. If None, uses train_data_path.')
    p.add_argument('--dst_path', type=str, default=None,
                   help='Root output dir (checkpoints, logs).')

    p.add_argument('--img_size', type=parse_tuple, default=(224, 224))
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--num_workers_val', type=int, default=None,
                   help='If None, uses num_workers.')
    p.add_argument('--sequence_length', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=4)

    # Temporal sampling
    p.add_argument('--temporal_stride', type=int, default=2,
                   help='Stride between frames inside a sequence.')
    p.add_argument('--overlap_step', type=int, default=5,
                   help='Sliding-window step in raw frames; if None, equals temporal_stride.')
    p.add_argument('--non_overlapping', action='store_true', default=False,
                   help='If True, use disjoint windows (no overlap).')
    p.add_argument('--midterm_horizon', type=int, default=3,
                   help='How many strides after last context frame the target lies for midterm eval.')

    # ---------------- Model / transformer ----------------
    p.add_argument('--feature_extractor', type=str, default='dino',
                   choices=['dino', 'eva2-clip', 'sam'])
    p.add_argument('--dinov2_variant', type=str, default='vitb14_reg',
                   choices=['vits14_reg','vitb14_reg'])
    p.add_argument('--d_layers', type=parse_list, default=[2,5,8,11])
    p.add_argument('--hidden_dim', type=int, default=768)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--layers', type=int, default=12)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--attn_dropout', type=float, default=0.3)
    p.add_argument('--masking', type=str, default='simple_replace',
                   choices=['half_half', 'simple_replace', 'half_half_previous'])
    p.add_argument('--train_mask_frames', type=int, default=1)
    p.add_argument('--train_mask_mode', type=str, default='full_mask',
                   choices=['full_mask', 'arccos', 'linear', 'cosine', 'square'])
    p.add_argument('--loss_type', type=str, default='SmoothL1',
                   choices=['SmoothL1', 'MSE', 'L1'])
    p.add_argument('--beta_smoothl1', type=float, default=0.1)
    p.add_argument('--output_activation', type=str, default='none',
                   choices=['none', 'sigmoid'])
    p.add_argument('--seperable_attention', action='store_true', default=False)
    p.add_argument('--seperable_window_size', type=int, default=1)
    p.add_argument('--use_first_last', action='store_true', default=False)
    p.add_argument('--use_fc_bias', action='store_true', default=False)
    p.add_argument('--down_up_sample', action='store_true', default=False)
    p.add_argument('--pca_ckpt', type=str, default=None)
    p.add_argument('--crop_feats', action='store_true', default=False)
    p.add_argument('--sliding_window_inference', action='store_true', default=False)
    p.add_argument('--high_res_adapt', action='store_true', default=False,
                   help='If True, adapts pos embeddings to higher resolution.')

    # DPT head (for eval_modality == depth/segm/normals)
    p.add_argument('--num_classes', type=int, default=256,
                   choices=[19, 256, 3],
                   help='19 segm, 256 depth bins, 3 normals.')
    p.add_argument('--use_bn', action='store_true', default=False)
    p.add_argument('--use_cls', action='store_true', default=False)
    p.add_argument('--nfeats', type=int, default=256)
    p.add_argument('--dpt_out_channels', type=parse_list, default=[128, 256, 512, 512])
    p.add_argument('--head_ckpt', type=str, default=None)

    # ---------------- Training loop ----------------
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--single_step_sample_train', action='store_true', default=False)
    p.add_argument('--precision', type=str, default='32-true',
                   choices=['16-true','16-mixed','32-true','32'])
    p.add_argument('--ckpt', type=str, default=None,
                   help='Checkpoint to resume training from.')
    p.add_argument('--num_gpus', type=int, default=1)
    p.add_argument('--accum_iter', type=int, default=1)
    p.add_argument('--warmup_p', type=float, default=0.0)
    p.add_argument('--lr_base', type=float, default=1e-3)
    p.add_argument('--gclip', type=float, default=1.0)
    p.add_argument('--eval_freq', type=int, default=1)
    p.add_argument('--vis_attn', action='store_true', default=False)

    # ---------------- Evaluation toggles ----------------
    p.add_argument('--evaluate', action='store_true', default=False,
                   help='Run validation-only using a checkpoint.')
    p.add_argument('--eval_last', action='store_true', default=False,
                   help='Use last checkpoint instead of best.')
    p.add_argument('--ckpt_eval', type=str, default=None,
                   help='Explicit checkpoint for eval only.')
    p.add_argument('--eval_ckpt_only', action='store_true', default=False,
                   help='Skip training and only evaluate.')
    p.add_argument('--eval_mode_during_training', action='store_true',
                   help='If True, uses eval_mode during training validation loop')
    p.add_argument('--eval_midterm', action='store_true', default=False)
    p.add_argument('--evaluate_baseline', action='store_true', default=False)
    p.add_argument('--eval_mode', action='store_true', default=False)

    # Modality: here we focus on depth eval
    p.add_argument('--eval_modality', type=str, default='depth',
                   choices=[None, 'segm', 'depth', 'surface_normals'],
                   help='For Waymo we typically want depth.')

    p.add_argument('--step', type=int, default=1)

    return p


def set_args(args):
    args.eval_ckpt_only = True
    args.eval_modality = "depth"
    args.ckpt_eval = "/workspace/DINO-Foresight/waymo_dinof/version_15/checkpoints/last.ckpt"


# --------------------------------------------------------------------------
# Dist / device
# --------------------------------------------------------------------------
def normalize_precision_arg(p):
    return 32 if p == '32' else p

def setup_dist_env(args):
    pl.seed_everything(args.seed, workers=True)

    if args.num_gpus > 1:
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
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.node = 0
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'rank={args.rank} - world_size={args.world_size} - gpu={args.gpu} - device={args.device}')
    return args


# --------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------
def make_loaders(args):
    """Create train/val DataLoaders from two Waymo roots."""

    # TRAIN: only frames
    train_ds = WaymoSimpleVideoDepthDataset(
        root=args.train_data_path,
        args=args,
        temporal_stride=args.temporal_stride,
        overlap_step=args.overlap_step,
        non_overlapping=args.non_overlapping,
        with_gt=False,
        midterm=False,
        midterm_horizon=args.midterm_horizon,
    )

    # VAL: frames + (gt_img, gt_depth) when eval_modality == depth
    with_gt_val = (args.eval_modality == 'depth')
    val_ds = WaymoSimpleVideoDepthDataset(
        root=args.val_data_path,
        args=args,
        temporal_stride=args.temporal_stride,
        overlap_step=args.overlap_step,
        non_overlapping=args.non_overlapping,
        with_gt=with_gt_val,
        midterm=args.eval_midterm,
        midterm_horizon=args.midterm_horizon,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers_val if args.num_workers_val is not None else args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# --------------------------------------------------------------------------
# LR scaling & trainer
# --------------------------------------------------------------------------
def scale_and_set_lr_args(args, steps_per_epoch):
    # match your desired behavior: fixed lr = 1.6e-4
    global_steps_per_epoch = max(
        1,
        steps_per_epoch // max(1, args.num_gpus * args.accum_iter)
    )
    args.max_steps = args.max_epochs * global_steps_per_epoch

    args.effective_batch_size = args.batch_size * args.world_size * args.accum_iter

    # fixed LR as requested
    args.lr = 1.6e-4

    args.warmup_steps = int(args.warmup_p * args.max_steps)

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
        name="waymo_dinof",
        default_hp_metric=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    ckpt_cb = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        filename='{epoch}-{step}-{val_loss:.5f}',
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
        log_every_n_steps=5,
        check_val_every_n_epoch=args.eval_freq,
        accumulate_grad_batches=args.accum_iter,
        logger=tb_logger,
        inference_mode=False,
    )
    print(f"TensorBoard logdir: {tb_logger.log_dir}")
    return trainer, tb_logger, ckpt_cb


# --------------------------------------------------------------------------
# Checkpoint resolution for eval-only
# --------------------------------------------------------------------------
def resolve_checkpoint_for_eval(args, ckpt_cb, trainer_logdir):
    if args.ckpt_eval and os.path.isfile(args.ckpt_eval):
        return args.ckpt_eval

    best = getattr(ckpt_cb, "best_model_path", "") or ""
    last = getattr(ckpt_cb, "last_model_path", "") or ""

    ckpt_dir = None
    if trainer_logdir and os.path.isdir(trainer_logdir):
        ckpt_dir = os.path.join(trainer_logdir, "checkpoints")

    if not best and ckpt_dir:
        files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime, reverse=True)
        if files:
            best = files[0]

    if args.eval_last:
        if ckpt_dir:
            candidates = glob.glob(os.path.join(ckpt_dir, "last.ckpt"))
            if candidates:
                last = candidates[0]
        return last or best

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
    parser = build_argparser()
    args = parser.parse_args()
    args.precision = normalize_precision_arg(args.precision)

    # eval_mode used internally by Dino_f
    args.eval_mode = args.eval_mode_during_training

    args = setup_dist_env(args)

    # Data
    train_loader, val_loader = make_loaders(args)
    steps_per_epoch = len(train_loader)
    args = scale_and_set_lr_args(args, steps_per_epoch)

    args = set_args(args)

    # Model
    if not args.high_res_adapt:
        model = Dino_f(args)
    else:
        model = Dino_f.load_from_checkpoint(args.ckpt, args=args, strict=False, map_location="cpu")

    # Trainer
    trainer, tb_logger, ckpt_cb = build_trainer(args)
    tb_logger.log_hyperparams({"batch_size": args.batch_size, "sequence_length": args.sequence_length})

    # ---------------- Training ----------------
    if not args.eval_ckpt_only:
        if args.ckpt and not args.high_res_adapt:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt)
        else:
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        args.evaluate = True
        args.eval_last = bool(args.eval_last)

    # ---------------- Evaluation ----------------
    if args.evaluate or args.eval_ckpt_only:
        args.eval_mode = True  # tell Dino_f to run in eval mode (depth metrics etc.)

        ckpt_path = resolve_checkpoint_for_eval(args, ckpt_cb, trainer.logger.log_dir if trainer.logger else None)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            ckpt_path = getattr(ckpt_cb, "best_model_path", "") or getattr(ckpt_cb, "last_model_path", "")
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError("Could not find a checkpoint to evaluate.")

        print(f'[Eval] Loading checkpoint: {ckpt_path}')
        model = Dino_f.load_from_checkpoint(ckpt_path, args=args, strict=False, map_location="cpu")
        model.to(args.device).eval()

        out_metrics = trainer.validate(model=model, dataloaders=val_loader, verbose=False) or [{}]
        m0 = out_metrics[0] if out_metrics else {}

        results = {}
        # generic mean loss
        if "val/mean_loss" in m0:
            results["Mean Loss"] = m0["val/mean_loss"]

        # depth-specific metrics (if eval_modality == depth and DPT head used)
        #for k in ("d1", "d2", "d3", "abs_rel", "rmse", "rmse_log", "sq_rel", "log_10", "silog"):
        #    if k in m0:
        #        results[k] = m0[k]
        for k in ("depth_absrel", "depth_delta1", "depth_delta2", "depth_delta3", "depth_rmse", "depth_silog"):
            if k in m0:
                results[k] = m0[k]

        write_results(trainer.log_dir, results)


if __name__ == "__main__":
    main()