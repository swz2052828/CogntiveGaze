import argparse

from .explain import explain
from .meta import meta_train
from .metacompare import metacompare
from .svr_search import svrsearch
from .training import train


def add_common_args(parser):
    parser.add_argument(
        "--data-path",
        default=None,
        help="Dataset root that contains <mean-path>/metadata.mat. Used for the old cropped layout too.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Explicit path to metadata.mat. Use this if raw/synthetic images live outside the processed dataset.",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Root for uncropped original images laid out as <raw-root>/<recording>/<frame>.jpg.",
    )
    parser.add_argument(
        "--synthetic-root",
        default=None,
        help="Root for uncropped synthetic images laid out as <synthetic-root>/<recording>/<frame>.jpg.",
    )
    parser.add_argument("--mean-path", default="mean7")
    parser.add_argument("--raw-folder", default="appleFace")
    parser.add_argument("--synthetic-folder", default="appleFaceFake")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--allow-missing-synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision autocast (bf16 where supported, e.g. RTX 5090; "
             "fp16 with loss scaling otherwise, e.g. RTX 2070 Super). Off by default "
             "so results are bit-for-bit unchanged unless you opt in.",
    )
    parser.add_argument(
        "--no-tf32",
        action="store_true",
        help="Disable TF32 matmul/conv. TF32 is on by default (a free speedup on "
             "Ampere+/Blackwell, ignored on Turing) with negligible accuracy effect.",
    )
    parser.add_argument(
        "--eye-path",
        default=None,
        help="Root for preprocessed eye crops (multistream only). "
             "Layout: <eye-path>/<rec>/appleLeftEye/<frame>.jpg.",
    )
    parser.add_argument("--face-folder", default="appleFace")
    parser.add_argument("--left-eye-folder", default="appleLeftEye")
    parser.add_argument("--right-eye-folder", default="appleRightEye")
    parser.add_argument(
        "--eye-size",
        type=int,
        default=224,
        help="Eye crop side length (multistream only). Matches existing CNN pipeline.",
    )
    parser.add_argument(
        "--use-grid",
        action="store_true",
        help="Include face-grid input (multistream only). Off by default because "
             "for seated subjects far from the camera the grid is near-constant "
             "per subject and provides no within-subject signal.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=25,
        help="Side length of the face-grid (multistream + --use-grid only).",
    )


def _gamma_arg(s):
    """argparse type for SVR gamma: accept 'scale' / 'auto' or a float."""
    if s in ("scale", "auto"):
        return s
    return float(s)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a ViT gaze regressor and output segmented regions that contribute "
            "to a true gaze coordinate. Training uses recording-level K-fold cross validation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_common_args(train_parser)
    train_parser.add_argument("--out-path", default="./vit_gaze_segmenter_output")
    train_parser.add_argument(
        "--input-mode",
        choices=("raw", "synthetic", "paired", "multistream"),
        default="raw",
        help=(
            "raw is recommended for the single-image model. paired keeps the "
            "older raw+synthetic fusion. multistream uses face + left eye + "
            "right eye (+ optional face-grid) and selects a backbone via --backbone."
        ),
    )
    train_parser.add_argument(
        "--backbone",
        choices=("vit", "itracker", "mobilenet_v3", "affnet", "mgazenet"),
        default="vit",
        help=(
            "Multistream backbone. vit (default) = shared ViT-B/16 with optional "
            "grid. itracker / mobilenet_v3 / affnet / mgazenet are CNN baselines "
            "ported from the project's reference implementations; all four "
            "require --use-grid (architectures use or condition on the face-grid)."
        ),
    )
    train_parser.add_argument("--weights", choices=("none", "imagenet"), default="none")
    train_parser.add_argument("--freeze-encoder", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine", "step"),
        default="none",
        help=(
            "LR schedule applied within each fold. "
            "'cosine' = CosineAnnealingLR(T_max=epochs, eta_min=0); "
            "'step' = StepLR(step_size=--step-size, gamma=--step-gamma). "
            "Default 'none' keeps constant LR."
        ),
    )
    train_parser.add_argument(
        "--step-size",
        type=int,
        default=3,
        help="Epoch interval between LR drops for --lr-scheduler step. Default 3.",
    )
    train_parser.add_argument(
        "--step-gamma",
        type=float,
        default=0.5,
        help="Multiplicative LR decay per step for --lr-scheduler step. Default 0.5.",
    )
    train_parser.add_argument(
        "--augment",
        choices=("none", "light", "medium"),
        default="none",
        help=(
            "Data augmentation applied to multistream crops (face + eyes) during training only. "
            "light: ColorJitter(0.2/0.2/0.2) + RandomResizedCrop(scale=0.90-1.0). "
            "medium: ColorJitter(0.4/0.4/0.4) + RandomResizedCrop(scale=0.85-1.0) + RandomGrayscale(p=0.05). "
            "No horizontal flip: gaze labels are screen-relative. Off by default."
        ),
    )
    train_parser.add_argument(
        "--subject-adv",
        action="store_true",
        help=(
            "Domain-adversarial subject invariance (DANN). Multistream only. "
            "Attaches a subject-ID classifier to the fused feature through a "
            "gradient-reversal layer so the encoder learns subject-invariant "
            "gaze features (regularizes the small cohort; composes with, does "
            "not replace, per-subject calibration). Training-only; the "
            "inference checkpoint is unchanged. Off by default."
        ),
    )
    train_parser.add_argument(
        "--adv-weight",
        type=float,
        default=0.1,
        help="Ceiling for the gradient-reversal strength lambda, which ramps "
             "0 -> this over training (Ganin schedule). Default 0.1.",
    )
    train_parser.add_argument(
        "--adv-warmup-frac",
        type=float,
        default=1.0,
        help="Fraction of total training steps over which lambda ramps to "
             "--adv-weight. 1.0 (default) ramps across the whole run; smaller "
             "values reach full strength sooner.",
    )
    train_parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile for extra throughput (falls back "
             "to eager mode if the backend/GPU does not support it).",
    )
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Run only one 0-based fold. Useful with Slurm array jobs.",
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stop training within a fold if the monitored validation "
             "metric does not improve for this many epochs. Off by default. "
             "The best checkpoint is still saved.",
    )
    train_parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum improvement in the monitored metric to count as progress "
             "(0 = strict improvement, the default). Useful with --patience to "
             "ignore noisy single-epoch dips. Also gates best-checkpoint saving.",
    )
    train_parser.add_argument(
        "--early-stop-metric",
        choices=("val_loss", "val_error"),
        default="val_error",
        help="Which validation metric drives --patience and best-checkpoint "
             "selection. Defaults to val_error (val_coord_error), the metric "
             "actually reported; pass val_loss to keep the prior behaviour.",
    )
    train_parser.add_argument(
        "--log-file",
        default=None,
        help="Also append every training/optimization log line to this file "
             "(timestamped, line-buffered so you can `tail -f` it live). Stdout is "
             "unchanged. Handy on Slurm, where stdout is often buffered or split "
             "across array-task .out files. Combine with --fold-index for a "
             "per-fold log, e.g. --log-file train_fold${SLURM_ARRAY_TASK_ID}.log.",
    )
    train_parser.add_argument("--seed", type=int, default=42)

    meta_parser = subparsers.add_parser(
        "metatrain",
        help="Meta-learn a per-subject calibration adapter (FiLM/LoRA) with "
             "FOMAML/ANIL. Multistream + a backbone exposing forward_features "
             "(vit). Reports pre- vs post-adaptation coord error on held-out "
             "recordings -- the comparison point against per-subject SVR.",
    )
    add_common_args(meta_parser)
    meta_parser.add_argument("--out-path", default="./vit_gaze_meta_output")
    meta_parser.add_argument("--input-mode", choices=("multistream",), default="multistream")
    meta_parser.add_argument(
        "--backbone",
        choices=("vit", "itracker", "mobilenet_v3", "affnet", "mgazenet"),
        default="vit",
        help="All multistream backbones expose forward_features and are "
             "supported. The CNN baselines (itracker/mobilenet_v3/affnet/"
             "mgazenet) require --use-grid. Use --init-checkpoint to start from "
             "a trained encoder (the frozen encoder is otherwise un-tuned).",
    )
    meta_parser.add_argument("--weights", choices=("none", "imagenet"), default="imagenet")
    meta_parser.add_argument("--freeze-encoder", action="store_true")
    meta_parser.add_argument(
        "--init-checkpoint", default=None,
        help="Load encoder+head from a prior `train` checkpoint so meta-learning "
             "starts from gaze-tuned features (strongly recommended; otherwise "
             "the frozen encoder is only ImageNet/random).",
    )
    meta_parser.add_argument(
        "--adapter", choices=("film", "lora"), default="film",
        help="Per-subject adapter meta-learned for calibration. film: (gamma,beta) "
             "scale+shift on the fused feature (tiny, robust at small K). lora: "
             "low-rank residual (more expressive, higher overfit risk at small K).",
    )
    meta_parser.add_argument("--lora-rank", type=int, default=8)
    meta_parser.add_argument("--lora-alpha", type=float, default=8.0)
    meta_parser.add_argument(
        "--meta-support", type=int, default=16,
        help="K: calibration frames per subject used to adapt (support set).",
    )
    meta_parser.add_argument(
        "--meta-query", type=int, default=32,
        help="Query frames per task per outer step (the post-adaptation loss).",
    )
    meta_parser.add_argument(
        "--inner-steps", type=int, default=20,
        help="Inner-loop SGD steps used to adapt the FiLM/LoRA fast-weights "
             "from the meta-learned init on each task's support set. The empirical "
             "sweet spot for FiLM at fused_dim~=2300 is ~20 (smoke runs at 5 left "
             "the adapter essentially unmoved; the K-knob did nothing).",
    )
    meta_parser.add_argument(
        "--inner-lr", type=float, default=1.0,
        help="Inner-loop SGD learning rate. With FiLM (gamma init=1, beta init=0) "
             "the adapter needs an aggressive lr to actually move within --inner-steps; "
             "1.0 worked in smoke (1e-2 was too conservative -- the adapter returned "
             "the base prediction regardless of K).",
    )
    meta_parser.add_argument("--outer-lr", type=float, default=1e-3)
    meta_parser.add_argument(
        "--adapt-steps", type=int, default=None,
        help="Inner steps used at enrollment/eval (defaults to --inner-steps).",
    )
    meta_parser.add_argument("--meta-iters", type=int, default=2000)
    meta_parser.add_argument("--tasks-per-batch", type=int, default=4)
    meta_parser.add_argument("--print-freq", type=int, default=100)
    meta_parser.add_argument("--batch-size", type=int, default=64)
    meta_parser.add_argument("--num-workers", type=int, default=4)
    meta_parser.add_argument("--folds", type=int, default=5)
    meta_parser.add_argument("--fold-index", type=int, default=None)
    meta_parser.add_argument("--seed", type=int, default=42)
    meta_parser.add_argument("--log-file", default=None)

    cmp_parser = subparsers.add_parser(
        "metacompare",
        help="Apples-to-apples per-subject calibration comparison at matched K: "
             "base / per-subject SVR / meta-learned adapter, scored on the same "
             "support/query draws.",
    )
    add_common_args(cmp_parser)
    cmp_parser.add_argument("--input-mode", choices=("multistream",), default="multistream")
    cmp_parser.add_argument(
        "--backbone", default=None,
        help="Accepted for symmetry with train / metatrain (so the same sbatch "
             "driver can pass it to every subcommand). Ignored: the actual "
             "backbone is read from each loaded checkpoint, which is the "
             "authoritative source.",
    )
    cmp_parser.add_argument("--base-checkpoint", required=True,
                            help="Trained `train` checkpoint (encoder+head).")
    cmp_parser.add_argument("--meta-checkpoint", required=True,
                            help="Trained `metatrain` checkpoint (encoder+head+adapter init).")
    cmp_parser.add_argument(
        "--meta-adv-checkpoint", default=None,
        help="Optional second `metatrain` checkpoint built on subject-adversarial "
             "features (i.e. metatrain --init-checkpoint <a --subject-adv run>). "
             "When given, a 4th method 'meta_adv' is scored on the same draws, "
             "yielding the four-way base / svr / meta / meta_adv comparison.")
    cmp_parser.add_argument("--k", type=int, default=16,
                            help="Calibration frames per subject (matched across the three methods).")
    cmp_parser.add_argument("--trials", type=int, default=5,
                            help="Random support/query draws per recording; results are averaged.")
    cmp_parser.add_argument(
        "--inner-steps", type=int, default=20,
        help="Inner-loop SGD steps used when adapting the meta adapter on K "
             "support frames. Should match the value used at metatrain time "
             "(both default to 20 -- see metatrain --inner-steps for the rationale).",
    )
    cmp_parser.add_argument(
        "--inner-lr", type=float, default=1.0,
        help="Inner-loop SGD lr; should match metatrain --inner-lr (both 1.0).",
    )
    cmp_parser.add_argument("--svr-C", type=float, default=1.0)
    cmp_parser.add_argument("--svr-eps", type=float, default=0.1)
    cmp_parser.add_argument("--svr-gamma", type=_gamma_arg, default="scale")
    cmp_parser.add_argument(
        "--svr-embed", action="store_true",
        help="Enable the SVR-on-embeddings baseline (Zhu et al.'s actual "
             "calibration recipe): per subject, fit two RBF-SVRs from the K "
             "support fused features to (x, y), predict on the query features. "
             "Replaces the readout entirely with a per-subject SVR.",
    )
    cmp_parser.add_argument("--svr-embed-C", type=float, default=1.0)
    cmp_parser.add_argument("--svr-embed-eps", type=float, default=0.1)
    cmp_parser.add_argument("--svr-embed-gamma", type=_gamma_arg, default="scale")
    cmp_parser.add_argument(
        "--fc-ft", action="store_true",
        help="Enable the head-only fine-tune baseline (Zhu et al. style): per "
             "subject, clone the base model's readout, Adam-train it on the K "
             "support frames for --fc-ft-steps, predict on the query frames. "
             "Adds a 'fc_ft' method to the comparison.",
    )
    cmp_parser.add_argument("--fc-ft-steps", type=int, default=20,
                            help="Full-batch Adam steps for fc_ft (default 20, matches Zhu et al.).")
    cmp_parser.add_argument("--fc-ft-lr", type=float, default=5e-5,
                            help="Adam lr for fc_ft (default 5e-5, matches Zhu et al.).")
    cmp_parser.add_argument("--fc-ft-weight-decay", type=float, default=5e-4,
                            help="Adam weight_decay for fc_ft (default 5e-4, matches Zhu et al.).")
    cmp_parser.add_argument("--batch-size", type=int, default=64)
    cmp_parser.add_argument("--num-workers", type=int, default=4)
    cmp_parser.add_argument("--folds", type=int, default=5)
    cmp_parser.add_argument("--fold-index", type=int, default=None)
    cmp_parser.add_argument("--seed", type=int, default=42)
    cmp_parser.add_argument("--log-file", default=None)
    cmp_parser.add_argument("--csv-out", default=None,
                            help="Append per-fold rows to this CSV for plotting.")

    svr_parser = subparsers.add_parser(
        "svrsearch",
        help="Swarm-style global hyperparameter search for the per-subject SVR "
             "baseline (inspired by Zhu et al., SwarmIntelligentCalibration). "
             "Tunes one (C, gamma, epsilon) triple per fold by minimizing mean "
             "Euclidean error across training-fold subjects; paste the result "
             "into metacompare via --svr-C/--svr-gamma/--svr-eps.",
    )
    add_common_args(svr_parser)
    svr_parser.add_argument("--input-mode", choices=("multistream",), default="multistream")
    svr_parser.add_argument(
        "--backbone", default=None,
        help="Accepted for symmetry with train / metatrain. Ignored: the "
             "backbone is read from --base-checkpoint, which is authoritative.",
    )
    svr_parser.add_argument("--base-checkpoint", required=True)
    svr_parser.add_argument(
        "--space", choices=("prediction", "embedding"), default="prediction",
        help="Which SVR baseline to tune. prediction (default) tunes our "
             "correction-style SVR ((pred_xy)->(true_xy)); embedding tunes the "
             "Zhu et al.-style SVR ((fused_feature)->coord), which replaces the "
             "readout. Use embedding for --svr-embed in metacompare.",
    )
    svr_parser.add_argument("--k", type=int, default=16,
                            help="Calibration frames per subject during HP search.")
    svr_parser.add_argument("--trials", type=int, default=3,
                            help="Random support/query draws per subject per fitness eval.")
    svr_parser.add_argument("--pop", type=int, default=30, help="PSO population size.")
    svr_parser.add_argument("--iters", type=int, default=50, help="PSO outer iterations.")
    svr_parser.add_argument("--batch-size", type=int, default=64)
    svr_parser.add_argument("--num-workers", type=int, default=4)
    svr_parser.add_argument("--folds", type=int, default=5)
    svr_parser.add_argument("--fold-index", type=int, default=None)
    svr_parser.add_argument("--seed", type=int, default=42)
    svr_parser.add_argument("--log-file", default=None)
    svr_parser.add_argument("--json-out", default=None,
                            help="Write the tuned per-fold hyperparameters to this JSON file.")

    explain_parser = subparsers.add_parser("explain")
    add_common_args(explain_parser)
    explain_parser.add_argument("--checkpoint", required=True)
    explain_parser.add_argument("--out-dir", default="./vit_gaze_segments")
    explain_parser.add_argument("--index", type=int, default=None)
    explain_parser.add_argument("--rec", type=int, default=None)
    explain_parser.add_argument("--frame", type=int, default=None)
    explain_parser.add_argument("--num-examples", type=int, default=5)
    explain_parser.add_argument(
        "--explain-source",
        choices=("raw", "synthetic", "both"),
        default="raw",
        help="For single-image checkpoints, run attribution on this image source.",
    )
    explain_parser.add_argument(
        "--attribution",
        choices=("smoothgrad", "occlusion", "both"),
        default="occlusion",
        help="Occlusion is slower but usually more trustworthy for segmentation.",
    )
    explain_parser.add_argument("--smoothgrad-samples", type=int, default=12)
    explain_parser.add_argument("--noise-std", type=float, default=0.03)
    explain_parser.add_argument("--occlusion-patch", type=int, default=24)
    explain_parser.add_argument("--occlusion-stride", type=int, default=12)
    explain_parser.add_argument(
        "--occlusion-batch",
        type=int,
        default=16,
        help="Number of occluded variants evaluated per forward pass. Raise it on "
             "a high-VRAM GPU (e.g. 64 on a 5090); lower it on an 8 GB card. Results "
             "are identical regardless of this value.",
    )
    explain_parser.add_argument("--threshold-percentile", type=float, default=85.0)

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "metatrain":
        meta_train(args)
    elif args.command == "metacompare":
        metacompare(args)
    elif args.command == "svrsearch":
        svrsearch(args)
    elif args.command == "explain":
        explain(args)
    else:
        raise ValueError(args.command)
