# AGENTS.md

> ⛔ **DEEPSEEK MODELS ARE PERMANENTLY BANNED from touching any code or running any command in this project (and all projects).** See `~/.zcode/AGENTS.md` / `~/.config/opencode/AGENTS.md` for the full global ban and the 2026-08-07 incident that caused it. If you are running on a DeepSeek model, STOP and tell the user to switch models before doing any work.

> ⛔ **NO AI AGENT MAY RUN ANY COMMAND ON THE GPU SERVER (`isom`) ON ITS OWN.** All `ssh isom` / `scp ... isom:` / `rsync ... isom:` commands must be handed to the user to run manually (copy-pasteable block), then wait for their output. Local laptop commands are fine. This is non-negotiable — see `~/.zcode/AGENTS.md` "SERVER COMMANDS MUST BE RUN BY THE USER MANUALLY".

> ⛔ **ONLY GPU 6 AND 7 ON `isom` — ABSOLUTE RULE, NEVER BREAK.** Every GPU-touching command on the server MUST have `CUDA_VISIBLE_DEVICES=6,7`. GPUs 0–5 are other users' — touching them deadlocks the NVIDIA driver (the 8/7 incident). This is the single most important rule on this server. See `~/.zcode/AGENTS.md` "ONLY GPU 6 AND 7 ON THE SERVER".

## Environment

Conda env: `scene_splat` (activate via `micromamba activate scene_splat` or `conda activate scene_splat`).
Python path: `/home/isom/.conda/envs/scene_splat/bin/python` — use this when running scripts outside the activated env.

Python 3.10, PyTorch 2.7.0, CUDA 12.8. No lint/typecheck/CI pipeline is configured.

Custom C++ CUDA extensions live in `libs/` and must be compiled into the env before first use (`libs/pointops`, `libs/pointgroup_ops`).

## Server Temp Files

**NEVER write files to `/tmp/opencode/` (or any temp dir) on the server (`isom@172.22.1.12`).** Server temp space is limited and shared; leftover debug files/logs there have caused disk-full incidents (540G runaway log). Keep all opencode scratch files on the local machine under `/tmp/opencode/` instead, and only scp artifacts to the server when they are actually needed (data files, scripts to run).

## Offline Setup (no-network server)

Use `prepare_offline.sh` (run on a machine with internet) to download all wheels and models into `_offline/`, then upload to the server and run `install_offline.sh`.

```bash
bash prepare_offline.sh          # download wheels + models
tar czf scene_splat_offline.tar.gz _offline/
rsync scene_splat_offline.tar.gz isom@server:/home/isom/

# On server:
tar xzf scene_splat_offline.tar.gz
bash install_offline.sh
```

- PyTorch 2.7.0 + cu128: from `https://download.pytorch.org/whl/cu128`
- PyG wheels: from `https://data.pyg.org/whl/torch-2.7.0+cu128.html`
- spconv-cu128 + cumm-cu128: from community index `https://ratharog.github.io/cumm-spconv/`
- SAM2 checkpoint downloaded automatically (unless `--skip-models`)

## Git Submodules

Two submodules (initialized via `.gitmodules`):
- `LitePT/` — Lightweight Point Transformer (prs-eth/LitePT). Integrated via `pointcept/models/litept/model.py` which adds `LitePT/` and `LitePT/libs/` to `sys.path`.
- `sam2_repo/` — Facebook SAM 2. Requires checkpoint at `sam2_repo/checkpoints/sam2.1_hiera_large.pt` for preprocessing.

## Key Entry Points

All live under `tools/`:

| Script | Purpose |
|---|---|
| `tools/train.py` | Main training/testing engine (LangPretrainer, PT-v3m1) |
| `tools/train_lite.py` | LitePT backbone training (adds `--density-invariant` flag) |
| `tools/train_lora.py` | LoRA fine-tuning (incomplete — has TODO) |
| `tools/ssl_pretrain.py` | Self-supervised pretraining (SimDINO-based) |
| `tools/test.py` | Standalone test via TESTERS registry |
| `tools/inference/lang_inference.py` | Standalone language feature inference |
| `tools/create_11ch_checkpoint.py` | Convert 5ch→11ch checkpoints for OVS data |

Shared CLI flags: `--config-file`, `--options KEY=VALUE`, `--num-gpus`, `--multi_node`.

## GPU Constraints

**IMPORTANT: Training must be strictly restricted to GPU 6 and 7 only.**
- Always launch training with `CUDA_VISIBLE_DEVICES=6,7` and `--num-gpus 2`.
- Never use GPUs 0–5 (shared with other workloads / reserved).
- The GPU 6/7 restriction applies to all training runs, including smoke tests.

```bash
# Single-GPU training
python tools/train.py --config-file configs/scannet/lang-pretrain-scannet-mcmc-wo-normal-contrastive.py \
  --options save_path=exp_runs/my_exp --num-gpus 1

# Multi-GPU
python tools/train.py --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
  --options save_path=exp_runs/my_exp batch_size=8 batch_size_val=4 num_worker=32 gpu_nums=4 --num-gpus 4

# Test only
python tools/train.py --config-file configs/concat_dataset/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.py \
  --options save_path=exp_runs/my_exp weight=model_best.pth test_only=True --num-gpus 4

# SSL pretraining
python tools/ssl_pretrain.py --config-file configs/concat_dataset/ssl-pretrain-concat-scan-ppv2-matt-3rscan-arkit-hyper-mcmc-base.py \
  --options save_path=exp_runs/ssl_pretrain/my_exp
```

## Config System

Configs use mmcv-style inheritance via `_base_` lists. Base configs are in `configs/_base_/`.

Config naming: `{task}-{backbone}-{dataset}-{variant}.py` where task = `lang-pretrain | ssl-pretrain | semseg | lora-finetune`.

**Critical**: When using SVD-compressed features (`load_compressed_lang_feat=True`), `FilterValidPoints` **must** come before `FilterCoordOutliers` in the transform pipeline. Wrong order causes coord/lang_feat shape mismatch. See `configs/custom/lang-pretrain-litept-ovs-gridsvd.py` for the correct pattern.

## Data Format

Each scene is a folder of `.npy` files: `coord.npy` (N,3), `color.npy` (N,3), `opacity.npy` (N,1), `quat.npy` (N,4), `scale.npy` (N,3), plus optional `lang_feat.npy` (N,D) and `valid_feat_mask.npy` (N). Semantic labels: `segment.npy` (N).

3DGS `.ply` → `.npy` conversion: `scripts/preprocess_gs.py`.

## Framework Architecture

Built on [Pointcept](https://github.com/Pointcept/Pointcept). Core code in `pointcept/`:
- `models/` — Registry-based model definitions (`@MODELS.register_module()`)
- `engines/train.py` — Training loop with distributed support
- `engines/test.py` — Testing loop with chunking for memory efficiency
- `datasets/` — Dataset loaders (GenericGSDataset is the base for 3DGS data)
- `datasets/transform.py` — Transform pipeline (FilterValidPoints, FilterCoordOutliers, etc.)

## Gotchas

- VL pretraining requires ≥48GB GPU memory.
- Batch sizes scale with GPU count; `batch_size = 2 * gpu_nums` is the typical formula.
- Evaluation during training uses grid sampling (fast); `test_only=True` uses full-scene chunking (thorough).
- `weight=path.pth` without `resume=True` resets epoch/scheduler (fine-tuning from checkpoint, not resuming).
- Structural classes (`wall`, `floor`, `ceiling`) are excluded from foreground mIoU by default.
- Hardcoded dataset paths appear in `preprocess.sh` and some configs — always check before running.
