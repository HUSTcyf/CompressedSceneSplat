import sys
import glob
import os
import shutil
import time
import json
import datetime
import torch
import torch.utils.data
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.cache import shared_dict
from pointcept.utils.misc import load_checkpoint
import pointcept.utils.comm as comm
from pointcept.engines.test import TESTERS

from .default import HookBase
from .builder import HOOKS
from pathlib import Path

from numpy.core.multiarray import scalar

# allow the numpy scalar in weights_only loads
torch.serialization.add_safe_globals([scalar])


@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        self._remain_iter = self.trainer.max_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)

    def before_step(self):
        self.curr_iter += 1
        # MSC pretrain do not have offset information. Comment the code for support MSC
        # info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] " \
        #        "Scan {batch_size} ({points_num}) ".format(
        #     epoch=self.trainer.epoch + 1, max_epoch=self.trainer.max_epoch,
        #     iter=self.trainer.comm_info["iter"], max_iter=len(self.trainer.train_loader),
        #     batch_size=len(self.trainer.comm_info["input_dict"]["offset"]),
        #     points_num=self.trainer.comm_info["input_dict"]["offset"][-1]
        # )
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            # Filter to only include scalar tensor keys
            self.model_output_keys = [
                key for key, value in model_output_dict.items()
                if isinstance(value, torch.Tensor) and value.numel() == 1
            ]
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None → only 'model_last'

    def after_epoch(self):
        # ---------------  COMPUTE (all ranks) ---------------
        # gather best‑metric bookkeeping exactly as before
        is_best = False
        best_type = "validation"
        best_value = None

        if is_main_process() and self.trainer.cfg.evaluate:
            cur_val = self.trainer.comm_info["current_metric_value"]
            cur_name = self.trainer.comm_info["current_metric_name"]
            if cur_val > self.trainer.best_metric_value:
                self.trainer.best_metric_value = cur_val
                is_best = True
                best_value = cur_val
                self.trainer.logger.info(
                    f"Best validation {cur_name} updated to: {cur_val:.4f}"
                )
            self.trainer.logger.info(
                f"Currently Best {cur_name}: {self.trainer.best_metric_value:.4f}"
            )

        # make **sure every pending kernel is done** before anyone touches I/O
        torch.cuda.synchronize()
        synchronize()

        # ---------------  I/O (rank‑0 only) ---------------
        if is_main_process():
            ckpt_dir = Path(self.trainer.cfg.save_path) / "model"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            fname_last = ckpt_dir / "model_last.pth"
            best_epoch_json_path = Path(self.trainer.cfg.save_path) / "best_epoch.json"

            # Prepare checkpoint data
            current_epoch = self.trainer.epoch + 1
            checkpoint_data = {
                "epoch": current_epoch,
                "state_dict": self.trainer.model.state_dict(),
                "optimizer": self.trainer.optimizer.state_dict(),
                "scheduler": self.trainer.scheduler.state_dict(),
                "scaler": (
                    self.trainer.scaler.state_dict()
                    if self.trainer.cfg.enable_amp
                    else None
                ),
                "best_metric_value": self.trainer.best_metric_value,
            }

            # Save checkpoint
            self.trainer.logger.info(f"Saving checkpoint to: {fname_last}")
            torch.save(checkpoint_data, f"{fname_last}.tmp")
            os.replace(f"{fname_last}.tmp", fname_last)

            # Save best model and update best_epoch.json
            if is_best:
                shutil.copyfile(fname_last, ckpt_dir / "model_best.pth")

                # Save best epoch information to JSON
                best_epoch_info = {
                    "best_epoch": current_epoch,
                    "best_type": best_type,
                    "best_value": float(best_value) if best_value is not None else float(self.trainer.best_metric_value),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "checkpoint_path": str(ckpt_dir / "model_best.pth"),
                }

                # Add validation metric details
                if self.trainer.cfg.evaluate and "current_metric_name" in self.trainer.comm_info:
                    best_epoch_info["validation_metric_name"] = self.trainer.comm_info["current_metric_name"]
                    best_epoch_info["validation_metric_value"] = float(self.trainer.comm_info["current_metric_value"])

                # Save to best_epoch.json
                with open(best_epoch_json_path, 'w') as f:
                    json.dump(best_epoch_info, f, indent=2)

                self.trainer.logger.info(
                    f"Best epoch info saved to: {best_epoch_json_path}"
                )
                self.trainer.logger.info(
                    f"  - Best epoch: {best_epoch_info['best_epoch']}"
                )
                self.trainer.logger.info(
                    f"  - Best type: {best_epoch_info['best_type']}"
                )
                self.trainer.logger.info(
                    f"  - Best value: {best_epoch_info['best_value']:.6f}"
                )

            # Save periodic checkpoints
            if self.save_freq and current_epoch % self.save_freq == 0:
                shutil.copyfile(
                    fname_last, ckpt_dir / f"epoch_{current_epoch}.pth"
                )

        # ---------------  RESUME TRAINING (all ranks) ---------------
        synchronize()  # wait until file rename finishes


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_eval(self):
        self.before_train()

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )

            # Get model state dict for shape validation
            model_state = self.trainer.model.state_dict()
            weight = OrderedDict()

            self.trainer.logger.info(
                f"Processing checkpoint keys with keyword: {self.keywords} -> {self.replacement}"
            )
            skipped_keys = {"shape_mismatch": [], "not_in_model": []}

            for key, value in checkpoint["state_dict"].items():
                # Process key for compatibility
                processed_key = key
                if not key.startswith("module."):
                    processed_key = "module." + key  # Add prefix for DDP format
                if self.keywords in processed_key:
                    processed_key = processed_key.replace(
                        self.keywords, self.replacement
                    )
                if comm.get_world_size() == 1:  # Remove prefix if not DDP
                    processed_key = (
                        processed_key[7:]
                        if processed_key.startswith("module.")
                        else processed_key
                    )

                # Key validation logic
                if processed_key in model_state:
                    if model_state[processed_key].shape == value.shape:
                        weight[processed_key] = value
                    else:
                        skipped_keys["shape_mismatch"].append(
                            f"{key} (ckpt shape: {value.shape} vs model shape: {model_state[processed_key].shape})"
                        )
                else:
                    skipped_keys["not_in_model"].append(processed_key)

            # Load filtered weights
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )

            # Detailed logging
            self.trainer.logger.info(
                f"Successfully loaded {len(weight)}/{len(checkpoint['state_dict'])} keys"
            )
            if skipped_keys["shape_mismatch"]:
                self.trainer.logger.warning(
                    f"Skipped {len(skipped_keys['shape_mismatch'])} keys due to shape mismatch:\n"
                )
            if skipped_keys["not_in_model"]:
                self.trainer.logger.warning(
                    f"Skipped {len(skipped_keys['not_in_model'])} keys not in model:\n"
                    f"{skipped_keys['not_in_model']}"
                )
            if self.strict:
                self.trainer.logger.info(
                    f"Strict mode: Missing keys: {load_state_info[0]}"
                )
                self.trainer.logger.info(
                    f"Strict mode: Unexpected keys: {load_state_info[1]}"
                )

            # Resume training if needed
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        # if cfg.test is a dict => single tester
        # if cfg.test is a list => multiple testers
        if isinstance(cfg.test, dict):
            tester = TESTERS.build(
                dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
            )  # e.g. test = dict(type='SemSegTester', verbose=True)
            if self.test_last:
                self.trainer.logger.info(
                    "=> Testing on model_last (current weight) ..."
                )
            else:
                best_path = os.path.join(
                    self.trainer.cfg.save_path, "model", "model_best.pth"
                )
                self.trainer.logger.info("=> Testing on model_best...")
                checkpoint = torch.load(best_path)
                state_dict = checkpoint["state_dict"]
                tester.model.load_state_dict(state_dict, strict=True)
                self.trainer.logger.info(f"Loaded ckpt from {best_path}")
            tester.test()
        elif isinstance(cfg.test, list):
            for i, test_cfg in enumerate(cfg.test):
                tester = TESTERS.build(
                    dict(type=test_cfg.type, cfg=cfg, model=self.trainer.model, index=i)
                )
                if self.test_last:
                    self.trainer.logger.info(
                        "=> Testing on model_last (current weight) ..."
                    )
                else:
                    best_path = os.path.join(
                        self.trainer.cfg.save_path, "model", "model_best.pth"
                    )
                    self.trainer.logger.info("=> Testing on model_best...")
                    load_checkpoint(tester.model, best_path)
                    self.trainer.logger.info(f"Loaded ckpt from {best_path}")
                tester.test()
                del tester
                torch.cuda.empty_cache()


@HOOKS.register_module()
class BeginningEvaluator(HookBase):  # for testing
    def __init__(self, test_last=False):
        self.test_last = test_last

    def before_epoch(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>> Beginning Evaluation Before Training >>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        if isinstance(cfg.test, dict):
            tester = TESTERS.build(
                dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
            )  # e.g. test = dict(type='SemSegTester', verbose=True)
            if self.test_last:
                self.trainer.logger.info("=> Testing on model_last (current weight)...")
            else:
                self.trainer.logger.info("=> Testing on model_best ...")
                best_path = os.path.join(
                    self.trainer.cfg.save_path, "model", "model_best.pth"
                )
                checkpoint = torch.load(best_path)
                self.trainer.logger.info(f"Loading ckpt from {best_path}")
                state_dict = checkpoint["state_dict"]
                tester.model.load_state_dict(state_dict, strict=True)
            tester.test()
        elif isinstance(cfg.test, list):
            for i, test_cfg in enumerate(cfg.test):
                tester = TESTERS.build(
                    dict(type=test_cfg.type, cfg=cfg, model=self.trainer.model, index=i)
                )
                if self.test_last:
                    self.trainer.logger.info(
                        "=> Testing on model_last (current weight)..."
                    )
                else:
                    self.trainer.logger.info("=> Testing on model_best ...")
                    best_path = os.path.join(
                        self.trainer.cfg.save_path, "model", "model_best.pth"
                    )
                    load_checkpoint(tester.model, best_path)
                    self.trainer.logger.info(f"Loading ckpt from {best_path}")
                tester.test()
                del tester
                torch.cuda.empty_cache()
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Beginning Evaluator, Skip Training >>>>>>>>>>>>>>>>"
        )
        sys.exit(0)


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()


@HOOKS.register_module()
class PerSceneLossVisualizer(HookBase):
    """
    Hook to visualize and save per-scene loss curves during training.

    Generates individual plots for each scene showing:
    - Total loss over training iterations
    - L1 loss over training iterations
    - Cosine loss over training iterations

    Saves plots to {save_path}/loss_curves/ directory.
    Updates plots after each epoch (configurable frequency).
    Accumulates loss history across epochs (no duplicate iterations).

    Note: Per-scene plots are only saved for OVS datasets (lerf_ovs, 3DOVS).
    For other datasets (e.g., ScanNet), only the average loss plot is saved.
    """

    def __init__(self, enabled=True, save_every_n_epochs=1, save_at_end=True):
        """
        Args:
            enabled: If False, skip visualization (useful for disabling without removing from config)
            save_every_n_epochs: Save loss curves every N epochs (1 = every epoch, None = only at end)
            save_at_end: If True, also save final curves at the end of training
        """
        self.enabled = enabled
        self.save_every_n_epochs = save_every_n_epochs
        self.save_at_end = save_at_end
        # Accumulated loss history (persists across epochs)
        self._accumulated_losses = {}
        self._scene_epoch_iter_counts = {}
        # Cache for OVS dataset detection (renamed to avoid recursion)
        self._is_ovs_dataset_cache = None

    def _is_ovs_dataset(self):
        """Check if the current dataset is an OVS dataset (lerf_ovs, 3DOVS, etc.)."""
        if self._is_ovs_dataset_cache is not None:
            return self._is_ovs_dataset_cache

        cfg = self.trainer.cfg

        # Check if train dataset is ConcatDataset with OVS data sources
        train_cfg = cfg.data.train if hasattr(cfg.data, 'train') else cfg.data.get('train', {})

        # Method 1: Check if it's a ConcatDataset with specific OVS paths
        if isinstance(train_cfg, dict) and train_cfg.get('type') == 'ConcatDataset':
            datasets = train_cfg.get('datasets', [])
            for dataset in datasets:
                data_root = dataset.get('data_root', '')
                if any(keyword in data_root.lower() for keyword in ['lerf_ovs', '3dovs', 'ovs']):
                    self._is_ovs_dataset_cache = True
                    return True

        # Method 2: Check data_root directly for single dataset
        if isinstance(train_cfg, dict):
            data_root = train_cfg.get('data_root', '')
            if any(keyword in data_root.lower() for keyword in ['lerf_ovs', '3dovs', 'ovs']):
                self._is_ovs_dataset_cache = True
                return True

        # Default: not an OVS dataset (e.g., ScanNet, ScanNet200)
        self._is_ovs_dataset_cache = False
        return False

    def _load_accumulated_losses(self):
        """Load accumulated loss history from previous runs."""
        import json
        from pathlib import Path

        save_path = Path(self.trainer.cfg.save_path)
        history_file = save_path / "loss_curves" / "_accumulated_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self._accumulated_losses = data.get('losses', {})
                    # Convert epoch keys from string to int (JSON keys are always strings)
                    epoch_iter_counts = data.get('epoch_iter_counts', {})
                    self._scene_epoch_iter_counts = {}
                    for scene_name, epoch_dict in epoch_iter_counts.items():
                        self._scene_epoch_iter_counts[scene_name] = {
                            int(epoch): count for epoch, count in epoch_dict.items()
                        }
                self.trainer.logger.info(f"[PerSceneLossVisualizer] Loaded accumulated history from {len(self._accumulated_losses)} scenes")
            except Exception as e:
                self.trainer.logger.warning(f"[PerSceneLossVisualizer] Failed to load accumulated history: {e}")
                self._accumulated_losses = {}
                self._scene_epoch_iter_counts = {}

    def _save_accumulated_losses(self):
        """Save accumulated loss history for future epochs."""
        import json
        from pathlib import Path

        save_path = Path(self.trainer.cfg.save_path)
        loss_curves_dir = save_path / "loss_curves"
        loss_curves_dir.mkdir(parents=True, exist_ok=True)
        history_file = loss_curves_dir / "_accumulated_history.json"

        try:
            data = {
                'losses': self._accumulated_losses,
                'epoch_iter_counts': self._scene_epoch_iter_counts,
            }
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.trainer.logger.warning(f"[PerSceneLossVisualizer] Failed to save accumulated history: {e}")

    def _merge_new_losses(self, scene_name, scene_idx, num_scenes, new_loss_data):
        """Merge new loss data with accumulated history (avoiding duplicate iterations).

        SIMPLIFIED VERSION: iterations are already global (computed in trainer),
        so no complex conversion is needed. Just merge with deduplication.

        Args:
            scene_name: Name of the scene
            scene_idx: Index of this scene in the scene order (0 to num_scenes-1) [unused, kept for compat]
            num_scenes: Total number of scenes [unused, kept for compat]
            new_loss_data: New loss data to merge (iterations are already global, epochs are already set)
        """
        if scene_name not in self._accumulated_losses:
            # First time seeing this scene - just copy the data
            self._accumulated_losses[scene_name] = {
                'total_loss': list(new_loss_data['total_loss']),
                'l1_loss': list(new_loss_data.get('l1_loss', [])),
                'cos_loss': list(new_loss_data.get('cos_loss', [])),
                'contrast_loss': list(new_loss_data.get('contrast_loss', [])),
                'iterations': list(new_loss_data['iterations']),  # Already global!
                'epochs': list(new_loss_data['epochs']),  # Already set by trainer!
            }
        else:
            # Merge: only add iterations that don't already exist (simple deduplication)
            accumulated = self._accumulated_losses[scene_name]
            existing_iters = set(accumulated['iterations'])

            for i, global_iter in enumerate(new_loss_data['iterations']):
                if global_iter not in existing_iters:
                    accumulated['iterations'].append(global_iter)
                    accumulated['epochs'].append(new_loss_data['epochs'][i])
                    accumulated['total_loss'].append(new_loss_data['total_loss'][i])
                    if new_loss_data.get('l1_loss'):
                        accumulated['l1_loss'].append(new_loss_data['l1_loss'][i])
                    if new_loss_data.get('cos_loss'):
                        accumulated['cos_loss'].append(new_loss_data['cos_loss'][i])
                    if new_loss_data.get('contrast_loss'):
                        accumulated['contrast_loss'].append(new_loss_data['contrast_loss'][i])
                    existing_iters.add(global_iter)

    def after_epoch(self):
        """Generate and save loss curves after each epoch."""
        if not self.enabled:
            return

        # Only run on main process to avoid duplicate work
        if not is_main_process():
            return

        # Load accumulated history from previous saves
        self._load_accumulated_losses()

        # Check if we should save this epoch
        if self.save_every_n_epochs is None:
            return
        if (self.trainer.epoch + 1) % self.save_every_n_epochs != 0:
            return

        self._generate_plots(f"epoch_{self.trainer.epoch + 1}")

    def after_train(self):
        """Generate and save final loss curves after training completes."""
        if not self.enabled:
            return

        # Only run on main process to avoid duplicate work
        if not is_main_process():
            return

        # Load accumulated history
        self._load_accumulated_losses()

        if self.save_at_end:
            self._generate_plots("final")

    def _verify_loss_data_consistency(self):
        """Verify that loss data is consistent: Total should equal L1 + Cos + Contrast (approximately)."""
        import numpy as np

        for scene_name, loss_data in self._accumulated_losses.items():
            total = np.array(loss_data['total_loss'])
            l1 = np.array(loss_data.get('l1_loss', []))
            cos = np.array(loss_data.get('cos_loss', []))
            contrast = np.array(loss_data.get('contrast_loss', []))

            if len(l1) > 0 and len(cos) > 0 and len(contrast) > 0:
                # Check consistency for first 10 points
                inconsistent_count = 0
                for i in range(min(10, len(total))):
                    expected = l1[i] + cos[i] + contrast[i]
                    actual = total[i]
                    if abs(expected - actual) > 0.01 and actual > 0:  # Allow small numerical errors
                        inconsistent_count += 1

                if inconsistent_count > 0:
                    self.trainer.logger.warning(
                        f"[PerSceneLossVisualizer] Scene '{scene_name}': "
                        f"{inconsistent_count}/10 data points have Total != L1 + Cos + Contrast"
                    )
                    # Print first 5 points for debugging - all on same line as requested
                    for i in range(min(5, len(total))):
                        if i < len(l1) and i < len(cos) and i < len(contrast):
                            expected = l1[i] + cos[i] + contrast[i]
                            actual = total[i]
                            self.trainer.logger.info(
                                f"  Point {i}: Total={actual:.6f}, L1={l1[i]:.6f}, "
                                f"Cos={cos[i]:.6f}, Contrast={contrast[i]:.6f}, "
                                f"L1+Cos+Contrast={expected:.6f}, Diff={expected-actual:.6f}"
                            )

    def _generate_plots(self, suffix):
        """Generate and save loss curves with given suffix."""
        # Check if trainer has per_scene_losses
        if not hasattr(self.trainer, 'per_scene_losses') or not self.trainer.per_scene_losses:
            self.trainer.logger.info("[PerSceneLossVisualizer] No per-scene losses tracked, skipping visualization.")
            return

        # DEBUG: Verify loss data consistency before plotting
        self._verify_loss_data_consistency()

        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path
        except ImportError:
            self.trainer.logger.warning("[PerSceneLossVisualizer] matplotlib not available, skipping visualization.")
            return

        # Create output directory
        save_path = Path(self.trainer.cfg.save_path)
        loss_curves_dir = save_path / "loss_curves"
        loss_curves_dir.mkdir(parents=True, exist_ok=True)

        # Get scene order (preserve order from per_scene_losses dict)
        scene_order = list(self.trainer.per_scene_losses.keys())
        num_scenes = len(scene_order)
        self.trainer.logger.info(f"[PerSceneLossVisualizer] Scene order: {scene_order}")

        # Merge new losses with accumulated history
        # Note: iterations are already global (computed in trainer as epoch * iters_per_epoch + local_iter)
        # No complex conversion needed - just merge with deduplication
        for scene_idx, scene_name in enumerate(scene_order):
            self._merge_new_losses(scene_name, scene_idx, num_scenes, self.trainer.per_scene_losses[scene_name])

        # Save accumulated history for next time
        self._save_accumulated_losses()

        # CRITICAL FIX: Don't clear per_scene_losses entirely!
        # We need to keep the LAST loss value for each scene to continue the curve in next epoch
        # Clear all but the last entry for each scene
        for scene_name in list(self.trainer.per_scene_losses.keys()):
            loss_data = self.trainer.per_scene_losses[scene_name]
            # Keep only the last entry for continuing the curve
            if len(loss_data['total_loss']) > 1:
                self.trainer.per_scene_losses[scene_name] = {
                    'total_loss': [loss_data['total_loss'][-1]],
                    'l1_loss': [loss_data['l1_loss'][-1]],
                    'cos_loss': [loss_data['cos_loss'][-1]],
                    'contrast_loss': [loss_data.get('contrast_loss', [0.0])[-1]] if loss_data.get('contrast_loss') else [0.0],
                    'iterations': [loss_data['iterations'][-1]],
                    'epochs': [loss_data['epochs'][-1]],
                }

        # Clear scenario iteration counter (safe to reset)
        if hasattr(self.trainer, '_scenario_iter_in_epoch'):
            self.trainer._scenario_iter_in_epoch.clear()

        # Use accumulated data for plotting (not the current epoch's partial data)
        plot_data = self._accumulated_losses if self._accumulated_losses else {}

        self.trainer.logger.info(f"[PerSceneLossVisualizer] Generating per-scene loss curves ({suffix})...")
        self.trainer.logger.info(f"  Total scenes: {len(plot_data)}")
        self.trainer.logger.info(f"  Output directory: {loss_curves_dir}")

        # Check if this is an OVS dataset
        is_ovs = self._is_ovs_dataset()
        self.trainer.logger.info(f"  OVS Dataset: {is_ovs} (per-scene plots: {'enabled' if is_ovs else 'disabled'})")

        # Generate plot for each scene (only for OVS datasets)
        if is_ovs:
            for scene_name, loss_data in plot_data.items():
                self._plot_scene_losses(scene_name, loss_data, loss_curves_dir, suffix)

        # Generate average loss plot across all scenes (always saved)
        self._plot_average_losses(plot_data, loss_curves_dir, suffix)

        # Update log message based on dataset type
        if is_ovs:
            self.trainer.logger.info(f"[PerSceneLossVisualizer] Saved {len(plot_data)} per-scene + 1 average loss curve plots ({suffix}).")
        else:
            self.trainer.logger.info(f"[PerSceneLossVisualizer] Saved 1 average loss curve plot ({suffix}). Per-scene plots disabled for non-OVS dataset.")

    def _plot_scene_losses(self, scene_name, loss_data, output_dir, suffix=""):
        """Generate and save a single scene's loss curves."""
        import matplotlib.pyplot as plt
        import numpy as np

        iterations = np.array(loss_data['iterations'])
        epochs = np.array(loss_data['epochs'])
        total_loss = np.array(loss_data['total_loss'])

        # DEBUG: Check data consistency before plotting
        if 'l1_loss' in loss_data and len(loss_data['l1_loss']) > 0:
            l1_loss = np.array(loss_data['l1_loss'])
            cos_loss = np.array(loss_data.get('cos_loss', []))

            # Check if lengths match
            if len(total_loss) != len(l1_loss) or len(total_loss) != len(cos_loss):
                self.trainer.logger.warning(
                    f"[PerSceneLossVisualizer] Scene '{scene_name}': "
                    f"Data length mismatch! total={len(total_loss)}, l1={len(l1_loss)}, cos={len(cos_loss)}"
                )

            # Check first 10 points for consistency
            mismatch_count = 0
            for i in range(min(10, len(total_loss), len(l1_loss), len(cos_loss))):
                expected = l1_loss[i] + cos_loss[i]
                actual = total_loss[i]
                if abs(expected - actual) > 0.01 and actual > 0:
                    mismatch_count += 1

            if mismatch_count > 5:
                self.trainer.logger.warning(
                    f"[PerSceneLossVisualizer] Scene '{scene_name}': "
                    f"{mismatch_count}/10 points have Total != L1 + Cos. "
                    f"This indicates a data recording bug!"
                )

        # Add initial point (0, 0) so all curves start from origin
        iterations = np.concatenate([[0], iterations])
        epochs = np.concatenate([[0], epochs])
        total_loss = np.concatenate([[0], total_loss])

        # Check if per-dimension data is available
        has_per_dim = ('per_dim_l1' in loss_data and
                       loss_data['per_dim_l1'] and
                       loss_data['per_dim_l1'][0] is not None)

        # Check if contrast_loss is available
        has_contrast = 'contrast_loss' in loss_data and loss_data['contrast_loss']

        # Create figure with subplots (5 rows if per-dim data available, otherwise 4)
        n_rows = 5 if has_per_dim else 4
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
        # Update title with epoch info
        title_suffix = f" (Epoch {epochs.max()})" if len(epochs) > 0 else ""
        fig.suptitle(f'Scene: {scene_name}{title_suffix}', fontsize=14, fontweight='bold')

        # Plot 1: Total Loss
        ax = axes[0]
        ax.plot(iterations, total_loss, 'b-', linewidth=1, alpha=0.7, label='Total Loss')
        ax.set_ylabel('Total Loss', fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Total Loss over Training', fontsize=12)

        # Plot 2: L1 Loss (if available)
        ax = axes[1]
        if loss_data['l1_loss']:
            l1_loss = np.concatenate([[0], np.array(loss_data['l1_loss'])])
            ax.plot(iterations, l1_loss, 'g-', linewidth=1, alpha=0.7, label='L1 Loss')
            ax.set_ylabel('L1 Loss', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'L1 Loss data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('L1 Loss', fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('L1 Loss over Training', fontsize=12)

        # Plot 3: Cosine Loss (if available)
        ax = axes[2]
        if loss_data['cos_loss']:
            cos_loss = np.concatenate([[0], np.array(loss_data['cos_loss'])])
            ax.plot(iterations, cos_loss, 'm-', linewidth=1, alpha=0.7, label='Cosine Loss')
            ax.set_ylabel('Cosine Loss', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Cosine Loss data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('Cosine Loss', fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Cosine Loss over Training', fontsize=12)

        # Plot 4: Contrastive Loss (if available)
        ax = axes[3]
        if has_contrast:
            contrast_loss = np.concatenate([[0], np.array(loss_data['contrast_loss'])])
            ax.plot(iterations, contrast_loss, 'c-', linewidth=1, alpha=0.7, label='Contrastive Loss')
            ax.set_ylabel('Contrastive Loss', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Contrastive Loss data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('Contrastive Loss', fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Contrastive Loss over Training', fontsize=12)

        # Plot 5: Per-Dimension L1 Loss (if available)
        if has_per_dim:
            ax = axes[4]
            per_dim_data = loss_data['per_dim_l1']

            # Filter out None values and stack tensors
            per_dim_tensors = [t.numpy() if hasattr(t, 'numpy') else np.array(t)
                              for t in per_dim_data if t is not None]

            if per_dim_tensors:
                # Stack: [num_iterations, num_dims]
                per_dim_matrix = np.stack(per_dim_tensors, axis=0)
                num_dims = per_dim_matrix.shape[1]

                # Pad with zeros at the beginning to match iterations length
                per_dim_padded = np.vstack([np.zeros((1, num_dims)), per_dim_matrix])

                # Plot each dimension as a separate line
                colors = plt.cm.viridis(np.linspace(0, 1, num_dims))
                for dim in range(num_dims):
                    ax.plot(iterations, per_dim_padded[:, dim],
                           color=colors[dim], alpha=0.7, linewidth=1, label=f'd[{dim}]')

                ax.set_ylabel('Per-Dim L1 Loss', fontsize=11)
                ax.set_xlabel('Iteration', fontsize=10)
                ax.legend(loc='upper right', ncol=4, fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_title('Per-Dimension L1 Loss (Unweighted) over Training', fontsize=12)

                # Add dimension weight info if available
                if 'per_dim_l1_weights' in loss_data and loss_data['per_dim_l1_weights']:
                    weights_data = loss_data['per_dim_l1_weights']
                    weights_tensor = weights_data[-1]  # Get latest weights
                    if weights_tensor is not None:
                        weights_np = weights_tensor.numpy() if hasattr(weights_tensor, 'numpy') else np.array(weights_tensor)
                        weight_info = f"Latest Weights: d[0]={weights_np[0]:.2f}, d[1]={weights_np[1]:.2f}, d[15]={weights_np[15]:.2f}"
                        ax.text(0.02, 0.02, weight_info, transform=ax.transAxes,
                               fontsize=8, verticalalignment='bottom',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Add statistics text box
        stats_text = f"Statistics:\n"
        stats_text += f"  Min Total Loss: {total_loss.min():.4f}\n"
        stats_text += f"  Max Total Loss: {total_loss.max():.4f}\n"
        stats_text += f"  Final Total Loss: {total_loss[-1]:.4f}\n"
        stats_text += f"  Num Iterations: {len(iterations)}\n"
        stats_text += f"  Epoch Range: {epochs.min():.0f} - {epochs.max():.0f}"

        fig.text(0.02, 0.5, stats_text, fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout(rect=[0.08, 0, 1, 0.96])

        # Save figure directly to loss_curves directory (overwrite previous version)
        # The suffix is only used for the "final" version
        if suffix == "final":
            output_path = output_dir / f"{scene_name}_loss_curves_final.png"
        else:
            output_path = output_dir / f"{scene_name}_loss_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Also save loss data as JSON for further analysis (overwrite previous version)
        import json
        if suffix == "final":
            json_path = output_dir / f"{scene_name}_loss_data_final.json"
        else:
            json_path = output_dir / f"{scene_name}_loss_data.json"
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            # CRITICAL FIX: Use original unmodified data, not the plotting arrays with prepended 0
            # The plotting arrays (iterations, epochs, total_loss) have been modified with
            # np.concatenate([[0], ...]) for visualization, but we need to save the raw data.
            json_data = {
                'iterations': loss_data['iterations'],  # Original data (no prepended 0)
                'epochs': loss_data['epochs'],          # Original data (no prepended 0)
                'total_loss': loss_data['total_loss'],  # Original data (no prepended 0)
            }
            if loss_data['l1_loss']:
                json_data['l1_loss'] = [float(x) for x in loss_data['l1_loss']]
            if loss_data['cos_loss']:
                json_data['cos_loss'] = [float(x) for x in loss_data['cos_loss']]
            if loss_data.get('contrast_loss'):
                json_data['contrast_loss'] = [float(x) for x in loss_data['contrast_loss']]

            # Save per-dimension loss data if available
            if 'per_dim_l1' in loss_data and loss_data['per_dim_l1']:
                per_dim_list = []
                for t in loss_data['per_dim_l1']:
                    if t is not None:
                        per_dim_list.append(t.tolist() if hasattr(t, 'tolist') else list(t))
                    else:
                        per_dim_list.append(None)
                json_data['per_dim_l1'] = per_dim_list

            # Save per-dimension weights if available
            if 'per_dim_l1_weights' in loss_data and loss_data['per_dim_l1_weights']:
                weights_list = []
                for t in loss_data['per_dim_l1_weights']:
                    if t is not None:
                        weights_list.append(t.tolist() if hasattr(t, 'tolist') else list(t))
                    else:
                        weights_list.append(None)
                json_data['per_dim_l1_weights'] = weights_list

            json.dump(json_data, f, indent=2)

    def _plot_average_losses(self, plot_data, output_dir, suffix=""):
        """Generate and save average loss curves across all scenes."""
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict

        if not plot_data:
            return

        # Collect all iterations and losses, using interpolation for alignment
        # Get the global iteration range (union of all iterations)
        all_iterations = set()
        for loss_data in plot_data.values():
            all_iterations.update(loss_data['iterations'])
        all_iterations = sorted(all_iterations)

        if not all_iterations:
            return

        # For each scene, create a mapping of iteration -> loss
        scene_losses = {}
        for scene_name, loss_data in plot_data.items():
            scene_losses[scene_name] = {
                'iterations': np.array(loss_data['iterations']),
                'total_loss': np.array(loss_data['total_loss']),
            }
            if loss_data.get('l1_loss'):
                scene_losses[scene_name]['l1_loss'] = np.array(loss_data['l1_loss'])
            if loss_data.get('cos_loss'):
                scene_losses[scene_name]['cos_loss'] = np.array(loss_data['cos_loss'])
            if loss_data.get('contrast_loss'):
                scene_losses[scene_name]['contrast_loss'] = np.array(loss_data['contrast_loss'])

        # Interpolate losses for all scenes at common iteration points
        # Use iteration 0 as the starting point (loss = 0)
        interp_iterations = np.array([0] + all_iterations)
        avg_total_loss = []
        avg_l1_loss = []
        avg_cos_loss = []
        avg_contrast_loss = []

        for iter_val in interp_iterations:
            total_losses_at_iter = []
            l1_losses_at_iter = []
            cos_losses_at_iter = []
            contrast_losses_at_iter = []

            for scene_name, scene_data in scene_losses.items():
                iter_idx = np.searchsorted(scene_data['iterations'], iter_val)
                if iter_idx == 0:
                    # Before the first data point, use 0
                    loss = 0
                elif iter_idx >= len(scene_data['iterations']):
                    # After the last data point, use the last value
                    loss = scene_data['total_loss'][-1]
                else:
                    # Interpolate between neighboring points
                    iter_prev = scene_data['iterations'][iter_idx - 1]
                    iter_next = scene_data['iterations'][iter_idx]
                    if iter_next == iter_prev:
                        loss = scene_data['total_loss'][iter_idx - 1]
                    else:
                        t = (iter_val - iter_prev) / (iter_next - iter_prev)
                        loss = (1 - t) * scene_data['total_loss'][iter_idx - 1] + t * scene_data['total_loss'][iter_idx]

                total_losses_at_iter.append(loss)

                # Same for L1 loss
                if 'l1_loss' in scene_data:
                    l1_loss = scene_data['l1_loss']
                    if iter_idx == 0:
                        loss = 0
                    elif iter_idx >= len(l1_loss):
                        loss = l1_loss[-1]
                    else:
                        iter_prev = scene_data['iterations'][iter_idx - 1]
                        iter_next = scene_data['iterations'][iter_idx]
                        if iter_next == iter_prev:
                            loss = l1_loss[iter_idx - 1]
                        else:
                            t = (iter_val - iter_prev) / (iter_next - iter_prev)
                            loss = (1 - t) * l1_loss[iter_idx - 1] + t * l1_loss[iter_idx]
                    l1_losses_at_iter.append(loss)

                # Same for Cosine loss
                if 'cos_loss' in scene_data:
                    cos_loss = scene_data['cos_loss']
                    if iter_idx == 0:
                        loss = 0
                    elif iter_idx >= len(cos_loss):
                        loss = cos_loss[-1]
                    else:
                        iter_prev = scene_data['iterations'][iter_idx - 1]
                        iter_next = scene_data['iterations'][iter_idx]
                        if iter_next == iter_prev:
                            loss = cos_loss[iter_idx - 1]
                        else:
                            t = (iter_val - iter_prev) / (iter_next - iter_prev)
                            loss = (1 - t) * cos_loss[iter_idx - 1] + t * cos_loss[iter_idx]
                    cos_losses_at_iter.append(loss)

                # Same for Contrastive loss
                if 'contrast_loss' in scene_data:
                    contrast_loss = scene_data['contrast_loss']
                    if iter_idx == 0:
                        loss = 0
                    elif iter_idx >= len(contrast_loss):
                        loss = contrast_loss[-1]
                    else:
                        iter_prev = scene_data['iterations'][iter_idx - 1]
                        iter_next = scene_data['iterations'][iter_idx]
                        if iter_next == iter_prev:
                            loss = contrast_loss[iter_idx - 1]
                        else:
                            t = (iter_val - iter_prev) / (iter_next - iter_prev)
                            loss = (1 - t) * contrast_loss[iter_idx - 1] + t * contrast_loss[iter_idx]
                    contrast_losses_at_iter.append(loss)

            avg_total_loss.append(np.mean(total_losses_at_iter))
            if l1_losses_at_iter:
                avg_l1_loss.append(np.mean(l1_losses_at_iter))
            if cos_losses_at_iter:
                avg_cos_loss.append(np.mean(cos_losses_at_iter))
            if contrast_losses_at_iter:
                avg_contrast_loss.append(np.mean(contrast_losses_at_iter))

        avg_total_loss = np.array(avg_total_loss)
        avg_l1_loss = np.array(avg_l1_loss) if avg_l1_loss else None
        avg_cos_loss = np.array(avg_cos_loss) if avg_cos_loss else None
        avg_contrast_loss = np.array(avg_contrast_loss) if avg_contrast_loss else None

        # Get max epoch info
        max_epoch = 0
        for loss_data in plot_data.values():
            epochs = loss_data.get('epochs', [])
            if epochs:
                max_epoch = max(max_epoch, max(epochs))

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()  # Flatten for easy indexing
        fig.suptitle(f'Average Loss Across All Scenes (N={len(plot_data)}) - Epoch {max_epoch}', fontsize=16, fontweight='bold')

        # Plot 1: Total Loss
        ax = axes[0]
        ax.plot(interp_iterations, avg_total_loss, 'b-', linewidth=2, alpha=0.8, label=f'Average Total Loss (N={len(plot_data)})')
        ax.set_ylabel('Total Loss', fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Average Total Loss over Training', fontsize=12, fontweight='bold')

        # Plot 2: L1 Loss (if available)
        ax = axes[1]
        if avg_l1_loss is not None:
            ax.plot(interp_iterations, avg_l1_loss, 'g-', linewidth=2, alpha=0.8, label=f'Average L1 Loss (N={len(plot_data)})')
            ax.set_ylabel('L1 Loss', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'L1 Loss data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('L1 Loss', fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Average L1 Loss over Training', fontsize=12, fontweight='bold')

        # Plot 3: Cosine Loss (if available)
        ax = axes[2]
        if avg_cos_loss is not None:
            ax.plot(interp_iterations, avg_cos_loss, 'm-', linewidth=2, alpha=0.8, label=f'Average Cosine Loss (N={len(plot_data)})')
            ax.set_ylabel('Cosine Loss', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Cosine Loss data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('Cosine Loss', fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Average Cosine Loss over Training', fontsize=12, fontweight='bold')

        # Plot 4: Contrastive Loss (if available)
        ax = axes[3]
        if avg_contrast_loss is not None:
            ax.plot(interp_iterations, avg_contrast_loss, 'c-', linewidth=2, alpha=0.8, label=f'Average Contrastive Loss (N={len(plot_data)})')
            ax.set_ylabel('Contrastive Loss', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Contrastive Loss data not available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel('Contrastive Loss', fontsize=11)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_title('Average Contrastive Loss over Training', fontsize=12, fontweight='bold')

        # Add statistics text box
        stats_text = f"Average Statistics:\n"
        stats_text += f"  Scenes: {len(plot_data)}\n"
        stats_text += f"  Min Total Loss: {avg_total_loss.min():.4f}\n"
        stats_text += f"  Max Total Loss: {avg_total_loss.max():.4f}\n"
        stats_text += f"  Final Total Loss: {avg_total_loss[-1]:.4f}\n"
        if avg_contrast_loss is not None:
            stats_text += f"  Final Contrastive Loss: {avg_contrast_loss[-1]:.4f}\n"
        stats_text += f"  Total Iterations: {len(interp_iterations)}"

        fig.text(0.02, 0.5, stats_text, fontsize=9, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout(rect=[0.08, 0, 1, 0.96])

        # Save figure
        if suffix == "final":
            output_path = output_dir / "average_loss_curves_final.png"
        else:
            output_path = output_dir / "average_loss_curves.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.trainer.logger.info(f"[PerSceneLossVisualizer] Saved average loss curves to {output_path}")