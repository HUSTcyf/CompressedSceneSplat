import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class LangPretrainer(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        verbose_losses=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria, verbose_losses=verbose_losses)

        # Learnable dimension-wise scaling to handle SVD feature magnitude mismatch
        # SVD-16 features have highly imbalanced magnitudes:
        # - dim[0] (DC): ~0.84 (largest, needs scale ~1.0)
        # - dim[1]: ~0.19 (needs scale ~0.8)
        # - dim[2]: ~0.11 (needs scale ~0.6)
        # - dim[3]: ~0.09 (needs scale ~0.5)
        # - dim[4]: ~0.08 (needs scale ~0.4)
        # - dim[5-15]: ~0.02-0.06 (needs scale ~0.3)
        dim_scale_init = torch.ones(16)
        dim_scale_init[0] = 1.0   # DC component
        dim_scale_init[1] = 0.8
        dim_scale_init[2] = 0.6
        dim_scale_init[3] = 0.5
        dim_scale_init[4] = 0.4
        dim_scale_init[5:] = 0.3  # Remaining dimensions
        self.dim_scale = nn.Parameter(dim_scale_init)

    def forward(self, input_dict, chunk_size=None):
        if (
            chunk_size is not None
            and chunk_size > 0
            and input_dict["coord"].shape[0] > chunk_size
        ):
            return self._chunked_forward(input_dict, chunk_size)
        point = Point(input_dict)
        point_feat = self.backbone(point)

        # CRITICAL FIX: Do NOT normalize features for L2 Loss
        # Normalization makes L2 Loss equivalent to 2*(1-Cosine), causing loss redundancy
        # and mode collapse. We keep features in their raw scale to preserve:
        # 1. Magnitude information (feature importance, confidence)
        # 2. Independent gradients from L2 and Cosine losses
        # 3. Better optimization landscape for 16-dim compressed features
        #
        # point_feat["feat"] = nn.functional.normalize(point_feat["feat"], p=2, dim=1)  # REMOVED

        # Apply learnable dimension-wise scaling to compensate for SVD feature magnitude mismatch
        # This allows the network to learn optimal scaling for each dimension
        feat = point_feat["feat"]
        feat_before = feat.clone()  # DEBUG: for verification

        # STABILITY CHECK: Ensure dim_scale is valid (not NaN/Inf)
        # This prevents cascading NaN failures when gradient updates go wrong
        if torch.isnan(self.dim_scale).any() or torch.isinf(self.dim_scale).any():
            print(f"\n[WARNING] dim_scale contains NaN/Inf! Resetting to safe values.")
            print(f"  dim_scale before reset: {self.dim_scale.detach().cpu().numpy()}")
            # Reset to safe initial values
            with torch.no_grad():
                self.dim_scale.copy_(torch.ones(16) * 0.3)
                self.dim_scale[0] = 1.0
            print(f"  dim_scale after reset: {self.dim_scale.detach().cpu().numpy()}")

        # Use ReLU to ensure non-negative scaling (negative values become 0)
        # Add a small epsilon (0.01) to prevent zero scaling
        # Then clamp to reasonable upper bound to prevent excessive amplification
        dim_scale_positive = torch.relu(self.dim_scale) + 0.01
        dim_scale_clamped = torch.clamp(dim_scale_positive, max=10.0)

        # NaN CHECK: After backbone output
        if torch.isnan(feat_before).any():
            print(f"\n🚨 NaN detected RIGHT AFTER BACKBONE!")
            print(f"  feat_before contains NaN!")
            print(f"  NaN count: {torch.isnan(feat_before).sum().item()}")
            print(f"  feat_before stats: min={feat_before.min().item():.6f}, max={feat_before.max().item():.6f}")

        feat = feat * dim_scale_clamped  # [N, 16] * [16] = [N, 16]

        # NaN CHECK: After scaling
        if torch.isnan(feat).any():
            print(f"\n🚨 NaN detected AFTER SCALING!")
            print(f"  dim_scale_clamped: {dim_scale_clamped.detach().cpu().numpy()}")
            print(f"  feat_before NaN: {torch.isnan(feat_before).any().item()}")
            print(f"  feat stats: min={feat.min().item():.6f}, max={feat.max().item():.6f}")
            # Stop propagation - this will cause an assertion error in the loss function
            # but we provide better diagnostics here
            assert False, "feat contains NaN after scaling! See diagnostic info above."

        # STABILITY CHECK: Detect and clip extreme values before returning
        # This prevents cascading failures from extreme activations
        feat_max = feat.abs().max().item()
        if feat_max > 100.0:
            print(f"\n⚠️ WARNING: Extreme feature values detected!")
            print(f"  feat_max: {feat_max:.6f} (threshold: 100.0)")
            print(f"  feat stats: min={feat.min().item():.6f}, max={feat.max().item():.6f}")
            print(f"  dim_scale_clamped: {dim_scale_clamped.detach().cpu().numpy()}")
            print(f"  Clipping to [-100, 100] to prevent instability...")
            feat = torch.clamp(feat, min=-100.0, max=100.0)

        # NaN CHECK: Verify target features are also valid
        lang_feat = input_dict.get("lang_feat", None)
        if lang_feat is not None and torch.isnan(lang_feat).any():
            print(f"\n🚨 NaN detected in TARGET lang_feat!")
            print(f"  lang_feat NaN count: {torch.isnan(lang_feat).sum().item()}")
            print(f"  lang_feat stats: min={lang_feat.min().item():.6f}, max={lang_feat.max().item():.6f}")
            assert False, "Target lang_feat contains NaN! Check data loading."

        # DEBUG: Verify scaling was applied (only print once)
        if not hasattr(self, '_scale_debug_printed'):
            print(f"\n[DEBUG] Applying dim_scale (with ReLU + epsilon, max_clamp=10.0):")
            print(f"  feat_before mean: {feat_before.mean():.6f}, std: {feat_before.std():.6f}")
            print(f"  feat_after mean: {feat.mean():.6f}, std: {feat.std():.6f}")
            print(f"  dim_scale (raw): {self.dim_scale.detach().cpu().numpy()}")
            print(f"  dim_scale_after_relu: {dim_scale_positive.detach().cpu().numpy()}")
            print(f"  dim_scale_final: {dim_scale_clamped.detach().cpu().numpy()}")
            print(f"  Reduction ratio: {feat.mean().item() / (feat_before.mean().item() + 1e-8):.4f}")
            self._scale_debug_printed = True

        point_feat["feat"] = feat

        # train
        if self.training:
            segment = input_dict["segment"] if "segment" in input_dict.keys() else None

            # CRITICAL FIX: Do NOT normalize target features either
            # Let L2 Loss work with raw features for proper gradient flow
            # lang_feat_normalized = nn.functional.normalize(input_dict["lang_feat"], p=2, dim=1)  # REMOVED
            lang_feat_target = input_dict["lang_feat"]

            loss = self.criteria(
                point_feat["feat"],
                lang_feat_target,
                valid_feat_mask=input_dict["valid_feat_mask"],
                segment=segment,
                epoch_progress=input_dict["epoch_progress"],
            )
            return dict(loss=loss, feat=point_feat["feat"])
        # test
        else:
            return dict(point_feat=point_feat)

    def _chunked_forward(self, input_dict, chunk_size):
        """
        Break the large point set into smaller chunks, pass each chunk through backbone,
        and concat the output features.
        NOTE: This only works if your model's global context isn't critical across chunks.
        """

        # We'll assume "coord" (Nx3 or NxD) is the main key to figure out total #points N.
        # Modify if your data structure is different.
        coords = input_dict["coord"]
        N = coords.shape[0]

        # Prepare a list to store chunk outputs
        chunk_outputs = []

        # We'll do the same logic as normal forward, but inside a loop
        # that processes chunk by chunk.
        is_training = self.training  # track if we are in training or eval

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            # split input_dict into chunks
            chunk_input_dict = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == N:
                    chunk_input_dict[k] = v[start_idx:end_idx]
                elif not isinstance(v, torch.Tensor):
                    # Copy non-tensor values (scalars like grid_size, etc.)
                    chunk_input_dict[k] = v
            if "condition" in input_dict.keys():
                chunk_input_dict["condition"] = input_dict["condition"][0]
            # need to address the 'offset' key separately, which is the same as N
            chunk_input_dict["offset"] = torch.tensor(
                [end_idx - start_idx], device=coords.device
            )
            chunk_point = Point(chunk_input_dict)

            chunk_point_feat = self.backbone(chunk_point)

            # Apply learnable dimension-wise scaling (same as in main forward)
            chunk_feat = chunk_point_feat["feat"]

            # STABILITY CHECK: Ensure dim_scale is valid (not NaN/Inf)
            if torch.isnan(self.dim_scale).any() or torch.isinf(self.dim_scale).any():
                print(f"\n[WARNING] dim_scale contains NaN/Inf in chunked forward! Resetting to safe values.")
                with torch.no_grad():
                    self.dim_scale.copy_(torch.ones(16) * 0.3)
                    self.dim_scale[0] = 1.0

            # Use ReLU to ensure non-negative scaling (negative values become 0)
            # Add a small epsilon (0.01) to prevent zero scaling
            # Then clamp to reasonable upper bound to prevent excessive amplification
            dim_scale_positive = torch.relu(self.dim_scale) + 0.01
            dim_scale_clamped = torch.clamp(dim_scale_positive, max=10.0)
            chunk_feat = chunk_feat * dim_scale_clamped  # [N_chunk, 16] * [16] = [N_chunk, 16]

            # STABILITY CHECK: Detect and clip extreme values (same as main forward)
            chunk_feat_max = chunk_feat.abs().max().item()
            if chunk_feat_max > 100.0:
                print(f"\n⚠️ WARNING: Extreme chunk feature values detected!")
                print(f"  chunk_feat_max: {chunk_feat_max:.6f} (threshold: 100.0)")
                print(f"  Clipping to [-100, 100] to prevent instability...")
                chunk_feat = torch.clamp(chunk_feat, min=-100.0, max=100.0)

            # NaN CHECK: Verify chunk features are valid
            if torch.isnan(chunk_feat).any():
                print(f"\n🚨 NaN detected in chunk_feat after scaling!")
                print(f"  chunk: [{start_idx}:{end_idx}]")
                assert False, "chunk_feat contains NaN! See diagnostic info above."

            chunk_point_feat["feat"] = chunk_feat

            # NOTE: Disabled to allow L2 loss to properly converge - target features are not normalized
            # chunk_point_feat["feat"] = nn.functional.normalize(
            #     chunk_point_feat["feat"], p=2, dim=1
            # )

            if is_training:
                segment = chunk_input_dict.get("segment", None)
                loss = self.criteria(
                    chunk_point_feat["feat"],
                    chunk_input_dict["lang_feat"],
                    valid_feat_mask=chunk_input_dict["valid_feat_mask"],
                    segment=segment,
                    epoch_progress=chunk_input_dict.get("epoch_progress", None),
                )
                chunk_outputs.append(loss)
            else:
                # If eval, store chunk feats to concat
                chunk_outputs.append(chunk_point_feat["feat"])

            # Clean up to free memory before next chunk
            del chunk_point, chunk_point_feat
            torch.cuda.empty_cache()

        if is_training:
            # sum or average the chunk losses
            # e.g., total_loss = sum(chunk_outputs) / len(chunk_outputs)
            total_loss = torch.stack(chunk_outputs).mean()
            return dict(loss=total_loss)
        else:
            full_feat = torch.cat(chunk_outputs, dim=0)  # shape [N, C]
            return dict(point_feat={"feat": full_feat})


@MODELS.register_module()
class DefaultSegmentorSkip(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        # (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module()
class DefaultPretrainer(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        # self.seg_head = (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        # seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(feat, input_dict["clip_feat"])
            return dict(loss=loss)
        # eval
        elif "clip_feat" in input_dict.keys():
            loss = self.criteria(feat, input_dict["clip_feat"])
            return dict(loss=loss, seg_logits=feat)
        # test
        else:
            return dict(seg_logits=feat)
