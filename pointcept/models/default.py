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
        norm_range_min=-1.0,  # Min value for output normalization range
        norm_range_max=1.0,  # Max value for output normalization range
        enable_normalize=True,  # Whether to normalize output to [norm_range_min, norm_range_max]
        enable_output_bias=False,  # Enable learnable bias after tanh to handle biased GT distributions
        output_bias_init=None,  # Initial bias values (e.g., [0.92, 0, 0, ...] for SVD Dim 0)
        feat_dim=16,  # Feature dimension for bias layer
        # 方案 X（2026-08-03）：训练目标在线对齐到 text16 共享空间
        # 每 chunk 的 per-scene SVD 基有旋转/置换歧义 → 模型学"平均波动"≈0（基歧义）。
        # 对齐：每 chunk 用 T 的类均值 Procrustes 对齐到 text16 类均值（与评测 Q 拟合同构），
        # 目标转到 text16 空间后跨 chunk 一致 → 消除基歧义，评测端 Q≈I 可零适配。
        align_text16=False,
        text_embeddings_path=None,  # 锚文本 .pt（768 维），SVD 投影到 svd_rank 维（与 tester 一致）
        svd_rank=16,
        align_min_points=10,  # 每 chunk 参与 Q 拟合的最少有效点数
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        # Enable per-dimension loss tracking when verbose_losses is enabled
        self.criteria = build_criteria(criteria, verbose_losses=verbose_losses, return_per_dim=verbose_losses)

        # 方案 X：加载锚文本并投影到共享子空间（与 tester 的 text16 构造完全一致）
        self.align_text16 = align_text16
        self.align_min_points = align_min_points
        if align_text16:
            assert text_embeddings_path is not None, "align_text16=True 需要 text_embeddings_path"
            import os
            from tools.compute_procrustes_alignment_simple import perform_svd_reduction
            assert os.path.isfile(text_embeddings_path), f"text_embeddings not found: {text_embeddings_path}"
            text768 = torch.load(text_embeddings_path, map_location="cpu", weights_only=True).numpy()
            text16, _, _ = perform_svd_reduction(text768, svd_rank, normalize=False)
            text16 = nn.functional.normalize(torch.from_numpy(text16), p=2, dim=1)
            self.register_buffer("text16", text16)  # [C, svd_rank]
            print(f"[LangPretrainer] align_text16: loaded {os.path.basename(text_embeddings_path)} "
                  f"-> text16 {tuple(text16.shape)} (svd_rank={svd_rank})")

        # Normalization settings
        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max
        self.enable_normalize = enable_normalize

        # Output bias layer to handle biased GT distributions
        # FIX for mode collapse: SVD GT has Dim 0 with mean=0.92 (highly biased positive)
        # tanh output is centered at 0, so we need a learnable bias to shift the distribution
        self.enable_output_bias = enable_output_bias
        self.output_bias = None
        if enable_output_bias:
            # Initialize bias to 0.0 (no shift) or use provided initial values
            # If output_bias_init is provided, use it for initial bias values
            if output_bias_init is not None:
                bias_init = torch.tensor(output_bias_init, dtype=torch.float32)
                self.output_bias = nn.Parameter(bias_init)
            else:
                self.output_bias = nn.Parameter(torch.zeros(feat_dim))

    def _fit_procrustes_q(self, X_c, Y, labels):
        """归一化 + 类平均 + 正交 Procrustes（与评测端 compute_procrustes_Q_cuda_with_labels 同构）。

        X_c: [N, d] 逐点特征; Y: [C, d] 文本（类嵌入）; labels: [N] 每点类 id ∈ [0, C)
        返回 Q [d, d]（detach，目标端预处理不进梯度）。
        """
        X_n = X_c / (X_c.norm(dim=1, keepdim=True) + 1e-8)  # 每点归一化（只保留方向）
        M = Y.shape[0]
        counts = torch.bincount(labels, minlength=M).clamp(min=1).to(X_n.dtype)
        sum_j = torch.zeros(M, X_n.shape[1], device=X_n.device, dtype=X_n.dtype)
        sum_j.index_add_(0, labels.long(), X_n)
        sum_j = sum_j / counts[:, None]  # 类平均（每类等权，避免大类主导）
        Mt = sum_j.t() @ Y  # [d, d]
        U, _, Vt = torch.linalg.svd(Mt, full_matrices=False)
        Q = U @ Vt
        if torch.det(Q) < 0:  # 保 proper rotation（与评测端一致）
            U[:, -1] *= -1
            Q = U @ Vt
        return Q.detach()

    def _canonicalize_sign(self, x, valid_mask):
        """每列最大绝对值取正（与 svd_sign.canonicalize_svd_sign 同规则，GPU）。

        2026-08-03 修复：Q 拟合（torch.linalg.svd 的 Procrustes 解）的列符号跨
        chunk 任意（det 修正只保证 det(Q)=+1，180° 翻转也是 proper rotation），
        对齐后目标 T' 的列符号跨 chunk 不一致 → 模型学"平均符号" → 模式坍缩
        （实测 Dim0 corr -0.23 + Trivial solution）。与 compressed 同规则
        canonicalize（判定用有效行），符号规则由数据确定、跨 chunk 一致。
        """
        out = x.clone()
        for k in range(out.shape[1]):
            col = out[:, k]
            sel = col[valid_mask]
            if sel.numel() == 0:
                continue
            i_max = torch.argmax(torch.abs(sel))
            if abs(sel[i_max]) < 1e-8:
                continue
            if sel[i_max] < 0:
                out[:, k] = -col
        return out

    def _align_target_to_text16(self, target, segment, valid_feat_mask, offset):
        """方案 X：逐 chunk 把目标 T 对齐到 text16 空间（T' = T @ Q_chunk）。

        Q_chunk 由 (T 类均值 → text16 类均值) 的 Procrustes 拟合——与评测 Q 拟合
        完全同构。对齐后目标跨 chunk 一致（都在 text16 空间），消除 per-chunk
        SVD 基的旋转/置换歧义。
        """
        if not self.align_text16:
            return target
        if segment is None or offset is None or len(offset) < 2:
            return target
        aligned = target.clone()
        n_cls = self.text16.shape[0]
        for i in range(len(offset) - 1):
            s, e = int(offset[i]), int(offset[i + 1])
            vm = (
                (valid_feat_mask[s:e] > 0)
                & (segment[s:e] >= 0)
                & (segment[s:e] < n_cls)
            )
            if vm.sum() < self.align_min_points:
                continue
            T_chunk = target[s:e][vm]
            seg_chunk = segment[s:e][vm]
            if len(torch.unique(seg_chunk)) < 2:
                continue  # 单类 chunk 拟合无意义（Q 不确定）
            Q = self._fit_procrustes_q(T_chunk, self.text16, seg_chunk)
            Tp = target[s:e] @ Q
            # 修复（2026-08-03）：Q 列符号跨 chunk 任意 → 对齐后目标符号不一致
            # → 模式坍缩。与 compressed 同规则 canonicalize（判定用有效行）。
            aligned[s:e] = self._canonicalize_sign(
                Tp, (valid_feat_mask[s:e] > 0)
            )
        return aligned

    def forward(self, input_dict, chunk_size=None):
        if (
            chunk_size is not None
            and chunk_size > 0
            and input_dict["coord"].shape[0] > chunk_size
        ):
            return self._chunked_forward(input_dict, chunk_size)
        point = Point(input_dict)
        point_feat = self.backbone(point)

        # 2026-08-03 双源真理修复：normalize 只用于评测分支。
        # 训练分支保留幅度（L1 需要；历史 mode collapse 修复——trainer 已删 tanh，
        # 训练输出 = backbone 原始，与这里一致）。
        if not self.training:
            point_feat["feat"] = nn.functional.normalize(point_feat["feat"], p=2, dim=1)

        # train
        if self.training:
            segment = input_dict.get("segment")
            # 方案 X：目标在线对齐到 text16 空间（逐 chunk，Q 拟合 detach）
            target = self._align_target_to_text16(
                input_dict["lang_feat"], segment,
                input_dict["valid_feat_mask"], input_dict.get("offset"),
            )
            # Pass coord, Gaussian parameters, and scene_path for Rendered2DLoss
            loss = self.criteria(
                point_feat["feat"],
                target,
                valid_feat_mask=input_dict["valid_feat_mask"],
                segment=segment,
                epoch_progress=input_dict["epoch_progress"],
                coord=input_dict.get("coord"),
                opacity=input_dict.get("opacity"),
                quat=input_dict.get("quat"),
                scale=input_dict.get("scale"),
                scene_path=input_dict.get("scene_path"),
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
        coords = input_dict["coord"]
        N = coords.shape[0]
        chunk_outputs = []
        is_training = self.training

        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)

            # Split input_dict into chunks
            chunk_input_dict = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == N:
                    chunk_input_dict[k] = v[start_idx:end_idx]
                elif not isinstance(v, torch.Tensor):
                    chunk_input_dict[k] = v
            if "condition" in input_dict.keys():
                chunk_input_dict["condition"] = input_dict["condition"][0]
            chunk_input_dict["offset"] = torch.tensor([end_idx - start_idx], device=coords.device)

            chunk_point = Point(chunk_input_dict)
            chunk_point_feat = self.backbone(chunk_point)

            if not is_training:
                chunk_point_feat["feat"] = nn.functional.normalize(
                    chunk_point_feat["feat"], p=2, dim=1
                )

            if is_training:
                # 方案 X：目标在线对齐到 text16 空间（chunked 路径，offset 为单 chunk 边界）
                target = self._align_target_to_text16(
                    chunk_input_dict["lang_feat"], chunk_input_dict.get("segment"),
                    chunk_input_dict["valid_feat_mask"], chunk_input_dict.get("offset"),
                )
                # Pass coord, Gaussian parameters, and scene_path for Rendered2DLoss
                loss = self.criteria(
                    chunk_point_feat["feat"],
                    target,
                    valid_feat_mask=chunk_input_dict["valid_feat_mask"],
                    segment=chunk_input_dict.get("segment"),
                    epoch_progress=chunk_input_dict.get("epoch_progress"),
                    coord=chunk_input_dict.get("coord"),
                    opacity=chunk_input_dict.get("opacity"),
                    quat=chunk_input_dict.get("quat"),
                    scale=chunk_input_dict.get("scale"),
                    scene_path=chunk_input_dict.get("scene_path"),
                )
                chunk_outputs.append(loss)
            else:
                chunk_outputs.append(chunk_point_feat["feat"])

            # Clean up memory
            del chunk_point, chunk_point_feat
            torch.cuda.empty_cache()

        if is_training:
            total_loss = torch.stack(chunk_outputs).mean()
            return dict(loss=total_loss)
        else:
            full_feat = torch.cat(chunk_outputs, dim=0)
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
