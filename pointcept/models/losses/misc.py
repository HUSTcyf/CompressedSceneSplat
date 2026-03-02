"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES


@LOSSES.register_module()
class L1Loss(nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
    ):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, valid_feat_mask, **kwargs):
        # L1 Loss: |pred - target| for valid features only
        # Unlike L2, L1 has constant gradient (±1), avoiding vanishing gradient

        # RUNTIME VALIDATION: Ensure data integrity
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        assert valid_feat_mask.shape[0] == pred.shape[0], f"Mask shape mismatch: {valid_feat_mask.shape} vs {pred.shape}"

        # Enhanced NaN detection with diagnostic information
        if torch.isnan(pred).any():
            nan_count = torch.isnan(pred).sum().item()
            nan_indices = torch.where(torch.isnan(pred).any(dim=1))[0][:10]  # First 10 samples with NaN
            print(f"\n🚨 NaN DETECTED IN PRED!")
            print(f"  Total NaN values: {nan_count}")
            print(f"  Samples with NaN (first 10): {nan_indices.tolist()}")
            print(f"  pred stats: min={pred.min().item():.6f}, max={pred.max().item():.6f}")
            print(f"              mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
            print(f"  pred device: {pred.device}, dtype: {pred.dtype}")
            # Check per-dimension NaN
            for dim in range(pred.shape[1]):
                if torch.isnan(pred[:, dim]).any():
                    print(f"    dim[{dim}] has NaN!")
            assert False, "pred contains NaN! See diagnostic info above."

        if torch.isnan(target).any():
            nan_count = torch.isnan(target).sum().item()
            print(f"\n🚨 NaN DETECTED IN TARGET!")
            print(f"  Total NaN values: {nan_count}")
            print(f"  target stats: min={target.min().item():.6f}, max={target.max().item():.6f}")
            print(f"                mean={target.mean().item():.6f}, std={target.std().item():.6f}")
            assert False, "target contains NaN!"

        if torch.isinf(pred).any():
            print(f"\n🚨 Inf DETECTED IN PRED!")
            print(f"  pred stats: min={pred.min().item():.6f}, max={pred.max().item():.6f}")
            assert False, "pred contains Inf!"

        if torch.isinf(target).any():
            print(f"\n🚨 Inf DETECTED IN TARGET!")
            assert False, "target contains Inf!"

        loss = torch.abs(pred[valid_feat_mask] - target[valid_feat_mask]).sum(dim=1)

        if self.reduction == "mean":
            # Average the loss over only the valid samples
            valid_count = valid_feat_mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.sum()  # fallback if there are no valid samples
        elif self.reduction == "sum":
            # Sum the loss over all valid samples
            loss = loss.sum()

        print("l1 loss:", self.loss_weight * loss.item())
        return self.loss_weight * loss


@LOSSES.register_module()
class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss based on feature dimension importance.

    Higher weight for dimensions with higher variance (more important),
    lower weight for dimensions with lower variance (less important/noise).

    This allows the model to focus on optimizing the most important features,
    enabling L1 loss to converge closer to zero for the principal components.
    """
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        loss_weight=1.0,
        weight_strategy="variance",  # "variance", "uniform", "predefined"
        min_weight=0.1,  # Minimum weight for any dimension
        max_weight=3.0,  # Maximum weight for any dimension
        momentum=0.99,  # Momentum for updating importance weights (EMA)
    ):
        super(WeightedL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.weight_strategy = weight_strategy
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.momentum = momentum

        # Running statistics for dimension importance
        self.register_buffer("dim_importance", None)  # Will be initialized on first forward
        self.register_buffer("num_updates", torch.tensor(0.0))

    def _compute_variance_weights(self, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights based on feature variance.

        Dimensions with higher variance are considered more important
        (they carry more information about the data).
        """
        # Extract valid target features
        valid_target = target[valid_mask > 0]  # [M, D]

        if valid_target.numel() == 0:
            return torch.ones(target.shape[1], device=target.device)

        # Compute variance per dimension
        dim_variance = valid_target.var(dim=0)  # [D]

        # Avoid division by zero
        dim_variance = dim_variance.clamp(min=1e-8)

        # Normalize to [min_weight, max_weight]
        variance_min = dim_variance.min()
        variance_max = dim_variance.max()

        if variance_max > variance_min:
            normalized = (dim_variance - variance_min) / (variance_max - variance_min)
            weights = self.min_weight + normalized * (self.max_weight - self.min_weight)
        else:
            weights = torch.ones_like(dim_variance)

        return weights

    def _update_importance(self, target: torch.Tensor, valid_mask: torch.Tensor):
        """Update dimension importance using exponential moving average."""
        valid_target = target[valid_mask > 0]  # [M, D]

        if valid_target.numel() == 0:
            return

        # Compute current importance (variance)
        current_importance = valid_target.var(dim=0)  # [D]

        # Initialize or update with EMA
        if self.dim_importance is None:
            self.dim_importance = current_importance
        else:
            self.dim_importance = (
                self.momentum * self.dim_importance +
                (1 - self.momentum) * current_importance
            )

        self.num_updates += 1

    def _get_importance_weights(self, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Get importance weights based on strategy."""
        feature_dim = target.shape[1]

        if self.weight_strategy == "uniform":
            return torch.ones(feature_dim, device=target.device)

        elif self.weight_strategy == "variance":
            # Compute variance-based weights
            weights = self._compute_variance_weights(target, valid_mask)
            return weights

        elif self.weight_strategy == "ema_variance":
            # Update and use EMA variance
            self._update_importance(target, valid_mask)

            if self.dim_importance is None:
                return torch.ones(feature_dim, device=target.device)

            # Normalize to [min_weight, max_weight]
            importance_min = self.dim_importance.min()
            importance_max = self.dim_importance.max()

            if importance_max > importance_min:
                normalized = (self.dim_importance - importance_min) / (importance_max - importance_min)
                weights = self.min_weight + normalized * (self.max_weight - self.min_weight)
            else:
                weights = torch.ones_like(self.dim_importance)

            return weights

        else:
            raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")

    def forward(self, pred, target, valid_feat_mask, **kwargs):
        """
        Compute weighted L1 loss.

        Args:
            pred: [N, D] predicted features
            target: [N, D] target features
            valid_feat_mask: [N] binary mask for valid features

        Returns:
            Weighted L1 loss
        """
        # Get dimension weights
        dim_weights = self._get_importance_weights(target, valid_feat_mask)  # [D]

        # Extract valid features
        valid_pred = pred[valid_feat_mask > 0]  # [M, D]
        valid_target = target[valid_feat_mask > 0]  # [M, D]

        # Compute absolute difference
        abs_diff = torch.abs(valid_pred - valid_target)  # [M, D]

        # Apply dimension weights
        weighted_diff = abs_diff * dim_weights.unsqueeze(0)  # [M, D]

        # Sum over dimensions
        loss = weighted_diff.sum(dim=1)  # [M]

        if self.reduction == "mean":
            # Average over both samples and dimensions (weighted)
            valid_count = valid_feat_mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        print(f"weighted l1 loss: {self.loss_weight * loss.item():.6f}, weights: [{dim_weights.min().item():.2f}, {dim_weights.max().item():.2f}]")
        return self.loss_weight * loss


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(alpha, (float, list)), (
            "AssertionError: alpha should be of type float"
        )
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(loss_weight, float), (
            "AssertionError: loss_weight should be of type float"
        )
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), (
            "The shape of pred doesn't match the shape of target"
        )
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), (
            "The shape of pred doesn't match the shape of target"
        )
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class CosineSimilarity(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(CosineSimilarity, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, valid_feat_mask, **kwargs):
        # Compute cosine similarity along the feature dimension (assumed to be dim=1)

        # RUNTIME VALIDATION: Ensure data integrity
        assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        assert valid_feat_mask.shape[0] == pred.shape[0], f"Mask shape mismatch: {valid_feat_mask.shape} vs {pred.shape}"

        # Enhanced NaN detection with diagnostic information
        if torch.isnan(pred).any():
            nan_count = torch.isnan(pred).sum().item()
            nan_indices = torch.where(torch.isnan(pred).any(dim=1))[0][:10]
            print(f"\n🚨 NaN DETECTED IN PRED (CosineSimilarity)!")
            print(f"  Total NaN values: {nan_count}")
            print(f"  Samples with NaN (first 10): {nan_indices.tolist()}")
            print(f"  pred stats: min={pred.min().item():.6f}, max={pred.max().item():.6f}")
            print(f"              mean={pred.mean().item():.6f}, std={pred.std().item():.6f}")
            for dim in range(pred.shape[1]):
                if torch.isnan(pred[:, dim]).any():
                    print(f"    dim[{dim}] has NaN!")
            assert False, "pred contains NaN! See diagnostic info above."

        if torch.isnan(target).any():
            print(f"\n🚨 NaN DETECTED IN TARGET (CosineSimilarity)!")
            assert False, "target contains NaN!"

        if torch.isinf(pred).any():
            print(f"\n🚨 Inf DETECTED IN PRED (CosineSimilarity)!")
            assert False, "pred contains Inf!"

        if torch.isinf(target).any():
            print(f"\n🚨 Inf DETECTED IN TARGET (CosineSimilarity)!")
            assert False, "target contains Inf!"

        cos = nn.CosineSimilarity(dim=1)
        loss = 1 - cos(pred[valid_feat_mask], target[valid_feat_mask])

        if self.reduction == "mean":
            # Compute the mean only over valid samples.
            valid_count = valid_feat_mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        print("cosine loss:", self.loss_weight * loss.item())
        return self.loss_weight * loss


@LOSSES.register_module()
class L2Loss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(L2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, valid_feat_mask, **kwargs):
        loss = ((pred[valid_feat_mask] - target[valid_feat_mask]) ** 2).sum(dim=1)

        if self.reduction == "mean":
            # Average the loss over only the valid samples
            valid_count = valid_feat_mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.sum()  # fallback if there are no valid samples
        elif self.reduction == "sum":
            # Sum the loss over all valid samples
            loss = loss.sum()

        # print("l2 loss:", self.loss_weight * loss.item())
        return self.loss_weight * loss


@LOSSES.register_module()
class AggregatedContrastiveLoss(nn.Module):
    def __init__(
        self, temperature=0.2, reduction="mean", loss_weight=1.0, schedule="all"
    ):
        """
        Args:
            temperature (float): Temperature scaling factor.
            reduction (str): 'mean' or 'sum' to average or sum the loss over classes.
            loss_weight (float): A multiplicative factor for the loss.
        """
        super(AggregatedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.schedule = schedule
        if "last_" in self.schedule:
            self.last_percent = float(self.schedule.split("_")[-1]) / 100
            print(
                "Contrastive loss apply in last {}% of training.".format(
                    self.schedule.split("_")[-1]
                )
            )
        elif self.schedule == "all":
            print("Contrastive loss is applied in all epochs.")
        elif self.schedule == "skip":
            print("Contrastive loss is skipped.")

    def forward(
        self, pred, target, valid_feat_mask, segment, epoch_progress=None, **kwargs
    ):
        """
        Args:
            pred (Tensor): Predicted language features of shape [N, D].
            target (Tensor): Ground truth language features (unused in contrastive loss).
            valid_feat_mask (Tensor): Binary mask of shape [N] (1 for valid features).
            segment (Tensor): Semantic segmentation labels of shape [N] (with -1 for ignore index).

        Returns:
            Tensor: The computed aggregated contrastive loss.
        """
        device = pred.device
        if "last_" in self.schedule and epoch_progress is not None:
            if epoch_progress <= (1 - self.last_percent):
                return torch.tensor(0.0, device=device)
        elif self.schedule == "skip":
            return torch.tensor(0.0, device=device)

        # If segmentation is not provided, return 0 loss.
        if segment is None:
            return torch.tensor(0.0, device=device)

        # Select only valid indices (mask > 0 and segment != -1)
        valid_idx = (valid_feat_mask > 0) & (segment != -1)
        if valid_idx.sum() == 0:
            return torch.tensor(0.0, device=device)

        features = pred[valid_idx]  # [M, D]
        labels = segment[valid_idx]  # [M]

        # Find unique semseg labels
        unique_labels = torch.unique(labels)

        aggregated_a = []
        aggregated_b = []
        used_labels = []
        for lab in unique_labels:
            # iterate over each unique class
            indices = (labels == lab).nonzero(as_tuple=True)[0]
            if indices.numel() < 20:
                continue  # insufficient samples (lowered from 100 for GridSample)

            # Shuffle the indices randomly
            perm = indices[torch.randperm(indices.size(0))]
            split = perm.size(0) // 2
            # Ensure we have at least one element in each group
            if split == 0 or (perm.size(0) - split) == 0:
                continue

            group_a = features[perm[:split]]
            group_b = features[perm[split:]]

            # IMPORTANT: Use mean() instead of sum() for aggregation
            # Using sum() causes larger classes to dominate the contrastive loss,
            # as their aggregated vectors have larger magnitudes before normalization.
            # This leads to over-segmentation where the model overfits to large classes
            # and produces noisy/fragmented predictions for smaller classes.
            # mean() ensures equal contribution from all classes regardless of size.
            agg_a = group_a.mean(dim=0)
            agg_b = group_b.mean(dim=0)
            # Alternative: sum() aggregation (causes class imbalance issues)
            # agg_a = group_a.sum(dim=0)
            # agg_b = group_b.sum(dim=0)
            aggregated_a.append(agg_a)
            aggregated_b.append(agg_b)
            used_labels.append(lab)

        if len(aggregated_a) == 0:
            return torch.tensor(0.0, device=device)

        # Stack aggregated features into tensors of shape [C, D] where C is the number of semseg classes.
        aggregated_a = torch.stack(aggregated_a, dim=0)
        aggregated_b = torch.stack(aggregated_b, dim=0)

        # Normalize the aggregated features
        aggregated_a = F.normalize(aggregated_a, p=2, dim=1)
        aggregated_b = F.normalize(aggregated_b, p=2, dim=1)

        # Compute cosine similarity matrix between the two sets and scale by temperature.
        # logits[i, j] = cosine_similarity(aggregated_a[i], aggregated_b[j]) / temperature
        logits = torch.matmul(aggregated_a, aggregated_b.T) / self.temperature

        # The diagonal elements are the positive pairs.
        targets = torch.arange(logits.size(0), device=device)

        # Compute cross-entropy loss in both directions.
        loss_a = F.cross_entropy(logits, targets)
        logits_b = torch.matmul(aggregated_b, aggregated_a.T) / self.temperature
        loss_b = F.cross_entropy(logits_b, targets)
        loss = (loss_a + loss_b) / 2.0

        if self.reduction == "sum":
            loss = loss * logits.size(0)
        # For "mean", cross_entropy already averages over the classes.

        # Optionally print or log the loss
        print("contrastive loss:", self.loss_weight * loss.item(), "classes:", len(used_labels))

        return self.loss_weight * loss


@LOSSES.register_module()
class SVDWeightedL1Loss(nn.Module):
    """
    SVD-Weighted L1 Loss: Weights dimensions based on SVD ordering.

    For SVD-compressed features, lower indices (dim 0, 1, 2, ...) capture more variance
    (principal components), while higher indices (dim 13, 14, 15) capture less variance (noise).

    This loss applies exponential decay weights: weight[i] = base * (decay_rate ^ i)
    This ensures the model focuses on optimizing the most important dimensions first.

    Args:
        base_weight: Base weight for dimension 0 (most important)
        decay_rate: Exponential decay rate per dimension (0 < decay_rate < 1)
        min_weight: Minimum weight for any dimension (prevents zero gradients)
        loss_weight: Overall loss weight multiplier
        reduction: "mean" or "sum"
    """
    def __init__(
        self,
        base_weight=1.0,
        decay_rate=0.85,
        min_weight=0.05,
        loss_weight=1.0,
        reduction="mean",
    ):
        super(SVDWeightedL1Loss, self).__init__()
        self.base_weight = base_weight
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, valid_feat_mask, **kwargs):
        """
        Compute SVD-weighted L1 loss.

        Args:
            pred: [N, D] predicted features (typically 16-dim SVD compressed)
            target: [N, D] target features (SVD compressed)
            valid_feat_mask: [N] binary mask for valid features

        Returns:
            SVD-weighted L1 loss
        """
        # Extract valid features
        valid_pred = pred[valid_feat_mask > 0]  # [M, D]
        valid_target = target[valid_feat_mask > 0]  # [M, D]

        if valid_pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad)

        D = valid_pred.shape[1]  # Feature dimension (typically 16)

        # Compute dimension weights based on SVD ordering
        # Early dimensions (low index) get higher weights
        weights = torch.tensor(
            [max(self.min_weight, self.base_weight * (self.decay_rate ** i)) for i in range(D)],
            device=valid_pred.device,
            dtype=valid_pred.dtype
        )  # [D]

        # Compute absolute difference
        abs_diff = torch.abs(valid_pred - valid_target)  # [M, D]

        # Apply dimension weights
        weighted_diff = abs_diff * weights.unsqueeze(0)  # [M, D]

        # Sum over dimensions
        loss = weighted_diff.sum(dim=1)  # [M]

        if self.reduction == "mean":
            valid_count = valid_feat_mask.sum()
            if valid_count > 0:
                loss = loss.sum() / valid_count
            else:
                loss = loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()

        print(f"svd_weighted_l1: {self.loss_weight * loss.item():.6f}, "
              f"weight_range: [{weights[0]:.3f}, {weights[-1]:.3f}], "
              f"decay_rate={self.decay_rate}")

        return self.loss_weight * loss


@LOSSES.register_module()
class ValidNonValidContrastiveLoss(nn.Module):
    """
    Contrastive loss between valid and non-valid Gaussians.
    
    This loss encourages the model to:
    1. Make valid Gaussians' features similar to target features
    2. Make non-valid Gaussians' features different from target features
    3. Separate valid and non-valid features in the embedding space
    
    Args:
        temperature: Temperature parameter for contrastive learning (default: 0.1)
        margin: Minimum margin for separating valid from non-valid (default: 0.5)
        reduction: Reduction method ("mean" or "sum")
        loss_weight: Weight for this loss in the total loss
    """
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.5,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, pred, target, valid_feat_mask, **kwargs):
        """
        Args:
            pred: Predicted features [N, D]
            target: Target features [N, D]
            valid_feat_mask: Real mask [N] where 1=valid (non-zero features), 0=non-valid (zero features)

        Returns:
            Scalar loss value
        """
        device = pred.device

        # Separate valid and non-valid features
        valid_mask = (valid_feat_mask > 0)
        non_valid_mask = (valid_feat_mask == 0)
        
        # Check if we have both valid and non-valid samples
        if not valid_mask.any() or not non_valid_mask.any():
            # If all samples are the same type, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=pred.requires_grad)
        
        valid_pred = pred[valid_mask]      # [N_valid, D]
        non_valid_pred = pred[non_valid_mask]  # [N_non_valid, D]
        
        valid_target = target[valid_mask]   # [N_valid, D]
        non_valid_target = target[non_valid_mask]  # [N_non_valid, D]
        
        # Normalize features for stable similarity computation
        valid_pred = F.normalize(valid_pred, p=2, dim=1)
        non_valid_pred = F.normalize(non_valid_pred, p=2, dim=1)
        
        # ============ Loss 1: Valid features should match target ============
        # Cosine similarity between valid predictions and target
        valid_sim = F.cosine_similarity(valid_pred, valid_target, dim=1)  # [N_valid]
        loss_valid_match = 1 - valid_sim.mean()  # Encourage similarity
        
        # ============ Loss 2: Non-valid features should differ from target ============
        # Cosine similarity between non-valid predictions and target
        non_valid_sim = F.cosine_similarity(non_valid_pred, non_valid_target, dim=1)  # [N_non_valid]
        # Push similarity below (1 - margin)
        loss_non_valid_separate = F.relu(non_valid_sim - (1 - self.margin)).mean()
        
        # ============ Loss 3: Valid and non-valid should be separated ============
        # Compute pairwise similarity between valid and non-valid
        # valid_pred: [N_valid, D], non_valid_pred: [N_non_valid, D]
        # sim_matrix: [N_valid, N_non_valid]
        sim_matrix = torch.matmul(valid_pred, non_valid_pred.T) / self.temperature
        # We want valid and non-valid to be dissimilar (low cosine similarity)
        # Use contrastive loss: encourage similarities to be low (below threshold)
        threshold = 0.0  # Cosine similarity threshold
        loss_separation = F.relu(sim_matrix - threshold).mean()
        
        # Combine losses
        total_loss = (
            loss_valid_match +           # Valid should match target
            loss_non_valid_separate +     # Non-valid should differ from target
            loss_separation               # Valid and non-valid should be separated
        )
        
        if self.reduction == "sum":
            total_loss = total_loss * pred.size(0)
        # For "mean", loss is already averaged
        
        return self.loss_weight * total_loss
