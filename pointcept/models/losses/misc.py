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

        # Print disabled - now handled by Criteria builder for consolidated output
        # print("l1 loss:", self.loss_weight * loss.item())
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

        # Print disabled - now handled by Criteria builder for consolidated output
        # print(f"weighted l1 loss: {self.loss_weight * loss.item():.6f}, weights: [{dim_weights.min().item():.2f}, {dim_weights.max().item():.2f}]")
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

        # Print disabled - now handled by Criteria builder for consolidated output
        # print("cosine loss:", self.loss_weight * loss.item())
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
    SVD-Weighted L1 Loss: Weights dimensions based on variance or SVD ordering.

    Two modes supported:
    1. "static": Exponential decay weights based on dimension index (d[0] > d[1] > ... > d[15])
    2. "variance": Data-driven weights based on actual variance of each dimension
       - Higher variance = more information = higher weight
       - Uses EMA (exponential moving average) for stable variance estimation

    Args:
        base_weight: Base weight for the highest-variance dimension
        decay_rate: Exponential decay rate per dimension (for "static" mode only)
        min_weight: Minimum weight for any dimension (prevents zero gradients)
        loss_weight: Overall loss weight multiplier
        reduction: "mean" or "sum"
        weight_strategy: "static" (exponential decay) or "variance" (data-driven)
        variance_momentum: EMA momentum for variance estimation (default: 0.99)
    """
    def __init__(
        self,
        base_weight=1.0,
        decay_rate=0.85,
        min_weight=0.05,
        loss_weight=1.0,
        reduction="mean",
        weight_strategy="variance",  # "static" or "variance"
        variance_momentum=0.99,
    ):
        super(SVDWeightedL1Loss, self).__init__()
        self.base_weight = base_weight
        self.decay_rate = decay_rate
        self.min_weight = min_weight
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.weight_strategy = weight_strategy
        self.variance_momentum = variance_momentum

        # Running statistics for variance-based weighting
        self.register_buffer("dim_variance", None)  # EMA of per-dimension variance
        self.register_buffer("num_updates", torch.tensor(0.0))
        self.register_buffer("weights", None)  # Cached weights

    def _compute_variance_weights(self, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute importance weights based on feature variance.

        Dimensions with higher variance are considered more important
        (they carry more information about the data).

        Args:
            target: [N, D] target features
            valid_mask: [N] binary mask for valid features

        Returns:
            [D] weight tensor
        """
        # Extract valid target features
        valid_target = target[valid_mask > 0]  # [M, D]

        if valid_target.numel() == 0:
            D = target.shape[1]
            return torch.ones(D, device=target.device, dtype=target.dtype)

        # Compute variance per dimension
        dim_variance = valid_target.var(dim=0)  # [D]

        # Update EMA of variance
        if self.dim_variance is None:
            self.dim_variance = dim_variance
        else:
            self.dim_variance = (
                self.variance_momentum * self.dim_variance +
                (1 - self.variance_momentum) * dim_variance
            )

        self.num_updates += 1

        # Compute weights based on variance
        # Higher variance -> higher weight
        variance_min = self.dim_variance.min()
        variance_max = self.dim_variance.max()

        if variance_max > variance_min:
            # Normalize variance to [min_weight, base_weight]
            normalized = (self.dim_variance - variance_min) / (variance_max - variance_min)
            weights = self.min_weight + normalized * (self.base_weight - self.min_weight)
        else:
            # All dimensions have the same variance
            weights = torch.full_like(self.dim_variance, self.base_weight)

        # Cache the weights
        self.weights = weights.detach().clone()

        return weights

    def _compute_static_weights(self, D: int, device: torch.device) -> torch.Tensor:
        """
        Compute static exponential decay weights.

        Early dimensions (low index) get higher weights.
        """
        weights = torch.tensor(
            [max(self.min_weight, self.base_weight * (self.decay_rate ** i)) for i in range(D)],
            device=device,
            dtype=torch.float32
        )

        # Cache the weights
        self.weights = weights.detach().clone()

        return weights

    def forward(self, pred, target, valid_feat_mask, return_per_dim=False, **kwargs):
        """
        Compute SVD-weighted L1 loss.

        Args:
            pred: [N, D] predicted features (typically 16-dim SVD compressed)
            target: [N, D] target features (SVD compressed)
            valid_feat_mask: [N] binary mask for valid features
            return_per_dim: If True, also return per-dimension losses

        Returns:
            SVD-weighted L1 loss (or tuple with loss dict if return_per_dim=True)
        """
        # Extract valid features
        valid_pred = pred[valid_feat_mask > 0]  # [M, D]
        valid_target = target[valid_feat_mask > 0]  # [M, D]

        if valid_pred.numel() == 0:
            loss = torch.tensor(0.0, device=pred.device, requires_grad=pred.requires_grad)
            if return_per_dim:
                return (loss, {'per_dim_losses': torch.zeros(pred.shape[1], device=pred.device)})
            return loss

        D = valid_pred.shape[1]  # Feature dimension (typically 16)

        # Compute dimension weights based on strategy
        if self.weight_strategy == "variance":
            weights = self._compute_variance_weights(target, valid_feat_mask)
            strategy_str = f"variance (updates={self.num_updates.item():.0f})"
        else:  # "static"
            weights = self._compute_static_weights(D, valid_pred.device)
            strategy_str = f"static (decay={self.decay_rate})"

        # Compute absolute difference
        abs_diff = torch.abs(valid_pred - valid_target)  # [M, D]

        # Store per-dimension losses (before weighting) for visualization
        per_dim_losses = abs_diff.mean(dim=0)  # [D]

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

        weighted_loss = self.loss_weight * loss

        # Print disabled - now handled by Criteria builder for consolidated output
        # # Print with strategy info and variance statistics
        # if self.weight_strategy == "variance" and self.dim_variance is not None:
        #     var_stats = self.dim_variance.detach().cpu().numpy()
        #     print(f"svd_weighted_l1 ({strategy_str}): loss={weighted_loss.item():.6f}, "
        #           f"weight_range=[{weights.min():.3f}, {weights.max():.3f}], "
        #           f"variance_range=[{var_stats.min():.6f}, {var_stats.max():.6f}]")
        # else:
        #     print(f"svd_weighted_l1 ({strategy_str}): loss={weighted_loss.item():.6f}, "
        #           f"weight_range=[{weights.min():.3f}, {weights.max():.3f}]")

        if return_per_dim:
            # Return tuple (loss, dict) for compatibility with verbose_losses
            loss_dict = {
                'per_dim_losses': per_dim_losses.detach().cpu(),  # [D] per-dimension loss (unweighted)
                'per_dim_weights': weights.detach().cpu(),  # [D] weights applied to each dimension
            }
            return (weighted_loss, loss_dict)
        return weighted_loss


@LOSSES.register_module()
class SpatialSmoothnessLoss(nn.Module):
    """
    Spatial Smoothness Loss for 3D Gaussian Feature Prediction.

    This loss encourages spatial consistency by penalizing large feature differences
    between neighboring 3D Gaussians. This prevents "salt-and-pepper" noise in the
    predicted features.

    The key insight: Features should vary smoothly across space for semantically
    consistent regions. Sudden jumps indicate spatially inconsistent predictions.

    Args:
        neighbor_k: Number of nearest neighbors to consider (default: 16)
        radius: Maximum distance for neighbors (default: 0.05, adjusted for grid_size)
        reduction: Reduction method ("mean" or "sum")
        loss_weight: Weight for this loss in the total loss
        warmup_epochs: Epochs before applying this loss (default: 3)
        decay_start: Epoch when weight starts decaying (default: 10)
    """
    def __init__(
        self,
        neighbor_k: int = 16,
        radius: float = 0.05,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        warmup_epochs: int = 3,
        decay_start: int = 10,
    ):
        super().__init__()
        self.neighbor_k = neighbor_k
        self.radius = radius
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.warmup_epochs = warmup_epochs
        self.decay_start = decay_start

    def forward(self, pred, target, valid_feat_mask, coord=None, epoch_progress=None, **kwargs):
        """
        Args:
            pred: Predicted features [N, D]
            target: Target features [N, D] (not used directly but for interface consistency)
            valid_feat_mask: Valid feature mask [N]
            coord: 3D coordinates [N, 3] for neighborhood computation
            epoch_progress: Current epoch progress [0, 1] for weight scheduling

        Returns:
            Scalar loss value
        """
        if coord is None:
            # If no coordinates provided, can't compute spatial smoothness
            return torch.tensor(0.0, device=pred.device)

        # Compute progressive weight based on epoch progress
        # Start with 0 weight, ramp up, then decay
        if epoch_progress is not None:
            current_epoch = epoch_progress * 100  # Assuming max 100 epochs
            if current_epoch < self.warmup_epochs:
                weight = 0.0
            elif current_epoch < self.decay_start:
                # Linear ramp up
                progress = (current_epoch - self.warmup_epochs) / (self.decay_start - self.warmup_epochs)
                weight = progress * self.loss_weight
            else:
                # Linear decay after decay_start
                decay_progress = min(1.0, (current_epoch - self.decay_start) / 20.0)
                weight = self.loss_weight * (1.0 - 0.5 * decay_progress)
        else:
            weight = self.loss_weight

        if weight == 0:
            return torch.tensor(0.0, device=pred.device)

        # Only consider valid points
        valid_mask = (valid_feat_mask > 0)
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=pred.device)

        # Extract valid coordinates and predictions
        valid_coord = coord[valid_mask]  # [M, 3]
        valid_pred = pred[valid_mask]     # [M, D]

        # Compute pairwise distances efficiently
        # Use chunked computation to avoid O(N^2) memory blowup
        M = valid_coord.shape[0]
        if M > 100000:
            # For large point clouds, sample random subset
            indices = torch.randperm(M, device=pred.device)[:50000]
            valid_coord = valid_coord[indices]
            valid_pred = valid_pred[indices]
            M = valid_coord.shape[0]

        # Compute squared distances
        dists = torch.cdist(valid_coord, valid_coord, p=2)  # [M, M]

        # For each point, find k nearest neighbors (excluding self)
        # Set diagonal to infinity so self is not selected
        dists.fill_diagonal_(float('inf'))

        # Find k nearest neighbors
        knn_dists, knn_indices = torch.topk(dists, k=min(self.neighbor_k, M-1), dim=1, largest=False)

        # Filter by radius
        radius_mask = knn_dists < self.radius

        # Compute feature differences for neighbors within radius
        # valid_pred: [M, D]
        # knn_indices: [M, K]
        neighbor_feats = valid_pred[knn_indices]  # [M, K, D]

        # Expand valid_pred for broadcasting
        valid_pred_expanded = valid_pred.unsqueeze(1)  # [M, 1, D]

        # Compute L2 distance between point and its neighbors
        feat_diff = (valid_pred_expanded - neighbor_feats) ** 2  # [M, K, D]
        feat_diff = feat_diff.sum(dim=2)  # [M, K] - L2 distance per neighbor

        # Apply radius mask
        feat_diff = feat_diff * radius_mask.float()

        # Average over valid neighbors (those within radius)
        valid_neighbor_count = radius_mask.sum(dim=1).clamp(min=1)  # [M], avoid div by zero
        smoothness_loss = feat_diff.sum(dim=1) / valid_neighbor_count  # [M]

        # Average over all points
        if self.reduction == "mean":
            loss = smoothness_loss.mean()
        else:
            loss = smoothness_loss.sum()

        return weight * loss


@LOSSES.register_module()
class Rendered2DLoss(nn.Module):
    """
    Rendered 2D Loss using Gaussian Splatting for Spatially Consistent Feature Prediction.

    This loss enforces spatial consistency by comparing 2D rendered features instead of
    raw 3D features. The rendering process naturally aggregates spatial information through
    Gaussian splatting - neighboring Gaussians contribute to the same pixels.

    Key features:
    - Loads pre-rendered GT feature maps from renders_npy folders
    - Renders predicted features using gsplat's CUDA rasterization
    - Computes L1 loss between rendered predicted and GT features
    - Uses all available rendered views for each scene

    Args:
        gaussian_train_root: Root path to gaussian_train directory (default: "gaussian_train")
        datasets_root: Root path to datasets directory (default: "datasets")
        loss_weight: Maximum weight for this loss (default: 1.0)
        warmup_progress: Training progress before applying this loss, 0-1 range (default: 0.015 for 3/200 epochs)
        target_progress: Training progress when weight reaches loss_weight, 0-1 range (default: 0.5 for 50%)
        max_num_views: Maximum number of views to use per scene (default: 4, to save memory)
    """
    def __init__(
        self,
        gaussian_train_root: str = "gaussian_train",
        datasets_root: str = "datasets",
        loss_weight: float = 1.0,
        warmup_progress: float = 0.015,  # 3/200 = 0.015
        target_progress: float = 0.5,  # 50% of training
        max_num_views: int = 4,
    ):
        super().__init__()
        self.gaussian_train_root = gaussian_train_root
        self.datasets_root = datasets_root
        self.loss_weight = loss_weight
        self.warmup_progress = warmup_progress
        self.target_progress = target_progress
        self.max_num_views = max_num_views

        # Cache for loaded GT renders and camera parameters
        self._gt_renders_cache = {}
        self._camera_params_cache = {}

    def _load_gt_renders(self, scene_path: str) -> list:
        """Load GT rendered feature maps from renders_npy folder."""
        import os
        import numpy as np

        # Handle batched data: scene_path might be a list
        if isinstance(scene_path, list):
            if len(scene_path) == 0:
                return []
            scene_path = scene_path[0]  # Use first scene path

        # Extract dataset and scene name from scene_path
        # scene_path can be absolute: /new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS/train/sofa
        # or relative: gaussian_train/3DOVS/train/sofa
        # Find 'gaussian_train' in the path and extract dataset and scene after it
        if 'gaussian_train' in scene_path:
            # Split on 'gaussian_train' and take the second part
            parts = scene_path.split('gaussian_train')
            if len(parts) < 2:
                return []
            path_after = parts[1].lstrip('/')  # e.g., "3DOVS/train/sofa"
            sub_parts = path_after.split('/')
            if len(sub_parts) < 3:
                return []
            dataset = sub_parts[0]  # e.g., "3DOVS"
            # The scene might be in 'train/{scene}' or just '{scene}'
            if sub_parts[1] == 'train' and len(sub_parts) >= 3:
                scene = sub_parts[2]  # e.g., "sofa"
            else:
                scene = sub_parts[1]  # Fallback
        else:
            # Fallback to original relative path logic
            parts = scene_path.split('/')
            if len(parts) < 4:
                return []
            dataset = parts[1]
            scene = parts[3]

        cache_key = f"{dataset}/{scene}"
        if cache_key in self._gt_renders_cache:
            return self._gt_renders_cache[cache_key]

        # Build path to renders_npy folder
        renders_npy_path = f"{self.gaussian_train_root}/{dataset}/train/{scene}/renders_npy"

        if not os.path.exists(renders_npy_path):
            self._gt_renders_cache[cache_key] = []
            return []

        # Load all .npy files in renders_npy folder
        gt_renders = []
        for fname in sorted(os.listdir(renders_npy_path)):
            if fname.endswith('.npy'):
                fpath = os.path.join(renders_npy_path, fname)
                try:
                    render_data = np.load(fpath)  # Shape: (H, W, D)
                    gt_renders.append({
                        'name': fname[:-4],  # Remove .npy extension
                        'data': render_data.astype(np.float32),
                    })
                except Exception as e:
                    print(f"Warning: Failed to load {fpath}: {e}")

        self._gt_renders_cache[cache_key] = gt_renders
        return gt_renders

    def _load_camera_params(self, scene_path: str) -> dict:
        """Load camera parameters from test cameras only.

        Note: GT renders are pre-rendered from test views, so we only need test camera parameters.
        This saves storage by not loading train camera parameters.

        Priority order:
        1. Scannet format: transforms_test.json
        2. COLMAP format: sparse/0/ (filtered to match renders_npy files)
        """
        import os

        # Handle batched data: scene_path might be a list
        if isinstance(scene_path, list):
            if len(scene_path) == 0:
                return {}
            scene_path = scene_path[0]  # Use first scene path

        # Extract dataset and scene name from scene_path
        # scene_path can be absolute: /new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS/train/sofa
        if 'gaussian_train' in scene_path:
            parts = scene_path.split('gaussian_train')
            if len(parts) < 2:
                return {}
            path_after = parts[1].lstrip('/')  # e.g., "3DOVS/train/sofa"
            sub_parts = path_after.split('/')
            if len(sub_parts) < 3:
                return {}
            dataset = sub_parts[0]
            if sub_parts[1] == 'train' and len(sub_parts) >= 3:
                scene = sub_parts[2]
            else:
                scene = sub_parts[1]
        else:
            # Fallback to original relative path logic
            parts = scene_path.split('/')
            if len(parts) < 4:
                return {}
            dataset = parts[1]
            scene = parts[3]

        cache_key = f"{dataset}/{scene}"
        if cache_key in self._camera_params_cache:
            return self._camera_params_cache[cache_key]

        import os

        # First, get list of GT render names (from renders_npy folder)
        gt_render_names = self._get_gt_render_names(scene_path)

        # Try Scannet format: transforms_test.json
        possible_json_paths = [
            f"{self.datasets_root}/{dataset}/{scene}/transforms_test.json",
            f"{self.datasets_root}/{dataset}/label/{scene}/transforms_test.json",
        ]

        camera_params = None
        for json_path in possible_json_paths:
            if os.path.exists(json_path):
                try:
                    import json
                    with open(json_path, 'r') as f:
                        camera_params = json.load(f)
                    self._camera_params_cache[cache_key] = camera_params
                    return camera_params
                except Exception as e:
                    pass

        # Try COLMAP format: sparse/0/ directory (filtered to test views only)
        possible_colmap_paths = [
            f"{self.datasets_root}/{dataset}/{scene}/sparse/0/",
            f"{self.datasets_root}/{dataset}/{scene}/undistorted/sparse/0/",
            f"{self.datasets_root}/{dataset}/{scene}/distorted/sparse/0/",
        ]

        for colmap_dir in possible_colmap_paths:
            if os.path.exists(colmap_dir):
                camera_params = self._load_colmap_params(colmap_dir, test_frame_names=gt_render_names)
                if camera_params is not None and len(camera_params.get('frames', [])) > 0:
                    self._camera_params_cache[cache_key] = camera_params
                    return camera_params

        self._camera_params_cache[cache_key] = {}
        return {}

    def _get_gt_render_names(self, scene_path: str) -> set:
        """Get set of GT render filenames (without extension) from renders_npy folder."""
        import os

        # Handle batched data: scene_path might be a list
        if isinstance(scene_path, list):
            if len(scene_path) == 0:
                return set()
            scene_path = scene_path[0]  # Use first scene path

        # Extract dataset and scene name from scene_path
        # scene_path can be absolute: /new_data/cyf/projects/SceneSplat/gaussian_train/3DOVS/train/sofa
        if 'gaussian_train' in scene_path:
            parts = scene_path.split('gaussian_train')
            if len(parts) < 2:
                return set()
            path_after = parts[1].lstrip('/')  # e.g., "3DOVS/train/sofa"
            sub_parts = path_after.split('/')
            if len(sub_parts) < 3:
                return set()
            dataset = sub_parts[0]
            if sub_parts[1] == 'train' and len(sub_parts) >= 3:
                scene = sub_parts[2]
            else:
                scene = sub_parts[1]
        else:
            # Fallback to original relative path logic
            parts = scene_path.split('/')
            if len(parts) < 4:
                return set()
            dataset = parts[1]
            scene = parts[3]

        # Build path to renders_npy folder
        renders_npy_path = f"{self.gaussian_train_root}/{dataset}/train/{scene}/renders_npy"

        if not os.path.exists(renders_npy_path):
            return set()

        # Get all .npy filenames and remove extension
        render_names = set()
        for fname in os.listdir(renders_npy_path):
            if fname.endswith('.npy'):
                # Remove .npy extension and any directory prefix
                name = os.path.splitext(fname)[0]
                render_names.add(name)

        return render_names

    def _load_colmap_params(self, colmap_dir: str, test_frame_names: set = None) -> dict:
        """Load camera parameters from COLMAP sparse directory.

        Uses custom binary reader to avoid pycolmap dependency.

        Args:
            colmap_dir: Path to COLMAP sparse directory (e.g., sparse/0/)
            test_frame_names: Set of test frame names to filter (e.g., from renders_npy)

        Returns a dict with the same format as Scannet transforms_test.json:
        {
            'w': width,
            'h': height,
            'fl_x': focal_length_x,
            'fl_y': focal_length_y,
            'cx': principal_point_x,
            'cy': principal_point_y,
            'frames': [
                {
                    'file_path': image_name,
                    'transform_matrix': 4x4 world-to-camera matrix
                },
                ...
            ]
        }
        """
        import os
        import numpy as np
        import struct
        import collections

        # Define COLMAP camera models (minimal set for PINHOLE)
        CameraModel = collections.namedtuple(
            "CameraModel", ["model_id", "model_name", "num_params"]
        )
        CAMERA_MODELS = {
            CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
            CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
            CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
            CameraModel(model_id=3, model_name="RADIAL", num_params=5),
            CameraModel(model_id=4, model_name="OPENCV", num_params=8),
        }
        CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                                for camera_model in CAMERA_MODELS])

        def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
            """Read and unpack the next bytes from a binary file."""
            data = fid.read(num_bytes)
            return struct.unpack(endian_character + format_char_sequence, data)

        def qvec2rotmat(qvec):
            """Convert quaternion to rotation matrix."""
            return np.array([
                [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
                 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
                [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
                 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
                [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
            ])

        if not os.path.exists(colmap_dir):
            return None

        # Try binary format first
        cameras_file = os.path.join(colmap_dir, "cameras.bin")
        images_file = os.path.join(colmap_dir, "images.bin")

        # Fall back to text format
        if not os.path.exists(cameras_file) or not os.path.exists(images_file):
            cameras_file = os.path.join(colmap_dir, "cameras.txt")
            images_file = os.path.join(colmap_dir, "images.txt")
            if not os.path.exists(cameras_file) or not os.path.exists(images_file):
                return None

        try:
            # Read cameras
            cameras = {}
            if cameras_file.endswith('.bin'):
                with open(cameras_file, "rb") as fid:
                    num_cameras = read_next_bytes(fid, 8, "Q")[0]
                    for _ in range(num_cameras):
                        camera_properties = read_next_bytes(
                            fid, num_bytes=24, format_char_sequence="iiQQ"
                        )
                        camera_id = camera_properties[0]
                        model_id = camera_properties[1]
                        model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
                        width = camera_properties[2]
                        height = camera_properties[3]
                        num_params = CAMERA_MODEL_IDS[model_id].num_params
                        params = read_next_bytes(
                            fid,
                            num_bytes=8 * num_params,
                            format_char_sequence="d" * num_params,
                        )
                        cameras[camera_id] = {
                            'id': camera_id,
                            'model': model_name,
                            'width': width,
                            'height': height,
                            'params': np.array(params),
                        }
            else:  # .txt format
                with open(cameras_file, "r") as fid:
                    for line in fid:
                        line = line.strip()
                        if len(line) > 0 and line[0] != "#":
                            elems = line.split()
                            camera_id = int(elems[0])
                            model = elems[1]
                            width = int(elems[2])
                            height = int(elems[3])
                            params = np.array(tuple(map(float, elems[4:])))
                            cameras[camera_id] = {
                                'id': camera_id,
                                'model': model,
                                'width': width,
                                'height': height,
                                'params': params,
                            }

            # Read images
            images = {}
            if images_file.endswith('.bin'):
                with open(images_file, "rb") as fid:
                    num_reg_images = read_next_bytes(fid, 8, "Q")[0]
                    for _ in range(num_reg_images):
                        binary_image_properties = read_next_bytes(
                            fid, num_bytes=64, format_char_sequence="idddddddi"
                        )
                        image_id = binary_image_properties[0]
                        qvec = np.array(binary_image_properties[1:5])
                        tvec = np.array(binary_image_properties[5:8])
                        camera_id = binary_image_properties[8]
                        image_name = ""
                        current_char = read_next_bytes(fid, 1, "c")[0]
                        while current_char != b"\x00":
                            image_name += current_char.decode("utf-8")
                            current_char = read_next_bytes(fid, 1, "c")[0]
                        num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
                        # Skip 2D points
                        fid.read(24 * num_points2D)
                        images[image_id] = {
                            'id': image_id,
                            'qvec': qvec,
                            'tvec': tvec,
                            'camera_id': camera_id,
                            'name': image_name,
                        }
            else:  # .txt format
                with open(images_file, "r") as fid:
                    while True:
                        line = fid.readline()
                        if not line:
                            break
                        line = line.strip()
                        if len(line) > 0 and line[0] != "#":
                            elems = line.split()
                            image_id = int(elems[0])
                            qvec = np.array(tuple(map(float, elems[1:5])))
                            tvec = np.array(tuple(map(float, elems[5:8])))
                            camera_id = int(elems[8])
                            image_name = elems[9]
                            # Skip the next line with 2D points
                            fid.readline()
                            images[image_id] = {
                                'id': image_id,
                                'qvec': qvec,
                                'tvec': tvec,
                                'camera_id': camera_id,
                                'name': image_name,
                            }

            if len(images) == 0:
                return None

            # Get intrinsics from the first camera (assuming single camera setup)
            first_image_id = list(images.keys())[0]
            first_image = images[first_image_id]
            cam = cameras[first_image['camera_id']]

            # Image dimensions and intrinsics
            w = cam['width']
            h = cam['height']
            params = cam['params']

            # Extract intrinsics based on camera model
            if cam['model'] in ['PINHOLE', 'OPENCV']:
                fl_x = params[0]
                fl_y = params[1]
                cx = params[2]
                cy = params[3]
            elif cam['model'] == 'SIMPLE_PINHOLE':
                fl_x = fl_y = params[0]
                cx = params[1]
                cy = params[2]
            elif cam['model'] == 'SIMPLE_RADIAL':
                fl_x = fl_y = params[0]
                cx = params[1]
                cy = params[2]
            else:
                # Default fallback
                fl_x = fl_y = params[0] if len(params) > 0 else w / 2
                cx = params[1] if len(params) > 1 else w / 2
                cy = params[2] if len(params) > 2 else h / 2

            # Build frames list (filtered to test frames only if test_frame_names provided)
            frames = []
            for img_id in images:
                im = images[img_id]

                # Extract frame name from image path
                image_name = os.path.basename(im['name'])
                frame_name = os.path.splitext(image_name)[0]  # Remove extension

                # Filter to test frames only
                if test_frame_names is not None:
                    if frame_name not in test_frame_names:
                        continue

                # Get world-to-camera matrix from quaternion
                rot = qvec2rotmat(im['qvec'])  # 3x3 rotation matrix
                trans = im['tvec'].reshape(3, 1)  # 3x1 translation vector

                # Construct 4x4 world-to-camera matrix
                bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
                w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)

                frames.append({
                    'file_path': frame_name,
                    'transform_matrix': w2c.tolist(),
                })

            if len(frames) == 0:
                return None

            return {
                'w': w,
                'h': h,
                'fl_x': fl_x,
                'fl_y': fl_y,
                'cx': cx,
                'cy': cy,
                'frames': frames,
            }

        except Exception as e:
            print(f"Warning: Failed to load COLMAP cameras from {colmap_dir}: {e}")
            return None

    def forward(
        self,
        pred,
        target,
        valid_feat_mask,
        coord=None,
        opacity=None,
        quat=None,
        scale=None,
        scene_path=None,
        epoch_progress=None,
        scenario=None,
        **kwargs
    ):
        """
        Args:
            pred: Predicted features [N, D]
            target: Target features [N, D] (not used, we use GT renders)
            valid_feat_mask: Valid feature mask [N]
            coord: 3D coordinates [N, 3]
            opacity: Gaussian opacity [N, 1]
            quat: Gaussian rotation quaternions [N, 4]
            scale: Gaussian scale [N, 3]
            scene_path: Path to scene directory (for loading GT renders and cameras)
            epoch_progress: Current epoch progress [0, 1]
            scenario: Training scenario ('dense', 'half', 'single') - only compute loss for 'dense'

        Returns:
            Scalar loss value
        """
        import torch
        import math
        import os
        import numpy as np
        from gsplat import rasterization

        device = pred.device

        # Only compute rendered2d loss for dense scenario to avoid Gaussian sampling issues
        if scenario is not None and scenario != 'dense':
            return torch.tensor(0.0, device=device, requires_grad=pred.requires_grad)

        # Debug: Check which parameters are missing (only print once)
        if not hasattr(self, '_debug_printed'):
            missing = []
            if coord is None:
                missing.append("coord")
            if opacity is None:
                missing.append("opacity")
            if quat is None:
                missing.append("quat")
            if scale is None:
                missing.append("scale")
            if scene_path is None:
                missing.append("scene_path")
            if missing:
                print(f"\n[Rendered2DLoss] Missing parameters: {missing}")
                print(f"  coord: {coord is not None}")
                print(f"  opacity: {opacity is not None}")
                print(f"  quat: {quat is not None}")
                print(f"  scale: {scale is not None}")
                print(f"  scene_path: {scene_path is not None}")
            else:
                print(f"\n[Rendered2DLoss] All parameters available!")
                # Also print epoch_progress if available
                if epoch_progress is not None:
                    print(f"  epoch_progress: {epoch_progress:.6f}")
                    print(f"  warmup_progress: {self.warmup_progress:.6f}")
                    print(f"  target_progress: {self.target_progress:.6f}")
                else:
                    print(f"  epoch_progress: None")
            self._debug_printed = True

        # Check if all required Gaussian parameters are provided
        if coord is None or opacity is None or quat is None or scale is None or scene_path is None:
            return torch.tensor(0.0, device=device, requires_grad=pred.requires_grad)

        # Compute progressive weight: ramp up from 0 to loss_weight by target_progress
        # DEBUG: Temporarily set weight to loss_weight (1.0) for debugging
        weight = self.loss_weight  # TODO: restore progressive scheduling after debug

        # Original progressive scheduling (disabled for debug)
        # if epoch_progress is not None:
        #     if epoch_progress < self.warmup_progress:
        #         weight = 0.0
        #     elif epoch_progress < self.target_progress:
        #         # Linear ramp from 0 to loss_weight
        #         # Add small epsilon to avoid weight=0 at start when warmup_progress=0
        #         eps = 1e-6
        #         effective_progress = max(epoch_progress, self.warmup_progress + eps)
        #         progress = (effective_progress - self.warmup_progress) / (self.target_progress - self.warmup_progress)
        #         weight = progress * self.loss_weight
        #     else:
        #         # Maintain at maximum weight
        #         weight = self.loss_weight
        # else:
        #     weight = self.loss_weight

        # Print when in warmup phase (only print once)
        if weight == 0 and not hasattr(self, '_warmup_printed'):
            print(f"\n[Rendered2DLoss] In warmup phase (weight=0)")
            if epoch_progress is not None:
                print(f"  epoch_progress: {epoch_progress:.6f}")
            else:
                print(f"  epoch_progress: None")
            print(f"  warmup_progress: {self.warmup_progress:.6f}")
            print(f"  Loss will activate after epoch_progress >= {self.warmup_progress:.3f}")
            self._warmup_printed = True

        if weight == 0:
            return torch.tensor(0.0, device=device, requires_grad=pred.requires_grad)

        # Load GT renders and camera parameters
        gt_renders = self._load_gt_renders(scene_path)
        camera_params = self._load_camera_params(scene_path)

        # Debug: Print loading status (only print once)
        if not hasattr(self, '_load_debug_printed'):
            print(f"\n[Rendered2DLoss] Loading status:")
            if isinstance(scene_path, list):
                print(f"  scene_path (list): {scene_path}")
            else:
                print(f"  scene_path: {scene_path}")
            print(f"  GT renders loaded: {len(gt_renders)}")
            print(f"  Camera params loaded: {'Yes' if camera_params else 'No'}")
            if camera_params:
                print(f"  Camera has 'frames': {'frames' in camera_params}")
                if 'frames' in camera_params:
                    print(f"  Number of frames: {len(camera_params['frames'])}")
            self._load_debug_printed = True

        if len(gt_renders) == 0 or 'frames' not in camera_params:
            return torch.tensor(0.0, device=device, requires_grad=pred.requires_grad)

        # Only consider valid points for rendering
        valid_mask = (valid_feat_mask > 0)
        if valid_mask.sum() < 10:
            return torch.tensor(0.0, device=device, requires_grad=pred.requires_grad)

        # Print info on first activation
        if not hasattr(self, '_printed_init_info'):
            import os
            # Parse scene_path correctly for display
            if isinstance(scene_path, list):
                display_path = scene_path[0] if scene_path else 'unknown'
            else:
                display_path = scene_path

            if 'gaussian_train' in display_path:
                parts = display_path.split('gaussian_train')
                if len(parts) >= 2:
                    path_after = parts[1].lstrip('/')
                    sub_parts = path_after.split('/')
                    if len(sub_parts) >= 3:
                        dataset = sub_parts[0]
                        scene = sub_parts[2] if sub_parts[1] == 'train' else sub_parts[1]
                    else:
                        dataset = 'unknown'
                        scene = 'unknown'
                else:
                    dataset = 'unknown'
                    scene = 'unknown'
            else:
                dataset = 'unknown'
                scene = 'unknown'

            print(f"\n[Rendered2DLoss] Initialized:")
            print(f"  Dataset: {dataset}, Scene: {scene}")
            print(f"  GT renders loaded: {len(gt_renders)}")
            print(f"  Camera params loaded: {len(camera_params.get('frames', []))} frames")
            print(f"  Max num views: {self.max_num_views}")
            print(f"  Warmup progress: {self.warmup_progress:.3f}, Target progress: {self.target_progress:.3f}")
            print(f"  Loss weight: {self.loss_weight}")
            self._printed_init_info = True

        # Print per-iteration info occasionally
        _print_counter = getattr(self, '_print_counter', 0)
        _print_counter += 1
        self._print_counter = _print_counter
        if _print_counter % 100 == 1:  # Print every ~100 iterations
            progress_pct = epoch_progress * 100 if epoch_progress is not None else 0
            print(f"[Rendered2DLoss] progress={progress_pct:.1f}%, weight={weight:.4f}, num_gaussians={valid_mask.sum()}")

        # Filter to valid Gaussians
        means3D = coord[valid_mask]  # [M, 3]
        pred_feat = pred[valid_mask]  # [M, D]
        opacity_filtered = opacity[valid_mask]  # [M, 1]
        quat_filtered = quat[valid_mask]  # [M, 4]
        scale_filtered = scale[valid_mask]  # [M, 3]

        # Get camera intrinsics
        w = camera_params.get('w', 1296)
        h = camera_params.get('h', 968)
        fl_x = camera_params.get('fl_x', 1170.187988)
        fl_y = camera_params.get('fl_y', 1170.187988)
        cx = camera_params.get('cx', w / 2)
        cy = camera_params.get('cy', h / 2)

        # Build camera intrinsic matrix
        K = torch.tensor([
            [fl_x, 0, cx],
            [0, fl_y, cy],
            [0, 0, 1],
        ], device=device, dtype=torch.float32)

        # Limit number of views to save memory
        num_frames = min(len(gt_renders), self.max_num_views)

        # Accumulate loss over all views
        total_loss = 0.0
        valid_views = 0

        for i in range(num_frames):
            gt_render = gt_renders[i]
            frame_name = gt_render['name']

            # Find corresponding frame in camera_params
            frame_idx = None
            for j, frame in enumerate(camera_params['frames']):
                # Extract frame name from file_path
                file_path = frame.get('file_path', '')
                frame_name_from_cam = os.path.basename(file_path)
                if frame_name_from_cam == frame_name or frame_name in file_path:
                    frame_idx = j
                    break

            if frame_idx is None:
                continue

            # Get camera extrinsics
            transform_matrix = np.array(camera_params['frames'][frame_idx]['transform_matrix'])
            viewmat = torch.from_numpy(transform_matrix).float().to(device)  # [4, 4]

            # Prepare Gaussian parameters for gsplat
            means = means3D  # [M, 3]
            quats = quat_filtered  # [M, 4]
            scales = scale_filtered  # [M, 3]
            opacities = opacity_filtered.squeeze(-1)  # [M]
            colors = pred_feat  # [M, D] - features to render

            # Get GT render dimensions
            gt_data = gt_render['data']  # (H_gt, W_gt, D)
            H_gt, W_gt, D_gt = gt_data.shape

            # Render predicted features using gsplat
            try:
                # Use gsplat.rasterization to render features
                render_colors, render_alphas, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,  # [M, D]
                    viewmats=viewmat.unsqueeze(0),  # [1, 4, 4]
                    Ks=K.unsqueeze(0),  # [1, 3, 3]
                    width=W_gt,
                    height=H_gt,
                    packed=False,
                    sh_degree=None,  # Use raw features, not SH
                )

                # render_colors shape: [1, H, W, D] -> [H, W, D]
                rendered_pred = render_colors[0]  # [H, W, D]

                # Get GT render as tensor
                gt_tensor = torch.from_numpy(gt_data).to(device)  # [H, W, D]

                # Normalize GT features to [-1, 1] range (per-pixel normalization)
                # Each pixel's D-dimensional feature vector is normalized independently
                gt_min = gt_tensor.min(dim=-1, keepdim=True).values  # [H, W, 1]
                gt_max = gt_tensor.max(dim=-1, keepdim=True).values  # [H, W, 1]
                gt_range = gt_max - gt_min
                # Raise error if range is zero for any pixel
                if (gt_range <= 0).any():
                    raise ValueError(f"GT feature range is zero or negative for some pixels. Min range: {gt_range.min().item()}")
                # Normalize to [0, 1] then scale to [-1, 1]
                gt_tensor = 2 * (gt_tensor - gt_min) / gt_range - 1

                # Create mask for valid pixels (where alpha > 0)
                valid_pixel_mask = render_alphas[0, ..., 0] > 0.01  # [H, W]

                if valid_pixel_mask.sum() < 100:
                    continue

                # Compute L1 loss on valid pixels only
                pred_valid = rendered_pred[valid_pixel_mask]  # [N_valid, D]
                gt_valid = gt_tensor[valid_pixel_mask]

                view_loss = torch.abs(pred_valid - gt_valid).mean()
                total_loss += view_loss
                valid_views += 1

            except Exception as e:
                print(f"Warning: Rendering failed for {frame_name}: {e}")
                continue

        if valid_views == 0:
            return torch.tensor(0.0, device=device, requires_grad=pred.requires_grad)

        # Average loss over all valid views
        avg_loss = total_loss / valid_views

        # Print loss value (similar to cosine loss)
        final_loss = weight * avg_loss
        print(f"rendered2d loss: {final_loss.item():.6f} (weight={weight:.4f}, views={valid_views})")

        return final_loss


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
