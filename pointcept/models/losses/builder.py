"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None, verbose_losses=False, return_per_dim=False):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        self.loss_names = []
        self.verbose_losses = verbose_losses  # Enable individual loss logging
        self.return_per_dim = return_per_dim  # Enable per-dimension loss tracking for SVDWeightedL1Loss
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))
            # Store loss name from config type
            loss_type = loss_cfg.get("type", "unknown")
            self.loss_names.append(loss_type)

    def __call__(self, pred, target, **kwargs):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        if self.verbose_losses:
            # Print individual losses and collect per-dimension losses
            import sys
            loss_values = {}
            per_dim_losses = {}  # Store per-dimension losses from SVDWeightedL1Loss
            per_dim_weights = {}  # Store per-dimension weights from SVDWeightedL1Loss

            for c, loss_name in zip(self.criteria, self.loss_names):
                # Call loss with return_per_dim flag for SVDWeightedL1Loss
                if loss_name == "SVDWeightedL1Loss" and self.return_per_dim:
                    loss_value, loss_dict = c(pred, target, return_per_dim=True, **kwargs)
                    loss += loss_value

                    # Extract per-dimension losses and weights
                    if 'per_dim_losses' in loss_dict:
                        per_dim_losses['per_dim_l1'] = loss_dict['per_dim_losses']
                    if 'per_dim_weights' in loss_dict:
                        per_dim_weights['per_dim_l1_weights'] = loss_dict['per_dim_weights']
                else:
                    loss_value = c(pred, target, **kwargs)
                    loss += loss_value

                # Normalize loss name
                name = loss_name.lower()
                if name == "cosinesimilarity":
                    name = "cos_loss"
                elif name == "l2loss":
                    name = "l2_loss"
                elif name == "smoothceloss":
                    name = "ce_loss"
                elif name == "crossentropyloss":
                    name = "ce_loss"
                elif name == "aggregatedcontrastiveloss":
                    name = "contrast_loss"
                elif name == "l1loss":
                    name = "l1_loss"
                elif name == "svdweightedl1loss":
                    name = "l1_loss"
                elif name == "rendered2dloss":
                    name = "rendered2d_loss"

                # Always call detach() before item() to avoid issues with non-leaf tensors
                loss_values[name] = loss_value.detach().item()

            # Return total loss, individual loss values, and per-dimension losses
            return loss, loss_values, per_dim_losses, per_dim_weights
        else:
            for c in self.criteria:
                loss += c(pred, target, **kwargs)
            return loss


def build_criteria(cfg, verbose_losses=False, return_per_dim=False):
    return Criteria(cfg, verbose_losses=verbose_losses, return_per_dim=return_per_dim)
