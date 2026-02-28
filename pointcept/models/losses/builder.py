"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None, verbose_losses=False):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        self.loss_names = []
        self.verbose_losses = verbose_losses  # Enable individual loss logging
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
            # Print individual losses
            import sys
            loss_values = {}
            for c, loss_name in zip(self.criteria, self.loss_names):
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
                loss_values[name] = loss_value.item()
            # Print to stdout without buffering - include contrastive loss if present
            # NOTE: For normalized features (||pred|| = ||target|| = 1):
            #   L2_loss = 2 * Cosine_loss, so with weights 0.5 and 1.0, they are equal
            l2_val = loss_values.get('l2_loss', 0)
            cos_val = loss_values.get('cos_loss', 0)
            contrast_val = loss_values.get('contrast_loss', None)

            if contrast_val is not None:
                print(f"L2: {l2_val:.6f}, Cos: {cos_val:.6f}, Contrast: {contrast_val:.6f}", flush=True, file=sys.stderr)
            else:
                print(f"L2: {l2_val:.6f}, Cos: {cos_val:.6f}", flush=True, file=sys.stderr)
        else:
            for c in self.criteria:
                loss += c(pred, target, **kwargs)
        return loss


def build_criteria(cfg, verbose_losses=False):
    return Criteria(cfg, verbose_losses=verbose_losses)
