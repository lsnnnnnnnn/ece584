import torch
import torch.nn as nn


class BoundHardTanh(nn.Hardtanh):
    def __init__(self):
        super(BoundHardTanh, self).__init__()

    @staticmethod
    def convert(act_layer):
        r"""Convert a HardTanh layer to BoundHardTanh layer

        Args:
            act_layer (nn.HardTanh): The HardTanh layer object to be converted.

        Returns:
            l (BoundHardTanh): The converted layer object.
        """
      
        l = BoundHardTanh()
        l.min_val = act_layer.min_val
        l.max_val = act_layer.max_val
        l.inplace = act_layer.inplace
        return l

    def boundpropogate(self, last_uA, last_lA, start_node=None):
        """
        Propagate upper and lower linear bounds through the HardTanh activation function
        based on pre-activation bounds.

        Args:
            last_uA (tensor): A (the coefficient matrix) that is bound-propagated to this layer
            (from the layers after this layer). It's exclusive for computing the upper bound.

            last_lA (tensor): A that is bound-propagated to this layer. It's exclusive for computing the lower bound.

            start_node (int): An integer indicating the start node of this bound propagation

        Returns:
            uA (tensor): The new A for computing the upper bound after taking this layer into account.

            ubias (tensor): The bias (for upper bound) produced by this layer.

            lA( tensor): The new A for computing the lower bound after taking this layer into account.

            lbias (tensor): The bias (for lower bound) produced by this layer.

        """
        # These are preactivation bounds that will be used for form the linear relaxation.
        preact_lb = self.lower_l
        preact_ub = self.upper_u

        eps = 1e-8
        upper_d = torch.zeros_like(preact_lb)
        upper_b = torch.zeros_like(preact_lb)
        lower_d = torch.zeros_like(preact_lb)
        lower_b = torch.zeros_like(preact_lb)

        mask_left = (preact_ub <= -1)
        mask_right = (preact_lb >= 1)
        mask_mid = (preact_lb >= -1) & (preact_ub <= 1)
        mask_cross_left = (preact_lb < -1) & (preact_ub > -1) & (preact_ub <= 1)
        mask_cross_right = (preact_lb >= -1) & (preact_lb < 1) & (preact_ub > 1)
        mask_cross_both = (preact_lb < -1) & (preact_ub > 1)

        # Case 1: Entirely in the left saturation region => y = -1.
        upper_b = torch.where(mask_left, preact_lb.new_tensor(-1.0), upper_b)
        lower_b = torch.where(mask_left, preact_lb.new_tensor(-1.0), lower_b)

        # Case 2: Entirely in the linear region => y = x.
        upper_d = torch.where(mask_mid, preact_lb.new_tensor(1.0), upper_d)
        lower_d = torch.where(mask_mid, preact_lb.new_tensor(1.0), lower_d)

        # Case 3: Entirely in the right saturation region => y = 1.
        upper_b = torch.where(mask_right, preact_lb.new_tensor(1.0), upper_b)
        lower_b = torch.where(mask_right, preact_lb.new_tensor(1.0), lower_b)

        # Case 4: Cross -1 only (preact_lb < -1 < preact_ub <= 1).
        if mask_cross_left.any():
            denom = (preact_ub - preact_lb).clamp(min=eps)
            k = (preact_ub + 1.0) / denom
            b = -1.0 - k * preact_lb
            upper_d = torch.where(mask_cross_left, k, upper_d)
            upper_b = torch.where(mask_cross_left, b, upper_b)

            k_lower = (k > 0.5).float()
            b_lower = k_lower - 1.0
            lower_d = torch.where(mask_cross_left, k_lower, lower_d)
            lower_b = torch.where(mask_cross_left, b_lower, lower_b)

        # Case 5: Cross 1 only (-1 <= preact_lb < 1 < preact_ub).
        if mask_cross_right.any():
            denom = (preact_ub - preact_lb).clamp(min=eps)
            k = (1.0 - preact_lb) / denom
            b = 1.0 - k * preact_ub
            k_upper = (k > 0.5).float()
            b_upper = 1.0 - k_upper
            upper_d = torch.where(mask_cross_right, k_upper, upper_d)
            upper_b = torch.where(mask_cross_right, b_upper, upper_b)

            lower_d = torch.where(mask_cross_right, k, lower_d)
            lower_b = torch.where(mask_cross_right, b, lower_b)

        # Case 6: Cross both -1 and 1 (preact_lb < -1 and preact_ub > 1).
        if mask_cross_both.any():
            k_u = 2.0 / (1.0 - preact_lb).clamp(min=eps)
            b_u = 1.0 - k_u
            upper_d = torch.where(mask_cross_both, k_u, upper_d)
            upper_b = torch.where(mask_cross_both, b_u, upper_b)

            k_l = 2.0 / (preact_ub + 1.0).clamp(min=eps)
            b_l = k_l - 1.0
            lower_d = torch.where(mask_cross_both, k_l, lower_d)
            lower_b = torch.where(mask_cross_both, b_l, lower_b)

        
        upper_d = upper_d.unsqueeze(1)
        lower_d = lower_d.unsqueeze(1)

        uA = lA = None
        ubias = lbias = 0

        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            uA = upper_d * pos_uA + lower_d * neg_uA
           
            mult_pos = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            mult_neg = neg_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_pos.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            ubias = ubias + mult_neg.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)

        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0)
            lA = upper_d * neg_lA + lower_d * pos_lA
            
            mult_neg = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            mult_pos = pos_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_neg.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            lbias = lbias + mult_pos.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)

        return uA, ubias, lA, lbias

