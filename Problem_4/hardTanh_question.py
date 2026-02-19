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

        l.optimize_alpha = False
        l.alpha_param = None

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

        l = preact_lb
        u = preact_ub
        eps = 1e-8

        upper_d = torch.zeros_like(l)
        upper_b = torch.zeros_like(l)
        lower_d = torch.zeros_like(l)
        lower_b = torch.zeros_like(l)

        # Case 1: fully in left saturation (y = -1)
        mask_left = (u <= -1)
        upper_b = torch.where(mask_left, l.new_tensor(-1.0), upper_b)
        lower_b = torch.where(mask_left, l.new_tensor(-1.0), lower_b)

        # Case 2: fully in linear region (y = z)
        mask_linear = (l >= -1) & (u <= 1)
        upper_d = torch.where(mask_linear, l.new_tensor(1.0), upper_d)
        lower_d = torch.where(mask_linear, l.new_tensor(1.0), lower_d)

        # Case 3: fully in right saturation (y = 1)
        mask_right = (l >= 1)
        upper_b = torch.where(mask_right, l.new_tensor(1.0), upper_b)
        lower_b = torch.where(mask_right, l.new_tensor(1.0), lower_b)

        # Case 4: crosses -1 only: l < -1 < u <= 1
        mask_cross_left = (l < -1) & (u > -1) & (u <= 1)
        if mask_cross_left.any():
            denom = (u - l).clamp(min=eps)
            
            s = (u + 1.0) / denom
            b = -1.0 - s * l
            upper_d = torch.where(mask_cross_left, s, upper_d)
            upper_b = torch.where(mask_cross_left, b, upper_b)

            if getattr(self, 'optimize_alpha', False):
                # alpha-CROWN
                if self.alpha_param is None:
                    self.alpha_param = nn.Parameter(torch.zeros_like(l))
                alpha = torch.sigmoid(self.alpha_param)
                lower_d = torch.where(mask_cross_left, alpha, lower_d)
                lower_b = torch.where(mask_cross_left, alpha - 1.0, lower_b)
            else:
                # Vanilla CROWN
                choose = (s > 0.5).float()
                lower_d = torch.where(mask_cross_left, choose, lower_d)
                lower_b = torch.where(mask_cross_left, choose - 1.0, lower_b)

        # Case 5: crosses 1 only: -1 <= l < 1 < u
        mask_cross_right = (l >= -1) & (l < 1) & (u > 1)
        if mask_cross_right.any():
            denom = (u - l).clamp(min=eps)
            
            s = (1.0 - l) / denom
            b = 1.0 - s * u
            lower_d = torch.where(mask_cross_right, s, lower_d)
            lower_b = torch.where(mask_cross_right, b, lower_b)

            if getattr(self, 'optimize_alpha', False):
                # alpha
                if self.alpha_param is None:
                    self.alpha_param = nn.Parameter(torch.zeros_like(l))
                alpha = torch.sigmoid(self.alpha_param)
                upper_d = torch.where(mask_cross_right, alpha, upper_d)
                upper_b = torch.where(mask_cross_right, 1.0 - alpha, upper_b)
            else:
                # CROWN
                choose = (s > 0.5).float()
                upper_d = torch.where(mask_cross_right, choose, upper_d)
                upper_b = torch.where(mask_cross_right, 1.0 - choose, upper_b)

        # Case 6: crosses both -1 and 1: l < -1 and u > 1
        mask_cross_both = (l < -1) & (u > 1)
        if mask_cross_both.any():
            
            s_u = 2.0 / (1.0 - l).clamp(min=eps)
            b_u = 1.0 - s_u
            upper_d = torch.where(mask_cross_both, s_u, upper_d)
            upper_b = torch.where(mask_cross_both, b_u, upper_b)

            s_l = 2.0 / (u + 1.0).clamp(min=eps)
            b_l = s_l - 1.0
            lower_d = torch.where(mask_cross_both, s_l, lower_d)
            lower_b = torch.where(mask_cross_both, b_l, lower_b)

        uA = lA = None
        ubias = lbias = 0

        upper_d = upper_d.unsqueeze(1)
        lower_d = lower_d.unsqueeze(1)

        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0)
            neg_uA = last_uA.clamp(max=0)
            uA = upper_d * pos_uA + lower_d * neg_uA

            mult_pos = pos_uA.view(last_uA.size(0), last_uA.size(1), -1)
            mult_neg = neg_uA.view(last_uA.size(0), last_uA.size(1), -1)
            ubias = mult_pos.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            ubias = ubias + mult_neg.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)

        if last_lA is not None:
            pos_lA = last_lA.clamp(min=0)
            neg_lA = last_lA.clamp(max=0)
            lA = upper_d * neg_lA + lower_d * pos_lA

            mult_pos = pos_lA.view(last_lA.size(0), last_lA.size(1), -1)
            mult_neg = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_pos.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)
            lbias = lbias + mult_neg.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)

        return uA, ubias, lA, lbias

