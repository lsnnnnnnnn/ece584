import argparse
import time
import torch
import torch.nn as nn

from model import SimpleNNHardTanh
from crown import BoundedSequential
from hardTanh_question import BoundHardTanh


def set_optimize_alpha(bounded_model: nn.Module, enabled: bool = True) -> int:

    cnt = 0
    for m in bounded_model.modules():
        if isinstance(m, BoundHardTanh):
            m.optimize_alpha = enabled
            cnt += 1
    return cnt


def compute_bounds(bounded_model: BoundedSequential, x_u: torch.Tensor, x_l: torch.Tensor):
    
    ub, lb = bounded_model.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True)
    return ub, lb


def ensure_alpha_params_exist(bounded_model: nn.Module) -> list:

    alpha_params = []
    for m in bounded_model.modules():
        if not isinstance(m, BoundHardTanh):
            continue
        # pre-activation bounds are stored after a vanilla bound pass
        l = getattr(m, "lower_l", None)
        if l is None:
            continue
        if m.alpha_param is None:
            m.alpha_param = nn.Parameter(torch.zeros_like(l))
        alpha_params.append(m.alpha_param)
    return alpha_params


def init_alpha_from_vanilla_heuristic(bounded_model: nn.Module, eps: float = 1e-4) -> int:
 
    cnt = 0
    for m in bounded_model.modules():
        if not isinstance(m, BoundHardTanh):
            continue
        if m.alpha_param is None:
            continue

        l = m.lower_l.detach()
        u = m.upper_u.detach()

        alpha = torch.full_like(l, 0.5)

        # Case 4: l < -1 < u <= 1
        mask_cross_left = (l < -1) & (u > -1) & (u <= 1)
        if mask_cross_left.any():
            denom = (u - l).clamp(min=1e-8)
            s_upper = (u + 1.0) / denom
            choose = (s_upper > 0.5).float()  # 1 -> identity, 0 -> constant -1
            alpha = torch.where(mask_cross_left, choose, alpha)

        # Case 5: -1 <= l < 1 < u
        mask_cross_right = (l >= -1) & (l < 1) & (u > 1)
        if mask_cross_right.any():
            denom = (u - l).clamp(min=1e-8)
            s_lower = (1.0 - l) / denom
            choose = (s_lower > 0.5).float()  # 1 -> identity, 0 -> constant 1
            alpha = torch.where(mask_cross_right, choose, alpha)

        alpha = alpha.clamp(min=eps, max=1.0 - eps)
        theta = torch.log(alpha / (1.0 - alpha))

        with torch.no_grad():
            m.alpha_param.copy_(theta)
        cnt += 1

    return cnt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str, help="input data (.pth)")
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--no_init_from_vanilla", action="store_true",
                        help="disable alpha initialization from vanilla heuristic")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    x_test, label = torch.load(args.data_file)
    model = SimpleNNHardTanh()
    model.load_state_dict(torch.load('models/hardtanh_model.pth'))
    model.eval()

    batch_size = x_test.size(0)
    x_test = x_test.reshape(batch_size, -1)

    x_u = x_test + args.eps
    x_l = x_test - args.eps

    bounded = BoundedSequential.convert(model)

    set_optimize_alpha(bounded, enabled=False)
    with torch.no_grad():
        ub0, lb0 = compute_bounds(bounded, x_u, x_l)
        width0 = (ub0 - lb0).sum().item()
    print(f"vanilla CROWN total width: {width0:.6f}")

    set_optimize_alpha(bounded, enabled=True)

    for p in bounded.parameters():
        p.requires_grad_(False)

    alpha_params = ensure_alpha_params_exist(bounded)
    for p in alpha_params:
        p.requires_grad_(True)

    if len(alpha_params) == 0:
        raise RuntimeError("No alpha parameters found/created.")

    if not args.no_init_from_vanilla:
        n_init = init_alpha_from_vanilla_heuristic(bounded)
        with torch.no_grad():
            ub_init, lb_init = compute_bounds(bounded, x_u, x_l)
            width_init = (ub_init - lb_init).sum().item()
        print(f"alpha-CROWN init-from-vanilla: initialized {n_init} layers, total width: {width_init:.6f}")

    print(f"alpha-CROWN: optimizing {len(alpha_params)} alpha tensors with Adam (iters={args.iters}, lr={args.lr})")
    optimizer = torch.optim.Adam(alpha_params, lr=args.lr)

    start = time.time()
    best = None
    best_state = None

    for t in range(1, args.iters + 1):
        optimizer.zero_grad(set_to_none=True)
        ub, lb = compute_bounds(bounded, x_u, x_l)
        loss = (ub - lb).sum()
        loss.backward()
        optimizer.step()

        total_width = loss.item()
        if best is None or total_width < best:
            best = total_width
            best_state = [p.detach().clone() for p in alpha_params]

        if (t % args.print_every) == 0 or t == 1 or t == args.iters:
            print(f"iter {t:4d} | total width = {total_width:.6f} | best = {best:.6f}")


    if best_state is not None:
        with torch.no_grad():
            for p, s in zip(alpha_params, best_state):
                p.copy_(s)

    elapsed = time.time() - start

    with torch.no_grad():
        ub1, lb1 = compute_bounds(bounded, x_u, x_l)
        width1 = (ub1 - lb1).sum().item()

    print("\n=== Summary ===")
    print(f"eps = {args.eps}")
    print(f"vanilla CROWN total width: {width0:.6f}")
    print(f"alpha-CROWN  total width: {width1:.6f}")
    if width0 > 0:
        print(f"relative improvement: {(width0 - width1) / width0 * 100.0:.2f}%")
    print(f"optimization time: {elapsed:.2f}s")

    y_size = ub1.size(1)
    for i in range(batch_size):
        for j in range(y_size):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb1[i][j].item(), u=ub1[i][j].item()))


if __name__ == "__main__":
    main()