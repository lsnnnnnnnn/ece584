import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SimpleNNHardTanh
from crown import BoundedSequential
from hardTanh_question import BoundHardTanh


def set_optimize_alpha(bounded_model, enabled: bool):
    for m in bounded_model.modules():
        if isinstance(m, BoundHardTanh):
            m.optimize_alpha = enabled


def compute_bounds(bounded_model, x_u, x_l):
    ub, lb = bounded_model.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True)
    return ub.detach().cpu().squeeze(0).numpy(), lb.detach().cpu().squeeze(0).numpy()


def optimize_alpha(bounded_model, x_u, x_l, iters=200, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    set_optimize_alpha(bounded_model, True)

    for p in bounded_model.parameters():
        p.requires_grad_(False)

    _ = bounded_model.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True)

    alpha_params = []
    for m in bounded_model.modules():
        if isinstance(m, BoundHardTanh) and m.alpha_param is not None:
            m.alpha_param.requires_grad_(True)
            alpha_params.append(m.alpha_param)

    opt = torch.optim.Adam(alpha_params, lr=lr)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        ub, lb = bounded_model.compute_bounds(x_U=x_u, x_L=x_l, upper=True, lower=True)
        loss = (ub - lb).sum()
        loss.backward()
        opt.step()


def plot_overlay(lb0, ub0, lb1, ub1, outpath, title):
    n = len(lb0)
    xs = np.arange(n)
    plt.figure(figsize=(10,4))
    for i in range(n):
        # vanilla
        plt.plot([xs[i]-0.1, xs[i]-0.1], [lb0[i], ub0[i]], linewidth=6, alpha=0.25)
        # alpha
        plt.plot([xs[i]+0.1, xs[i]+0.1], [lb1[i], ub1[i]], linewidth=6, alpha=0.25)
    plt.xticks(xs)
    plt.xlabel('logit index')
    plt.ylabel('value')
    plt.title(title + ' (vanilla left, alpha right)')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_widths(lb0, ub0, lb1, ub1, outpath, title):
    w0 = ub0 - lb0
    w1 = ub1 - lb1
    xs = np.arange(len(w0))
    plt.figure(figsize=(10,3.5))
    plt.bar(xs-0.2, w0, width=0.4, label='vanilla')
    plt.bar(xs+0.2, w1, width=0.4, label='alpha')
    plt.xticks(xs)
    plt.xlabel('logit index')
    plt.ylabel('width (ub-lb)')
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prefix', type=str, default='alpha')
    args = parser.parse_args()

    x_test, label = torch.load(args.data_file)
    model = SimpleNNHardTanh()
    model.load_state_dict(torch.load('models/hardtanh_model.pth'))
    model.eval()

    x_test = x_test.reshape(x_test.size(0), -1)
    x_u = x_test + args.eps
    x_l = x_test - args.eps

    bounded = BoundedSequential.convert(model)

    set_optimize_alpha(bounded, False)
    ub0, lb0 = compute_bounds(bounded, x_u, x_l)

    optimize_alpha(bounded, x_u, x_l, iters=args.iters, lr=args.lr, seed=args.seed)
    ub1, lb1 = compute_bounds(bounded, x_u, x_l)

    plot_overlay(lb0, ub0, lb1, ub1, f'{args.prefix}_overlay.png', f'{args.prefix}: bounds overlay')
    plot_widths(lb0, ub0, lb1, ub1, f'{args.prefix}_widths.png', f'{args.prefix}: widths comparison')

    print('Saved:', f'{args.prefix}_overlay.png', 'and', f'{args.prefix}_widths.png')
    print('Total width vanilla:', float((ub0-lb0).sum()))
    print('Total width alpha  :', float((ub1-lb1).sum()))


if __name__ == '__main__':
    main()