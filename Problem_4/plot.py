import os, torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import SimpleNNHardTanh
from crown import BoundedSequential

torch.set_num_threads(1)

def sample_min_max(model, x0, eps, n_samples=3000, batch=256, seed=0):
    torch.manual_seed(seed)
    mins = None
    maxs = None
    with torch.no_grad():
        for start in range(0, n_samples, batch):
            bs = min(batch, n_samples - start)
            delta = (torch.rand(bs, x0.numel()) * 2 - 1) * eps
            x = x0.view(1, -1) + delta
            out = model(x)
            bmin = out.min(dim=0).values
            bmax = out.max(dim=0).values
            mins = bmin if mins is None else torch.minimum(mins, bmin)
            maxs = bmax if maxs is None else torch.maximum(maxs, bmax)
    return mins, maxs

def plot_intervals(df, title, outpath):
    xs = df["logit"].values
    lb = df["lb_crown"].values
    ub = df["ub_crown"].values
    smin = df["sample_min"].values
    smax = df["sample_max"].values
    nom = df["nominal"].values

    plt.figure(figsize=(10, 4))
    for i in range(len(xs)):
        plt.plot([xs[i], xs[i]], [lb[i], ub[i]], linewidth=6, alpha=0.25)   # CROWN interval
        plt.plot([xs[i], xs[i]], [smin[i], smax[i]], linewidth=2, alpha=0.9) # sampled range
        plt.scatter([xs[i]], [nom[i]], s=25)                                  # nominal
    plt.xticks(xs)
    plt.xlabel("logit index")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_widths(df, title, outpath):
    plt.figure(figsize=(10, 3.5))
    plt.bar(df["logit"].values, df["width_crown"].values)
    plt.xticks(df["logit"].values)
    plt.xlabel("logit index")
    plt.ylabel("CROWN width (ub-lb)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def run_one(data_path, out_prefix, eps=0.01, n_samples=3000, seed=0):
    x_test, label = torch.load(data_path)
    x = x_test.reshape(1, -1)
    xU = x + eps
    xL = x - eps

    model = SimpleNNHardTanh()
    model.load_state_dict(torch.load("models/hardtanh_model.pth"))
    model.eval()

    with torch.no_grad():
        nominal = model(x).squeeze(0)

    bounded = BoundedSequential.convert(model)
    ub, lb = bounded.compute_bounds(x_U=xU, x_L=xL, upper=True, lower=True)
    ub = ub.detach().squeeze(0)
    lb = lb.detach().squeeze(0)

    smin, smax = sample_min_max(model, x, eps, n_samples=n_samples, seed=seed)

    df = pd.DataFrame({
        "logit": np.arange(10),
        "lb_crown": lb.numpy(),
        "ub_crown": ub.numpy(),
        "width_crown": (ub - lb).numpy(),
        "nominal": nominal.numpy(),
        "sample_min": smin.numpy(),
        "sample_max": smax.numpy(),
    })
    df["sample_within_crown?"] = (df["sample_min"] >= df["lb_crown"] - 1e-6) & (df["sample_max"] <= df["ub_crown"] + 1e-6)

    df.to_csv(f"{out_prefix}_bounds_table.csv", index=False)
    plot_intervals(df, f"{out_prefix}: CROWN interval (thick) + sampled range (thin) + nominal (dot)",
                   f"{out_prefix}_bounds.png")
    plot_widths(df, f"{out_prefix}: CROWN bound widths", f"{out_prefix}_widths.png")

if __name__ == "__main__":
    run_one("data1.pth", "data1", eps=0.01, n_samples=8000, seed=123)
    run_one("data2.pth", "data2", eps=0.01, n_samples=3000, seed=456)
