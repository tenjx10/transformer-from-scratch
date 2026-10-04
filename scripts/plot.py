import json
import os

import matplotlib.pyplot as plt

VAL0_BASE = 9.2423        # base
VAL0_NO_RMSNORM = 16.6762
VAL0_POST_NORM = 9.2420
VAL0_NOPE = 9.2452
VAL0_SILU = 9.2629

def plot(runs, labels, out, metric="val_loss", x="step", ymax=None, ystep=None, val0=None):
    plt.figure()
    for i, (run_dir, label) in enumerate(zip(runs, labels)):
        with open(os.path.join(run_dir, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(run_dir, "log.jsonl")) as f:
            records = [json.loads(line) for line in f]
        records = [r for r in records if metric in r]

        if x == "step":
            xs = [r["step"] for r in records]
        elif x == "time":
            xs = [r["time"] / 60 for r in records]
        else:
            xs = [r["step"] * config["batch_size"] * config["context_length"] for r in records]


        ys = [r[metric] for r in records]
        if val0 is not None:
            xs, ys = [0] + xs, [val0[i]] + ys


        plt.plot(xs, ys, label=label)

    plt.xlabel({"step": "step", "time": "wall-clock time (minutes)", "tokens": "tokens processed"}[x])
    plt.ylabel(metric)

    if ymax is not None:
        plt.ylim(0, ymax)

        if ystep is not None:
            plt.yticks([i * ystep for i in range(int(ymax / ystep) + 1)])

    plt.legend()
    plt.savefig(out)
    plt.close()
    print(f"saved {out}")


def main():
    os.makedirs("plots", exist_ok=True)

    lr_runs = ["experiments/lr_3e-4", "experiments/lr_1e-3", "experiments/lr_3e-3", "experiments/lr_1e-2", "experiments/lr_1e-1", "experiments/lr_1"]
    lr_labels = ["lr=3e-4", "lr=1e-3", "lr=3e-3", "lr=1e-2", "lr=1e-1", "lr=1 (off scale)"]
    plot(lr_runs, lr_labels, "plots/learning_rate_val.png", ymax=10, ystep=0.5, val0=[VAL0_BASE] * 6)
    plot(lr_runs, lr_labels, "plots/learning_rate_val_zoom.png", ymax=5.5, ystep=0.5)
    plot(lr_runs, lr_labels, "plots/learning_rate_train.png", metric="train_loss", ymax=12, ystep=0.5)

    bs_runs = ["experiments/bs_8", "experiments/lr_3e-3", "experiments/bs_64"]
    bs_labels = ["batch 8", "batch 32", "batch 64"]
    plot(bs_runs, bs_labels, "plots/batch_size_val_tokens.png", x="tokens", ymax=10, ystep=0.5, val0=[VAL0_BASE] * 3)
    plot(bs_runs, bs_labels, "plots/batch_size_val_time.png", x="time", ymax=4, ystep=0.5)

    plot(["experiments/lr_3e-3", "experiments/no_rmsnorm_lr_3e-3", "experiments/no_rmsnorm_lr_1e-3"], ["RMSNorm, lr=3e-3", "no RMSNorm, lr=3e-3", "no RMSNorm, lr=1e-3"], "plots/layer_norm_train.png", metric="train_loss", ymax=12)
    plot(["experiments/lr_3e-3", "experiments/lr_1e-3", "experiments/no_rmsnorm_lr_1e-3"], ["RMSNorm, lr=3e-3", "RMSNorm, lr=1e-3", "no RMSNorm, lr=1e-3"], "plots/layer_norm_val.png", ymax=17, ystep=1, val0=[VAL0_BASE, VAL0_BASE, VAL0_NO_RMSNORM])

    norm_runs = ["experiments/lr_3e-3", "experiments/post_norm_lr_3e-3"]
    norm_labels = ["pre-norm, lr=3e-3", "post-norm, lr=3e-3"]
    plot(norm_runs, norm_labels, "plots/pre_norm_val.png", ymax=10, ystep=0.5, val0=[VAL0_BASE, VAL0_POST_NORM])
    plot(norm_runs, norm_labels, "plots/pre_norm_val_zoom.png", ymax=3, ystep=0.25)
    plot(norm_runs, norm_labels, "plots/pre_norm_train.png", metric="train_loss", ymax=10, ystep=0.5)

    pos_runs = ["experiments/lr_3e-3", "experiments/nope_lr_3e-3"]
    pos_labels = ["RoPE, lr=3e-3", "NoPE, lr=3e-3"]
    plot(pos_runs, pos_labels, "plots/nope_val.png", ymax=10, ystep=0.5, val0=[VAL0_BASE, VAL0_NOPE])
    plot(pos_runs, pos_labels, "plots/nope_val_zoom.png", ymax=3.5, ystep=0.25)
    plot(pos_runs, pos_labels, "plots/nope_train.png", metric="train_loss", ymax=10, ystep=0.5)

    ffn_runs = ["experiments/lr_3e-3", "experiments/silu_lr_3e-3"]
    ffn_labels = ["SwiGLU (d_ff=1344)", "SiLU (d_ff=2048)"]
    plot(ffn_runs, ffn_labels, "plots/swiglu_val.png", ymax=10, ystep=0.5, val0=[VAL0_BASE, VAL0_SILU])
    plot(ffn_runs, ffn_labels, "plots/swiglu_val_zoom.png", ymax=3.5, ystep=0.25)
    plot(ffn_runs, ffn_labels, "plots/swiglu_train.png", metric="train_loss", ymax=10, ystep=0.5)


if __name__ == "__main__":
    main()
