import argparse
import csv
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from model.model import Llama
from model.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tiny LLaMA model and export experiment logs.")
    parser.add_argument("--data-path", "--data_path", type=str, default="data/input.txt")
    parser.add_argument("--run-root", "--run_root", type=str, default="runs")
    parser.add_argument("--report-dir", "--report_dir", type=str, default="reports")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=8)
    parser.add_argument("--block-size", "--block_size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2)
    parser.add_argument("--n-heads", "--n_heads", type=int, default=4)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=1000)
    parser.add_argument("--eval-interval", "--eval_interval", type=int, default=50)
    parser.add_argument("--eval-iters", "--eval_iters", type=int, default=20)
    parser.add_argument("--smooth-window", "--smooth_window", type=int, default=5)
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", "--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-checkpoint", "--save_checkpoint", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def load_data(path: Path, tokenizer: BPETokenizer) -> torch.Tensor:
    print(f"[data] loading {path}")
    text = path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    print(f"[data] characters={len(text):,}, tokens={len(token_ids):,}")
    return torch.tensor(token_ids, dtype=torch.long)


def split_data(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    split = int(0.9 * len(data))
    return data[:split], data[split:]


def validate_data(train_data: torch.Tensor, val_data: torch.Tensor, block_size: int) -> None:
    min_len = block_size + 1
    if len(train_data) < min_len:
        raise ValueError(f"train split is too short for block_size={block_size}")
    if len(val_data) < min_len:
        raise ValueError(f"validation split is too short for block_size={block_size}")


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: Llama,
    criterion: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses: Dict[str, List[float]] = {"train": [], "val": []}
    for split, data in (("train", train_data), ("val", val_data)):
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            logits = model(x)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            losses[split].append(loss.item())
    model.train()
    return sum(losses["train"]) / len(losses["train"]), sum(losses["val"]) / len(losses["val"])


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def make_run_dir(run_root: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_config(path: Path, args: argparse.Namespace, vocab_size: int, device: torch.device) -> None:
    config = vars(args).copy()
    config["vocab_size"] = vocab_size
    config["device_resolved"] = str(device)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def append_metrics(path: Path, row: Dict[str, float]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path: Path, model: Llama, args: argparse.Namespace) -> None:
    checkpoint = {
        "config": {
            "vocab_size": model.vocab_size,
            "dim": args.dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "max_position_embeddings": args.block_size,
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)


def points_to_polyline(points: List[Tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def smooth_values(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values

    radius = window // 2
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - radius)
        end = min(len(values), i + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def write_loss_curve_svg(path: Path, metrics: List[Dict[str, float]], smooth_window: int) -> None:
    if len(metrics) < 2:
        return

    width, height = 900, 520
    left, right, top, bottom = 72, 28, 36, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    steps = [row["step"] for row in metrics]
    train_losses = smooth_values([row["train_loss"] for row in metrics], smooth_window)
    val_losses = smooth_values([row["val_loss"] for row in metrics], smooth_window)
    min_step, max_step = min(steps), max(steps)
    min_loss = min(train_losses + val_losses)
    max_loss = max(train_losses + val_losses)
    if math.isclose(min_loss, max_loss):
        min_loss -= 0.5
        max_loss += 0.5

    def x_scale(step: float) -> float:
        if math.isclose(min_step, max_step):
            return left + plot_w / 2
        return left + (step - min_step) / (max_step - min_step) * plot_w

    def y_scale(loss: float) -> float:
        return top + (max_loss - loss) / (max_loss - min_loss) * plot_h

    train_points = [(x_scale(step), y_scale(loss)) for step, loss in zip(steps, train_losses)]
    val_points = [(x_scale(step), y_scale(loss)) for step, loss in zip(steps, val_losses)]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width / 2}" y="24" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">Smoothed Training and Validation Loss</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>
  <text x="{left + plot_w / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">step</text>
  <text x="20" y="{top + plot_h / 2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 20 {top + plot_h / 2})">loss</text>
  <text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Arial" font-size="12">{max_loss:.3f}</text>
  <text x="{left - 8}" y="{top + plot_h + 4}" text-anchor="end" font-family="Arial" font-size="12">{min_loss:.3f}</text>
  <text x="{left}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{min_step:.0f}</text>
  <text x="{left + plot_w}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{max_step:.0f}</text>
  <polyline fill="none" stroke="#2563eb" stroke-width="3" points="{points_to_polyline(train_points)}"/>
  <polyline fill="none" stroke="#dc2626" stroke-width="3" points="{points_to_polyline(val_points)}"/>
  <rect x="{width - 190}" y="48" width="145" height="58" fill="#fff" stroke="#ddd"/>
  <line x1="{width - 176}" y1="68" x2="{width - 138}" y2="68" stroke="#2563eb" stroke-width="3"/>
  <text x="{width - 128}" y="72" font-family="Arial" font-size="13">smooth train</text>
  <line x1="{width - 176}" y1="90" x2="{width - 138}" y2="90" stroke="#dc2626" stroke-width="3"/>
  <text x="{width - 128}" y="94" font-family="Arial" font-size="13">smooth val</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_report(
    path: Path,
    run_dir: Path,
    args: argparse.Namespace,
    metrics: List[Dict[str, float]],
    curve_name: str,
) -> None:
    first = metrics[0]
    last = metrics[-1]
    train_delta = first["train_loss"] - last["train_loss"]
    val_delta = first["val_loss"] - last["val_loss"]
    text = f"""# LLaMA From Scratch 实验报告

## 实验概览

本次实验使用当前项目中的 LLaMA 风格 Decoder-only Transformer，在 `data/input.txt` 上进行小规模语言模型训练，并自动记录训练日志、验证集损失与损失曲线。

## 训练配置

| 项目 | 数值 |
|---|---:|
| batch size | {args.batch_size} |
| block size | {args.block_size} |
| dim | {args.dim} |
| layers | {args.n_layers} |
| heads | {args.n_heads} |
| max steps | {args.max_steps} |
| eval interval | {args.eval_interval} |
| eval iters | {args.eval_iters} |
| smooth window | {args.smooth_window} |
| learning rate | {args.learning_rate} |
| weight decay | {args.weight_decay} |

## 结果摘要

| 指标 | 初始 | 最终 | 下降量 |
|---|---:|---:|---:|
| train loss | {first["train_loss"]:.4f} | {last["train_loss"]:.4f} | {train_delta:+.4f} |
| val loss | {first["val_loss"]:.4f} | {last["val_loss"]:.4f} | {val_delta:+.4f} |
| train perplexity | {first["train_ppl"]:.2f} | {last["train_ppl"]:.2f} | - |
| val perplexity | {first["val_ppl"]:.2f} | {last["val_ppl"]:.2f} | - |

![Smoothed Loss Curve]({curve_name})

损失曲线仅绘制平滑后的 loss，平滑窗口为 `{args.smooth_window}` 个评估点。

## 产物位置

- 运行目录：`{run_dir.as_posix()}`
- 训练日志：`metrics.csv`
- 损失曲线：`loss_curve.svg`
- 配置快照：`config.json`

## 结论

训练损失从 `{first["train_loss"]:.4f}` 下降到 `{last["train_loss"]:.4f}`，说明当前模型、数据管道与反向传播链路能够正常工作。

验证损失从 `{first["val_loss"]:.4f}` 变化到 `{last["val_loss"]:.4f}`。由于本实验是小规模快速训练，结果主要用于验证工程闭环，而不是追求最终语言建模效果。

下一步更有价值的实验包括：

- 使用更长训练步数观察稳定收敛趋势。
- 保存 checkpoint 后用 `generate.py --checkpoint` 评估生成质量。
- 对比开启与关闭 KV Cache 的生成速度。
- 加入更系统的超参数消融实验。
"""
    path.write_text(text, encoding="utf-8")


def mirror_report_artifacts(run_dir: Path, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "metrics.csv", report_dir / "train_metrics.csv")
    shutil.copyfile(run_dir / "loss_curve.svg", report_dir / "loss_curve.svg")
    shutil.copyfile(run_dir / "report.md", report_dir / "experiment_report.md")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = BPETokenizer(args.tokenizer)
    data = load_data(Path(args.data_path), tokenizer)
    train_data, val_data = split_data(data)
    validate_data(train_data, val_data, args.block_size)

    run_dir = make_run_dir(Path(args.run_root))
    metrics_path = run_dir / "metrics.csv"
    curve_path = run_dir / "loss_curve.svg"
    report_path = run_dir / "report.md"
    write_config(run_dir / "config.json", args, tokenizer.vocab_size, device)

    model = Llama(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_position_embeddings=args.block_size,
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"[run] {run_dir}")
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}, device={device}")

    metrics: List[Dict[str, float]] = []
    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_step = 0

    for step in range(args.max_steps + 1):
        should_log = step % args.eval_interval == 0 or step == args.max_steps
        if should_log:
            train_loss, val_loss = estimate_loss(model, criterion, train_data, val_data, args, device)
            now = time.perf_counter()
            elapsed = now - start_time
            interval = max(now - last_log_time, 1e-9)
            steps_since_log = step - last_log_step
            tokens_since_log = args.batch_size * args.block_size * steps_since_log
            tokens_per_sec = tokens_since_log / interval if step > 0 else 0.0
            last_log_time = now
            last_log_step = step

            row = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": perplexity(train_loss),
                "val_ppl": perplexity(val_loss),
                "elapsed_sec": elapsed,
                "tokens_per_sec": tokens_per_sec,
            }
            metrics.append(row)
            append_metrics(metrics_path, row)
            write_loss_curve_svg(curve_path, metrics, args.smooth_window)
            print(
                f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                f"train ppl {row['train_ppl']:.2f} | val ppl {row['val_ppl']:.2f}"
            )

        if step == args.max_steps:
            break

        model.train()
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        logits = model(x)
        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    if args.save_checkpoint:
        save_checkpoint(run_dir / "checkpoint_last.pt", model, args)

    write_report(report_path, run_dir, args, metrics, "loss_curve.svg")
    mirror_report_artifacts(run_dir, Path(args.report_dir))
    print(f"[done] metrics: {metrics_path}")
    print(f"[done] curve:   {curve_path}")
    print(f"[done] report:  {report_path}")
    print(f"[done] latest report mirror: {Path(args.report_dir) / 'experiment_report.md'}")


if __name__ == "__main__":
    main()
