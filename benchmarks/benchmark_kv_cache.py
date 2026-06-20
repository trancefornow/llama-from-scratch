import argparse
import os
import statistics
import sys
import time
from typing import Callable, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.model import Llama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark full recompute vs static KV cache decoding.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--prompt-len", "--prompt_len", type=int, default=128)
    parser.add_argument("--new-tokens", "--new_tokens", type=int, default=64)
    parser.add_argument("--vocab-size", "--vocab_size", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2)
    parser.add_argument("--n-heads", "--n_heads", type=int, default=4)
    parser.add_argument("--max-position-embeddings", "--max_position_embeddings", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_model(args: argparse.Namespace, device: torch.device) -> Llama:
    max_seq_len = args.max_position_embeddings or (args.prompt_len + args.new_tokens)
    model = Llama(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_position_embeddings=max_seq_len,
    ).to(device)
    model.eval()
    return model


def decode_full_recompute(
    model: Llama,
    prompt: torch.Tensor,
    new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = prompt.clone()
    next_logits: List[torch.Tensor] = []

    for _ in range(new_tokens):
        logits = model(tokens)
        step_logits = logits[:, -1, :]
        next_logits.append(step_logits)
        next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)

    return tokens, torch.stack(next_logits, dim=1)


def decode_with_cache(
    model: Llama,
    prompt: torch.Tensor,
    new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = prompt.clone()
    model.clear_cache()

    logits = model(prompt, start_pos=0, use_cache=True)
    cur_pos = prompt.size(1)
    next_logits: List[torch.Tensor] = []

    for step in range(new_tokens):
        step_logits = logits[:, -1, :]
        next_logits.append(step_logits)
        next_token = torch.argmax(step_logits, dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)

        if step == new_tokens - 1:
            break

        logits = model(next_token, start_pos=cur_pos, use_cache=True)
        cur_pos += 1

    return tokens, torch.stack(next_logits, dim=1)


def time_call(
    fn: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor]]:
    for _ in range(warmup):
        fn()
    sync_if_needed(device)

    times: List[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        sync_if_needed(device)
        times.append(time.perf_counter() - start)

    assert result is not None
    return statistics.median(times), result


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.prompt_len <= 0 or args.new_tokens <= 0:
        raise ValueError("prompt_len and new_tokens must be positive.")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive.")

    device = resolve_device(args.device)
    model = build_model(args, device)
    prompt = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.prompt_len),
        device=device,
    )
    model.setup_cache(
        max_batch_size=args.batch_size,
        max_seq_len=args.prompt_len + args.new_tokens,
    )

    print("=" * 80)
    print("KV Cache Benchmark")
    print("=" * 80)
    print(
        f"device={device} batch={args.batch_size} prompt_len={args.prompt_len} "
        f"new_tokens={args.new_tokens}"
    )
    print(
        f"model: vocab={args.vocab_size} dim={args.dim} layers={args.n_layers} "
        f"heads={args.n_heads}"
    )

    with torch.inference_mode():
        full_time, full_result = time_call(
            lambda: decode_full_recompute(model, prompt, args.new_tokens),
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        cache_time, cache_result = time_call(
            lambda: decode_with_cache(model, prompt, args.new_tokens),
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )

    full_tokens, full_logits = full_result
    cache_tokens, cache_logits = cache_result
    max_abs_diff = (full_logits - cache_logits).abs().max().item()
    tokens_match = torch.equal(full_tokens, cache_tokens)

    generated = args.batch_size * args.new_tokens
    full_tok_per_sec = generated / full_time
    cache_tok_per_sec = generated / cache_time
    speedup = full_time / cache_time

    print("\nResults")
    print("-" * 80)
    print(f"full recompute : {full_time:.4f}s | {full_tok_per_sec:.2f} tok/s")
    print(f"static KV cache: {cache_time:.4f}s | {cache_tok_per_sec:.2f} tok/s")
    print(f"speedup        : {speedup:.2f}x")
    print(f"max_abs_diff   : {max_abs_diff:.8f}")
    print(f"tokens_match   : {tokens_match}")


if __name__ == "__main__":
    main()
