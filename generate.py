import argparse
import time
from typing import Any, Dict, List, Tuple

import torch

from model.model import Llama
from model.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with the tiny LLaMA model.")
    parser.add_argument("--prompt", type=str, default="Far over")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", "--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint", "--ckpt", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2)
    parser.add_argument("--n-heads", "--n_heads", type=int, default=4)
    parser.add_argument("--max-position-embeddings", "--max_position_embeddings", type=int, default=2048)
    parser.add_argument("--use-cache", "--use_cache", dest="use_cache", action="store_true", default=True)
    parser.add_argument("--no-cache", "--no_cache", dest="use_cache", action="store_false")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def load_checkpoint(path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dict or a raw state_dict.")

    config = checkpoint.get("config", checkpoint.get("model_args", {}))
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint.get("model", checkpoint)),
    )
    if not isinstance(state_dict, dict):
        raise ValueError("Could not find a valid state_dict in checkpoint.")

    return dict(config), state_dict


def build_model(args: argparse.Namespace, tokenizer: BPETokenizer, device: torch.device) -> Llama:
    config = {
        "vocab_size": tokenizer.vocab_size,
        "dim": args.dim,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_position_embeddings": args.max_position_embeddings,
    }
    state_dict = None

    if args.checkpoint:
        checkpoint_config, state_dict = load_checkpoint(args.checkpoint, device)
        config.update({k: v for k, v in checkpoint_config.items() if k in config})
    else:
        print("[warn] No checkpoint provided; using randomly initialized weights.")

    model = Llama(**config).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(f"[info] Loaded checkpoint: {args.checkpoint}")

    model.eval()
    return model


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def ensure_context_fits(model: Llama, prompt_len: int, max_new_tokens: int) -> None:
    total_len = prompt_len + max_new_tokens
    if prompt_len == 0:
        raise ValueError("Prompt must contain at least one token.")
    if total_len > model.max_position_embeddings:
        raise ValueError(
            f"prompt_len + max_new_tokens = {total_len}, but model only supports "
            f"{model.max_position_embeddings} positions."
        )


def generate_full_recompute(
    model: Llama,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    tokens = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_token = sample_next_token(logits[:, -1, :], temperature, top_k)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


def generate_with_cache(
    model: Llama,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    tokens = input_ids.clone()
    model.setup_cache(max_batch_size=tokens.size(0), max_seq_len=tokens.size(1) + max_new_tokens)
    model.clear_cache()

    logits = model(tokens, start_pos=0, use_cache=True)
    cur_pos = tokens.size(1)

    for step in range(max_new_tokens):
        next_token = sample_next_token(logits[:, -1, :], temperature, top_k)
        tokens = torch.cat((tokens, next_token), dim=1)
        if step == max_new_tokens - 1:
            break

        logits = model(next_token, start_pos=cur_pos, use_cache=True)
        cur_pos += 1

    return tokens


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = BPETokenizer(args.tokenizer)
    input_ids: List[int] = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    model = build_model(args, tokenizer, device)
    ensure_context_fits(model, prompt_len=input_tensor.size(1), max_new_tokens=args.max_new_tokens)

    print(f"[info] device={device}, prompt_tokens={input_tensor.size(1)}, use_cache={args.use_cache}")

    start_time = time.perf_counter()
    with torch.inference_mode():
        if args.use_cache:
            output_ids = generate_with_cache(
                model,
                input_tensor,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
            )
        else:
            output_ids = generate_full_recompute(
                model,
                input_tensor,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
            )
    elapsed = time.perf_counter() - start_time

    generated_ids = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    new_tokens = max(0, len(generated_ids) - len(input_ids))
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else float("inf")

    print("\n" + "=" * 80)
    print(generated_text)
    print("=" * 80)
    print(f"[info] generated_tokens={new_tokens}, elapsed={elapsed:.3f}s, tok/s={tokens_per_sec:.2f}")


if __name__ == "__main__":
    main()
