import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.model import Llama


def _build_model():
    torch.manual_seed(1234)
    model = Llama(
        vocab_size=97,
        dim=64,
        n_layers=2,
        n_heads=4,
        max_position_embeddings=64,
    )
    model.eval()
    return model


def _assert_close(name, full_logits, cached_logits):
    max_abs = (full_logits - cached_logits).abs().max().item()
    assert torch.allclose(full_logits, cached_logits, atol=1e-5, rtol=1e-5), (
        f"{name} cache path diverged from full recompute; max_abs={max_abs:.8f}"
    )
    print(f"[OK] {name}: cache logits match full recompute, max_abs={max_abs:.8f}")


def test_token_by_token_cache_matches_full_forward():
    batch_size = 2
    seq_len = 24
    model = _build_model()
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        full_logits = model(tokens)

        model.setup_cache(max_batch_size=batch_size, max_seq_len=seq_len)
        model.clear_cache()

        cached_pieces = []
        for pos in range(seq_len):
            step_tokens = tokens[:, pos : pos + 1]
            step_logits = model(step_tokens, start_pos=pos, use_cache=True)
            cached_pieces.append(step_logits)

        cached_logits = torch.cat(cached_pieces, dim=1)

    _assert_close("token-by-token", full_logits, cached_logits)


def test_chunked_cache_matches_full_forward():
    batch_size = 2
    seq_len = 24
    chunk_size = 7
    model = _build_model()
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        full_logits = model(tokens)

        model.setup_cache(max_batch_size=batch_size, max_seq_len=seq_len)
        model.clear_cache()

        cached_pieces = []
        for start_pos in range(0, seq_len, chunk_size):
            end_pos = min(start_pos + chunk_size, seq_len)
            chunk_tokens = tokens[:, start_pos:end_pos]
            chunk_logits = model(chunk_tokens, start_pos=start_pos, use_cache=True)
            cached_pieces.append(chunk_logits)

        cached_logits = torch.cat(cached_pieces, dim=1)

    _assert_close("chunked prefill", full_logits, cached_logits)


def test_cache_is_static_and_preallocated():
    batch_size = 2
    seq_len = 24
    model = _build_model()
    tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len))

    model.setup_cache(max_batch_size=batch_size, max_seq_len=seq_len)
    cache_ids = [
        (id(layer.attention.cache_k), id(layer.attention.cache_v))
        for layer in model.layers
    ]

    with torch.no_grad():
        for pos in range(seq_len):
            model(tokens[:, pos : pos + 1], start_pos=pos, use_cache=True)

    for layer_id, layer in enumerate(model.layers):
        attention = layer.attention
        assert attention.cache_k.shape == (
            batch_size,
            attention.n_heads,
            seq_len,
            attention.head_dim,
        )
        assert attention.cache_v.shape == attention.cache_k.shape
        assert cache_ids[layer_id] == (
            id(attention.cache_k),
            id(attention.cache_v),
        ), "cache tensor was reallocated during decoding"

    print("[OK] cache tensors are preallocated and reused in-place")


if __name__ == "__main__":
    print("=" * 60)
    print("Verifying static start_pos KV cache")
    print("=" * 60)
    test_token_by_token_cache_matches_full_forward()
    test_chunked_cache_matches_full_forward()
    test_cache_is_static_and_preallocated()
    print("\nAll static KV cache checks passed.")
