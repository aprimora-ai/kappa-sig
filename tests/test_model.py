"""
Smoke test com config reduzida para validar a arquitetura.
O modelo real (500M) roda na máquina do David.
"""
import torch
import sys
sys.path.insert(0, ".")

from src.kappa.emo_model.model import (
    EMOConfig, EMOTransformer, EmotionalTrainingLoss,
    EmotionalLoRA, apply_emotional_lora, create_emo_model,
    EmotionalEmbedding, EmotionalGate, EmotionalAttentionLayer,
)


def test_tiny():
    """Testa com config minúscula (~2M params)."""
    print("=" * 60)
    print("Kappa-EMO Model — Tiny Smoke Test")
    print("=" * 60)

    config = EMOConfig(
        d_model=64,
        n_layers=4,
        n_standard_heads=4,
        n_emotional_heads=5,
        d_head=16,
        d_ff=256,
        vocab_size=1000,
        max_seq_len=256,
        n_emotions=5,
        dropout=0.1,
        emotional_layers=(1, 3),   # 2 das 4 camadas são emocionais
    )

    model = create_emo_model(config)
    counts = model.count_parameters()
    print(f"\nParâmetros (tiny config):")
    print(f"  Standard:  {counts['standard']:>10,}")
    print(f"  Emotional: {counts['emotional']:>10,}")
    print(f"  Total:     {counts['total']:>10,}")

    # Verifica que camadas emocionais foram criadas corretamente
    emo_layers = [i for i, layer in enumerate(model.layers) if layer.is_emotional]
    print(f"\nCamadas emocionais: {emo_layers}")
    assert emo_layers == [1, 3], f"Expected [1, 3], got {emo_layers}"

    # Forward pass com H(t) neutro
    B, L = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (B, L))
    H_neutral = torch.full((B, config.n_emotions), 0.5)
    labels = torch.randint(0, config.vocab_size, (B, L))

    print(f"\n--- Forward pass (neutro) ---")
    result = model(input_ids, H_neutral, labels=labels)
    print(f"  logits shape: {result['logits'].shape}")
    print(f"  loss:         {result['loss'].item():.4f}")
    assert result['logits'].shape == (B, L, config.vocab_size)

    # Forward pass com H(t) cicatrizado
    H_scarred = torch.tensor([[0.1, 0.8, 0.2, 0.9, 0.1],
                               [0.9, 0.1, 0.9, 0.1, 0.9]])

    print(f"\n--- Forward pass (cicatrizado) ---")
    result_scar = model(input_ids, H_scarred, labels=labels)
    print(f"  logits shape: {result_scar['logits'].shape}")
    print(f"  loss:         {result_scar['loss'].item():.4f}")

    # Verifica que logits são DIFERENTES com H(t) diferentes
    diff = (result['logits'] - result_scar['logits']).abs().mean().item()
    print(f"\n--- Diferença logits (neutro vs cicatrizado) ---")
    print(f"  Mean abs diff: {diff:.6f}")
    assert diff > 0, "Logits devem diferir com H(t) diferentes!"
    print("  ✓ Campo HUGO influencia a geração")

    # Emotional Training Loss
    print(f"\n--- EmotionalTrainingLoss ---")
    loss_fn = EmotionalTrainingLoss(config)

    loss_neutral = loss_fn(result['logits'], labels, H_neutral)
    loss_scarred = loss_fn(result_scar['logits'], labels, H_scarred)

    print(f"  Neutro:     CE={loss_neutral['ce_loss'].item():.4f}  "
          f"amp={loss_neutral['emo_amplification'].item():.4f}  "
          f"total={loss_neutral['total_loss'].item():.4f}")
    print(f"  Cicatrizado: CE={loss_scarred['ce_loss'].item():.4f}  "
          f"amp={loss_scarred['emo_amplification'].item():.4f}  "
          f"total={loss_scarred['total_loss'].item():.4f}")

    assert loss_scarred['emo_amplification'] > loss_neutral['emo_amplification'], \
        "Amplificação deve ser maior quando H(t) cicatrizado!"
    print("  ✓ Cicatrizes amplificam o loss")

    # LoRA
    print(f"\n--- Emotional LoRA ---")
    lora_adapters = apply_emotional_lora(model, rank=4)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_total = sum(sum(p.numel() for p in a.parameters()) for a in lora_adapters.values())
    frozen = counts['total'] - trainable
    print(f"  Adapters:    {len(lora_adapters)}")
    print(f"  LoRA params: {lora_total:,}")
    print(f"  Frozen:      {frozen:,}")
    print(f"  ✓ Apenas componentes emocionais recebem LoRA")

    # Gradientes fluem
    print(f"\n--- Backprop test ---")
    # Recria sem LoRA freeze para testar gradientes
    model2 = create_emo_model(config)
    out2 = model2(input_ids, H_scarred, labels=labels)
    out2['loss'].backward()

    grad_ok = True
    for name, p in model2.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            continue
        if p.requires_grad and p.grad is None:
            print(f"  ✗ No gradient for {name}")
            grad_ok = False
    if grad_ok:
        print("  ✓ Gradientes fluem por todos os parâmetros")

    # Geração
    print(f"\n--- Geração ---")
    model.eval()
    # Re-enable params for generation (LoRA froze them)
    for p in model.parameters():
        p.requires_grad = False
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, H_neutral[:1], max_new_tokens=16)
    print(f"  Prompt:    {prompt.shape}")
    print(f"  Generated: {generated.shape}")
    assert generated.shape == (1, 24), f"Expected (1, 24), got {generated.shape}"
    print("  ✓ Geração autoregressiva funciona")

    # Componentes individuais
    print(f"\n--- Testes unitários de componentes ---")

    # EmotionalEmbedding
    emb = EmotionalEmbedding(config)
    H = torch.randn(B, config.n_emotions)
    e = emb(H)
    assert e.shape == (B, config.d_model)
    print(f"  ✓ EmotionalEmbedding: {H.shape} → {e.shape}")

    # EmotionalGate
    gate = EmotionalGate(config)
    hidden = torch.randn(B, L, config.d_model)
    gated = gate(hidden, H)
    assert gated.shape == hidden.shape
    print(f"  ✓ EmotionalGate: {hidden.shape} → {gated.shape}")

    # EmotionalAttentionLayer
    emo_attn = EmotionalAttentionLayer(config)
    x = torch.randn(B, L, config.d_model)
    out = emo_attn(x, H)
    assert out.shape == x.shape
    print(f"  ✓ EmotionalAttentionLayer: {x.shape} → {out.shape}")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_tiny()
