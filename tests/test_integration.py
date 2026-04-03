#!/usr/bin/env python3
"""
Integration test — valida que todos os módulos do pipeline funcionam.
Usa config tiny para caber no container de 4GB.
"""
import sys
import tempfile
import json
from pathlib import Path

sys.path.insert(0, ".")

import torch


def test_imports():
    print("--- Test: Imports ---")
    from src.kappa.emo_model import (
        EMOConfig, EMOTransformer, EmotionalEmbedding, EmotionalGate,
        EmotionalAttentionLayer, EmotionalTrainingLoss, EmotionalLoRA,
        apply_emotional_lora, create_emo_model, EMOTokenizer,
        TrainConfig, EMOTrainer,
    )
    from src.kappa.emo_model.dataset import (
        StreamingPretrainDataset, EmotionalDataset,
        create_pretrain_dataloader,
    )
    print("  ✓ Todos os imports OK")


def test_tokenizer():
    print("\n--- Test: Tokenizer ---")
    from src.kappa.emo_model.tokenizer import EMOTokenizer

    with tempfile.TemporaryDirectory() as tmpdir:
        # Cria corpus fake
        corpus_file = Path(tmpdir) / "corpus.txt"
        sentences = [
            "O mercado financeiro brasileiro apresenta volatilidade crescente.",
            "A análise topológica revela padrões ocultos nos dados.",
            "O campo HUGO modula a geração de texto no modelo.",
            "Investidores devem considerar a diversificação de portfólio.",
            "O número de Ohio Oh(t) indica a instabilidade estrutural.",
        ] * 200  # Repetir para ter corpus mínimo
        corpus_file.write_text("\n".join(sentences), encoding="utf-8")

        tok = EMOTokenizer()
        model_path = tok.train(
            texts=iter(sentences),
            vocab_size=500,
            output_dir=str(Path(tmpdir) / "tok"),
            model_prefix="test_bpe",
            max_samples=1000,
        )

        assert tok.vocab_size == 500
        encoded = tok.encode("O mercado financeiro")
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        decoded = tok.decode(encoded)
        assert "mercado" in decoded.lower()

        print(f"  ✓ Tokenizer treinado: vocab={tok.vocab_size}")
        print(f"  ✓ Encode: 'O mercado financeiro' → {encoded[:8]}...")
        print(f"  ✓ Decode: → '{decoded}'")

        return tok, model_path


def test_dataset_local(tok):
    print("\n--- Test: Dataset (local) ---")
    from src.kappa.emo_model.dataset import create_pretrain_dataloader

    with tempfile.TemporaryDirectory() as tmpdir:
        # Cria corpus
        corpus_file = Path(tmpdir) / "test.txt"
        lines = ["Esta é uma frase de teste para o dataset streaming do modelo. " * 5] * 100
        corpus_file.write_text("\n".join(lines), encoding="utf-8")

        loader = create_pretrain_dataloader(
            tokenizer=tok,
            max_seq_len=64,
            batch_size=4,
            max_samples=50,
            local_path=tmpdir,
        )

        batch = next(iter(loader))
        assert batch["input_ids"].shape == (4, 64), f"Got {batch['input_ids'].shape}"
        assert batch["labels"].shape == (4, 64)
        assert batch["H_t"].shape == (4, 5)
        assert batch["attention_mask"].shape == (4, 64)

        # H(t) deve ser neutro [0.5]*5 em Phase 1
        assert torch.allclose(batch["H_t"], torch.full((4, 5), 0.5))

        print(f"  ✓ Batch shapes OK: input_ids={batch['input_ids'].shape}")
        print(f"  ✓ H(t) neutro: {batch['H_t'][0].tolist()}")
        print(f"  ✓ Labels com mask: {(batch['labels'] == -100).sum()} padding tokens masked")


def test_emotional_dataset(tok):
    print("\n--- Test: Emotional Dataset ---")
    from src.kappa.emo_model.dataset import create_emotional_dataloader

    with tempfile.TemporaryDirectory() as tmpdir:
        # Cria dataset fake
        data_file = Path(tmpdir) / "test.jsonl"
        samples = []
        for i in range(20):
            samples.append(json.dumps({
                "prompt": f"Analise o cenário {i}",
                "response": f"A análise indica que o cenário {i} apresenta volatilidade.",
                "H_t": [round(0.1 + i * 0.04, 2)] * 5,
                "quality_score": 0.7,
            }))
        data_file.write_text("\n".join(samples), encoding="utf-8")

        loader = create_emotional_dataloader(
            tokenizer=tok,
            data_path=tmpdir,
            max_seq_len=64,
            batch_size=4,
        )

        batch = next(iter(loader))
        assert batch["input_ids"].shape[0] == 4
        assert batch["H_t"].shape == (4, 5)
        # H(t) NÃO deve ser neutro (variado)
        assert not torch.allclose(batch["H_t"][0], batch["H_t"][1])

        print(f"  ✓ Emotional batch OK: input_ids={batch['input_ids'].shape}")
        print(f"  ✓ H(t) variado: {batch['H_t'][0].tolist()}")


def test_trainer_tiny(tok):
    print("\n--- Test: Trainer (3 steps) ---")
    from src.kappa.emo_model.model import EMOConfig, create_emo_model
    from src.kappa.emo_model.dataset import create_pretrain_dataloader
    from src.kappa.emo_model.trainer import TrainConfig, EMOTrainer

    config = EMOConfig(
        d_model=32, n_layers=2, n_standard_heads=2, n_emotional_heads=2,
        d_head=16, d_ff=64, vocab_size=tok.vocab_size,
        max_seq_len=32, n_emotions=5, emotional_layers=(1,),
    )
    model = create_emo_model(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Corpus
        corpus = Path(tmpdir) / "corpus"
        corpus.mkdir()
        (corpus / "train.txt").write_text(
            "\n".join(["Teste de treinamento do modelo EMO. " * 10] * 50),
            encoding="utf-8",
        )

        loader = create_pretrain_dataloader(
            tokenizer=tok, max_seq_len=32, batch_size=2,
            max_samples=20, local_path=str(corpus),
        )

        train_cfg = TrainConfig(
            learning_rate=1e-3, batch_size=2, gradient_accumulation_steps=1,
            warmup_steps=1, max_steps=3, eval_interval=100, save_interval=100,
            log_interval=1, use_amp=False, amp_dtype="float32",
            output_dir=str(Path(tmpdir) / "ckpt"), phase=1,
            emotional_loss_enabled=False,
        )

        trainer = EMOTrainer(model=model, train_config=train_cfg, train_loader=loader)
        trainer.train()

        # Verifica checkpoint
        assert (Path(tmpdir) / "ckpt" / "final" / "model.pt").exists()
        assert (Path(tmpdir) / "ckpt" / "final" / "config.json").exists()

        print(f"  ✓ Trainer executou 3 steps sem erros")
        print(f"  ✓ Checkpoint salvo em {tmpdir}/ckpt/final/")


def main():
    print("=" * 60)
    print("Kappa-EMO — Integration Test")
    print("=" * 60)

    test_imports()
    tok, _ = test_tokenizer()
    test_dataset_local(tok)
    test_emotional_dataset(tok)
    test_trainer_tiny(tok)

    print(f"\n{'=' * 60}")
    print("ALL INTEGRATION TESTS PASSED!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
