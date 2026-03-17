# RinGo-DLLM — Diffusion LLM × Apple Neural Engine 実装計画
> Mac mini M4 / トイモデル / スクラッチ学習

---

## 概要

**目標:** Apple Neural Engine (ANE) で動作する Masked Diffusion Language Model を、PyTorchでゼロから設計・学習・変換・デプロイする。

**モデル選定: Masked Diffusion LM (MDLM系)**
ANEとの相性を最優先に選定。連続ノイズ（Gaussian）ではなくトークンを離散的にマスクする方式は、Transformerの標準的なforward passと親和性が高く、CoreML変換が最もシンプルになる。

---

## フェーズ構成

```
Phase 1: 設計 (1〜2日)
Phase 2: PyTorch実装 & 学習 (3〜7日)
Phase 3: CoreML変換 (2〜3日)
Phase 4: ANE最適化 & プロファイリング (2〜4日)
Phase 5: 推論デモ (1日)
```

---

## Phase 1 — アーキテクチャ設計

### モデル仕様（トイスケール）

| パラメータ | 値 | 備考 |
|---|---|---|
| モデルサイズ | ~10M params | ANEメモリに余裕で収まる |
| Transformer層数 | 6 layers | BERT-small相当 |
| Hidden dim | 256 | |
| Attention heads | 4 | head_dim = 64 (ANE最適) |
| FFN dim | 1024 | |
| 語彙サイズ | 30,522 | BERT tokenizer流用 |
| 最大コンテキスト長 | 128 tokens | 固定長（ANE制約） |

### Diffusion設定

| パラメータ | 値 |
|---|---|
| 拡散ステップ T | 100 (学習) / 10〜50 (推論) |
| ノイズ種別 | マスキング（[MASK]トークン） |
| ノイズスケジュール | Linear masking rate (α_t = 1 - t/T) |
| 損失関数 | Cross-entropy（マスクされたトークンのみ） |

### アーキテクチャ図

```
入力トークン列 (128 tokens)
        ↓
  [マスク適用 (forward process)]
  x_0 → x_t (rate α_t でランダムマスク)
        ↓
  Token Embedding + Positional Embedding
        ↓
  ┌─────────────────────────┐
  │  Transformer Encoder    │  × 6 layers
  │  MultiHeadAttention     │
  │  LayerNorm + FFN        │
  └─────────────────────────┘
        ↓
  Linear Head (hidden_dim → vocab_size)
        ↓
  [MASKトークン位置のlogits → Cross-Entropy Loss]
```

### ANE対応設計の注意点

- **固定入力サイズ必須**: ANEは動的shapeを苦手とするため、常にpadding込みの128固定長で処理
- **float16**: ANEはfloat32非対応のため、学習後にfloat16変換
- **Softmax/LayerNorm**: CoreMLのTransformerAttentionをそのまま使うか、op分解して確認
- **[MASK]埋め込みは学習可能パラメータ**: 別ベクトルとして用意

---

## Phase 2 — PyTorch実装 & 学習

### ディレクトリ構成

```
diffusion_llm_ane/
├── model/
│   ├── transformer.py      # Transformer Encoderブロック
│   ├── diffusion_lm.py     # MDLMラッパー + ノイズスケジューラ
│   └── config.py           # ハイパーパラメータ定義
├── data/
│   ├── dataset.py          # WikiText-2 DataLoader
│   └── tokenizer.py        # BertTokenizer ラッパー
├── train.py                # 学習メインスクリプト
├── sample.py               # 推論スクリプト（PyTorch版）
└── convert/
    ├── export_coreml.py    # CoreML変換スクリプト
    └── benchmark.py        # ANE vs GPU vs CPU比較
```

### 学習設定

| 設定 | 値 |
|---|---|
| データセット | WikiText-2 (約2MB、手頃なサイズ) |
| バッチサイズ | 64 |
| 学習率 | 1e-4 (AdamW) |
| エポック | 50〜100 |
| デバイス | MPS (Metal Performance Shaders) |
| 予想学習時間 | ~1〜3時間 (M4 MPS) |

### 学習ループの要点

```python
# forward process: トークンをランダムにマスク
t = torch.randint(0, T, (batch_size,))
mask_rate = 1 - t / T  # 時刻tでのマスク率
masked_x, mask_indices = apply_mask(x, mask_rate)

# model forward
logits = model(masked_x, t)  # tを条件として受け取る

# loss: マスクされた位置のみ
loss = cross_entropy(logits[mask_indices], x[mask_indices])
```

---

## Phase 3 — CoreML変換

### 変換戦略

PyTorch → TorchScript (trace) → coremltools → `.mlpackage`

```python
import coremltools as ct

# 固定入力でトレース
example_input = torch.zeros(1, 128, dtype=torch.long)
example_t = torch.tensor([50])  # 拡散ステップ

traced = torch.jit.trace(model, (example_input, example_t))

# CoreML変換
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, 128), dtype=np.int32),
        ct.TensorType(name="t", shape=(1,), dtype=np.float16),
    ],
    compute_precision=ct.precision.FLOAT16,  # ANE必須
    compute_units=ct.ComputeUnit.ALL,         # まずALL、後でANE_ONLYを試す
)
mlmodel.save("diffusion_lm.mlpackage")
```

### 変換時の主な落とし穴

| 問題 | 対処法 |
|---|---|
| Embedding opがANEに乗らない | `ct.ComputeUnit.ALL`でCPU fallbackを許容 |
| LayerNormが未対応 | RMSNormに置換するか、op分解 |
| Dynamic mask生成 | 推論時はマスク位置を外部入力として渡す |
| int32 Embeddingがfloat16非互換 | Embedding後にcast |

---

## Phase 4 — ANE最適化 & プロファイリング

### 計測項目

```
1. Xcode Instruments の "Core ML" テンプレートでどのopがどのユニットで実行されるかを確認
2. compute_units別のlatency比較:
   - CPU_ONLY
   - CPU_AND_GPU
   - ALL (CPU+GPU+ANE)
   - ALL_AND_ANE_FIRST (優先度高)
3. 拡散ステップ数 T別の推論時間 (T=10, 20, 50)
```

### 期待される結果

| compute_units | latency (1 denoising step) |
|---|---|
| CPU_ONLY | ~50ms |
| CPU_AND_GPU (MPS) | ~5ms |
| ALL (ANE含む) | ~2〜8ms (op依存) |

※ ANEがすべてのopを受け取れれば最速だが、fallbackが多いと逆に遅くなる可能性あり。

### ANE活用率を上げるチューニング

- Attentionを `ct.models.neural_network.NeuralNetwork` のTransformerAttentionに置換
- Batch Normalization的なパターンよりLayerNormの方が良い
- head_dim = 64 に固定（ANEはこのサイズを得意とする）

---

## Phase 5 — 推論デモ

### Masked Infilling (穴埋め) デモ

```
入力: "The capital of France is [MASK] [MASK] [MASK]."
T=50ステップで逆拡散
出力: "The capital of France is Paris , France ."
```

### 推論フロー（逆拡散）

```
x_T = 全トークンをMASK
for t = T, T-1, ..., 1:
    logits = mlmodel.predict(x_t, t)
    # 信頼度の高いトークンから順に確定 (greedy or sampling)
    x_{t-1} = unmask_high_confidence(x_t, logits, rate=(1-t/T))
output = x_0
```

---

## 技術スタック

| 用途 | ツール |
|---|---|
| モデル実装・学習 | PyTorch 2.x (MPSバックエンド) |
| Tokenizer | HuggingFace `transformers` (BertTokenizerFast) |
| データ | HuggingFace `datasets` (wikitext-2-raw-v1) |
| CoreML変換 | `coremltools` 8.x |
| プロファイリング | Xcode Instruments + `coremltools.models.MLModel` |
| Python環境 | Python 3.11, venv推奨 |

---

## リスクと対策

| リスク | 確率 | 対策 |
|---|---|---|
| CoreML変換でopが未対応 | 高 | `ct.ComputeUnit.ALL` でCPU fallbackを許容し、段階的にANE比率を上げる |
| ANEが使われず全部CPUに落ちる | 中 | Xcode Instrumentsで確認 → アーキテクチャを調整 |
| M4 MPS上の学習が遅い | 低 | M4はMPS性能が高く、10Mモデルなら2〜3時間以内 |
| Diffusion品質が低い (トイゆえ) | 高(許容) | 目的はANE動作確認なので品質は問わない |

---

## マイルストーン

- [ ] **Week 1**: Phase 1-2完了 — PyTorchモデルが学習・推論できる状態
- [ ] **Week 2**: Phase 3完了 — CoreML変換成功、`.mlpackage`生成
- [ ] **Week 2-3**: Phase 4完了 — ANEで少なくとも1層以上が動作することを確認
- [ ] **Week 3**: Phase 5完了 — 穴埋めデモが動作

---

## 参考文献

- **MDLM**: "Simplified and Generalized Masked Diffusion for Discrete Data" (Sahoo et al., 2024)
- **SEDD**: "Discrete Diffusion Modeling by Estimating the Ratios of the Unnormalized Probabilities" (Lou et al., 2024)
- **Apple CoreML docs**: https://coremltools.readme.io/docs
- **ANE最適化**: "Optimizing Models for Core ML" — WWDC 2023 Session
- **PyTorch MPS**: https://developer.apple.com/metal/pytorch/
