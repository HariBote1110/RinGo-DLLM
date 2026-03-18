# RinGo-DLLM

Masked Diffusion Language Model (MDLM) の実装実験プロジェクト。
PyTorch で学習し、Apple Neural Engine (ANE) への CoreML 変換と推論最適化までを一貫して行う。

---

## 🎯 プロジェクトの目標

- Masked Diffusion LM をゼロから実装して動かす
- RTX 3070 Ti (Windows PC) で学習 → Mac mini M4 の ANE で高速推論
- CoreML + ANE を活用した「モバイル・エッジ向け言語モデル」パイプラインの確立

---

## 📐 アーキテクチャ

### モデル: Masked Diffusion Language Model

GPT 系の「左から右に生成」とは異なり、**ノイズ（[MASK]）から復元する双方向モデル**。

```
入力（マスク済み）: The [MASK] of the [MASK] was discovered in the [MASK] century.
          ↓ 逆拡散（T ステップ）
出力（復元）    : The history of the universe was discovered in the 20th century.
```

#### 拡散プロセス

| フェーズ | 処理 |
|---|---|
| **Forward diffusion** | タイムステップ `t` に応じた割合でトークンを `[MASK]` に置換 |
| **Reverse diffusion** | モデルが高信頼度のトークンから順にアンマスク（T ステップ繰り返し） |

#### マスクスケジュール

```
linear:  mask_rate = t / T
cosine:  mask_rate = (1 - cos(π·t/T)) / 2   ← 現在使用中（中間 t の学習が密に）
```

### Transformer エンコーダ（Pre-LayerNorm）

```
Token Embedding (vocab=30,522)
  + Positional Embedding (seq_len=128)
  + Sinusoidal Time Embedding (t → 256d → hidden_dim)
        ↓
 TransformerEncoder × 12 layers
   └─ Pre-LN → Flash Attention (SDPA) → Residual
   └─ Pre-LN → FFN (GELU) → Residual
        ↓
 LM Head (hidden → vocab, weight-tied with token embedding)
```

### モデルサイズ比較

| 設定 | パラメータ数 | hidden_dim | layers | heads |
|---|---|---|---|---|
| `base` | 13M | 256 | 6 | 4 |
| `large` | **55M** | 512 | 12 | 8 |

---

## 🏋️ 学習

### 環境

| 項目 | 内容 |
|---|---|
| GPU | NVIDIA RTX 3070 Ti (8 GB VRAM) |
| OS | Windows 11 + WSL2 (Ubuntu) |
| Python | 3.11 |
| PyTorch | 2.6 (CUDA) |
| データセット | WikiText-103（~103M トークン、WikiText-2 の 50 倍）|

### 学習設定（large config）

| ハイパーパラメータ | 値 |
|---|---|
| バッチサイズ | 64（AMP 有効時） |
| 学習率 | 3e-4（Cosine decay → 1e-5）|
| Warmup | 5,000 steps |
| エポック数 | 30（Early stopping: patience=5）|
| Grad clip | 1.0 |
| Optimiser | AdamW (β=0.9, 0.999) |

### 学習の高速化

| 最適化 | 効果 |
|---|---|
| **Flash Attention** (SDPA) | Attention が O(L²) メモリ → O(L)、CUDA フューズドカーネル |
| **AMP BF16** | 演算を BF16 で実行、VRAM 半減 → バッチサイズ 2 倍 |
| **バッチサイズ 64** | GPU 使用率が ~90% まで向上 |
| ~~torch.compile~~ | WSL2 の Triton ビルド制約で断念 |

**改善前後の比較（55M モデル、WikiText-103）:**

| 設定 | Epoch 時間 | バッチ数/epoch | VRAM |
|---|---|---|---|
| FP32、batch=32 | ~95 秒 | 27,811 | ~5 GB |
| **BF16 AMP、batch=64** | **~45 秒（推定）** | **13,905** | **~8 GB** |

### 学習の実行方法

```bash
cd diffusion_llm_ane

# 初回（base モデル、WikiText-2）
python train.py --config base

# 本番（large モデル、WikiText-103、Discord 通知付き）
python train.py --config large --dataset wikitext-103 --epochs 30 \
    --webhook "https://discord.com/api/webhooks/..." --notify-every 1

# チェックポイントから再開
python train.py --config large --dataset wikitext-103 \
    --resume checkpoints_wt103/best_model.pt --no-compile
```

### Discord Webhook 通知

`.env` ファイルを `diffusion_llm_ane/` に作成：

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

エポック完了・Best model 更新・エラー発生時に自動通知。

---

## 📊 学習結果

### val_loss の推移

| モデル | データセット | Epoch | val_loss | 備考 |
|---|---|---|---|---|
| 13M (base) | WikiText-2 | ~62 | 7.22 | Epoch 2 でプラトー |
| 55M (large) | WikiText-2 | 4 | 7.23 | モデル拡大でも改善なし |
| **55M (large)** | **WikiText-103** | **1** | **7.19** | データ増量で改善 ✅ |

**プラトーの原因はモデルサイズではなくデータ量**。WikiText-2（~2MB）ではモデルの容量に対してデータが不足していた。

---

## 🍎 CoreML 変換・ANE 推論

### 変換フロー

```
PyTorch チェックポイント (.pt)
        ↓ torch.jit.trace
   TorchScript モデル
        ↓ coremltools.convert()
  CoreML モデル (.mlpackage)
        ↓ ANE / GPU / CPU で推論
```

### 変換コマンド

```bash
# 標準変換（FP16）
python convert/export_coreml.py \
    --checkpoint checkpoints_wt103/best_model.pt \
    --output convert/diffusion_lm_wt103.mlpackage

# ANE 最適化版（Conv2d 変換）
python convert/export_coreml_ane.py \
    --checkpoint checkpoints/best_model.pt \
    --output convert/diffusion_lm_ane.mlpackage

# INT8 / INT4 量子化
python convert/quantise.py
```

### ベンチマーク結果（Mac mini M4）

#### 13M モデル（标準 Linear）

| Compute Unit | ms/step (T=20) | 備考 |
|---|---|---|
| CPU_ONLY | **2.91 ms** 🏆 | ANE より速い（モデルが小さすぎ）|
| ALL (ANE) | 4.93 ms | 転送オーバーヘッドが支配的 |

#### 55M モデル（WikiText-103、FP16）

| Compute Unit | ms/step (T=20) | 備考 |
|---|---|---|
| CPU_ONLY | 7.99 ms | |
| **ALL (ANE)** | **3.16 ms** 🏆 | **CPU の 2.5 倍速！** |

**55M 以上の規模になって初めて ANE の計算能力が転送オーバーヘッドを上回る。**

#### ANE 最適化版（Conv2d 変換 + 量子化）

| モデル | Compute | ms/step (T=20) |
|---|---|---|
| FP16（Conv2d） | ALL | 1.31 ms |
| INT8（Conv2d） | ALL | ~1.4 ms |
| INT4（Conv2d） | ALL | ~1.5 ms |
| INT4（標準） | CPU_ONLY | 2.68 ms |

---

## 🔧 ベンチマーク実行

```bash
# 単一モデルのベンチマーク
python convert/benchmark.py \
    --model convert/diffusion_lm_wt103.mlpackage \
    --steps 10 20 50

# 全モデル一括ベンチマーク
python convert/benchmark_all.py
```

---

## 🎲 生成サンプリング（逆拡散）

```bash
# 穴埋め生成
python sample.py \
    --checkpoint checkpoints_wt103/best_model.pt \
    --prompt "The [MASK] of the [MASK] was discovered in [MASK]." \
    --steps 50 --temperature 1.0 --top-k 10

# ノイズからの全生成（全 [MASK]）
python sample.py \
    --checkpoint checkpoints_wt103/best_model.pt \
    --prompt "[MASK] [MASK] ... [MASK]" \
    --steps 100 --temperature 0.8 --top-k 50
```

> **注意**: 現時点（学習初期）のモデルはパープレキシティ ~1,300 と高く、
> 生成テキストは高頻度語（the, of, and...）の羅列になる。
> 学習が進むにつれて改善される見込み。

---

## 📁 ファイル構成

```
diffusion_llm_ane/
├── model/
│   ├── config.py           # base モデル設定（13M）
│   ├── config_large.py     # large モデル設定（55M）
│   ├── transformer.py      # Flash Attention 対応 Transformer
│   ├── transformer_ane.py  # ANE 最適化版（Conv2d 形式）
│   ├── diffusion_lm.py     # メインモデル
│   └── diffusion_lm_ane.py # ANE 最適化版モデル
├── data/
│   ├── dataset.py          # WikiText データセット + キャッシュ
│   └── tokenizer.py        # BertTokenizerFast シングルトン
├── convert/
│   ├── export_coreml.py    # 標準 CoreML 変換
│   ├── export_coreml_ane.py# ANE 最適化 CoreML 変換
│   ├── quantise.py         # INT8 / INT4 量子化
│   ├── benchmark.py        # 単一モデル・ベンチマーク
│   └── benchmark_all.py    # 全モデル一括ベンチマーク
├── train.py                # 学習スクリプト（AMP + Flash Attn）
├── sample.py               # 逆拡散サンプリング
├── notify.py               # Discord Webhook 通知
├── sanity_check.py         # 動作確認スクリプト
└── requirements.txt        # 依存パッケージ
```

---

## 🔍 学んだこと・気づき

### ANE を活かすための条件

- **モデルが小さすぎると ANE は逆効果**（13M: ANE < CPU）
- **55M 以上になると ANE が CPU の 2〜3 倍速**になる
- **Conv2d 形式への変換**が ANE 最適化の鍵（`nn.Linear` のまま変換しても ANE に乗りにくい）
- ANE は **推論専用**。学習（バックプロパゲーション）はできない

### 学習のボトルネック

- WikiText-2（2MB）ではモデルがどれだけ大きくても Epoch 2 でプラトー
- **データ量が最大のボトルネック**（モデルサイズより重要）
- WikiText-103（50 倍）に切り替えて初めて継続的な改善が見られた

### WSL2 + SSH での遠隔学習

- `tmux` でセッション永続化、`PYTHONUNBUFFERED=1` でリアルタイムログ
- `torch.compile` は WSL2 の Triton ビルド制約で使用不可
- スリープ対策: `powercfg /change standby-timeout-ac 0`
- Windows 側で WSL2 ウィンドウを操作するとセッションが落ちることがある

---

## 🚀 今後の展望

| 課題 | 対応策 |
|---|---|
| 生成品質の改善 | 学習継続（WikiText-103、30 epochs）|
| 会話モデル化 | GPT 系 Decoder アーキテクチャに移行を検討 |
| torch.compile | ネイティブ Linux 環境（WSL2 外）なら有効化可能 |
| さらなる ANE 最適化 | シーケンス長 256 以上でより効果的 |
