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

### 学習設定（large config v3 — 最新）

| ハイパーパラメータ | 値 |
|---|---|
| バッチサイズ | 48（AMP BF16 有効時） |
| 学習率 | 3e-4（Cosine decay → 1e-5）|
| Warmup | 3,000 steps |
| エポック数 | 30（Early stopping: patience=3）|
| Grad clip | 1.0 |
| Optimiser | AdamW (β=0.9, 0.999) |
| Dropout | 0.2 |
| Weight decay | 0.05 |
| 拡散ステップ T | 25 |
| 損失関数 | 全位置 CE（マスク位置重み 5.0） |

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

### val_loss の推移（概要）

| モデル | データセット | 設定 | Best Epoch | Best val_loss | 備考 |
|---|---|---|---|---|---|
| 13M (base) | WikiText-2 | v1 | ~62 | 7.22 | Epoch 2 でプラトー |
| 55M (large) | WikiText-2 | v1 | 4 | 7.23 | モデル拡大でも改善なし |
| 55M (large) | WikiText-103 | v1 | 4 | 7.19 | データ増量で改善 |
| 55M (large) | WikiText-103 | v2 | 8 | 4.63 | 全位置ロス + T=25 |
| **55M (large)** | **WikiText-103** | **v3** | **9** | **4.64** | **正則化強化（Best）** |

### 設定バージョンの変遷

| 設定 | 変更点 | 効果 |
|---|---|---|
| v1 | マスク位置のみ CE、T=100、LR=1e-4 | val_loss 7.19 で頭打ち |
| v2 | 全位置ロス（mask_weight=5.0）、T=25、LR=5e-4 | val_loss 4.63 まで改善、ただし過学習発生 |
| **v3** | dropout=0.2、weight_decay=0.05、LR=3e-4、patience=3 | 過学習を抑制、安定した収束 |

**プラトーの原因はモデルサイズではなくデータ量と損失関数**。WikiText-2 → WikiText-103 への切り替え、および全位置ロスの導入が劇的な改善をもたらした。

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

#### ANE 最適化版（Conv2d 変換 + 量子化、v3 モデル）

| モデル | Compute | ms/step (T=20) | サイズ |
|---|---|---|---|
| FP16（Conv2d） | ALL | 3.03 ms | 106 MB |
| **INT8（Conv2d）** | **ALL** | **2.27 ms 🏆** | **54 MB** |
| INT4（Conv2d） | ALL | 2.33 ms | 27 MB |
| FP16（標準） | ALL | 3.21 ms | 106 MB |
| INT4（標準） | ALL | 2.48 ms | 27 MB |
| 全モデル | CPU_ONLY | ~8.6 ms | — |

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
# 穴埋め生成（top-p + 繰り返しペナルティ + 進捗表示）
python sample.py \
    --checkpoint checkpoints_wt103_v3/best_model.pt \
    --prompt "The [MASK] of the [MASK] was discovered in [MASK]." \
    --top-p 0.9 --repetition-penalty 1.2 --verbose

# ノイズからの全生成（複数サンプル、シード固定）
python sample.py \
    --checkpoint checkpoints_wt103_v3/best_model.pt \
    --num-samples 5 --seed 42 --verbose

# カスタム設定
python sample.py \
    --checkpoint checkpoints_wt103_v3/best_model.pt \
    --steps 25 --temperature 0.8 --top-k 50 --top-p 0.95
```

### サンプリングオプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--steps` | 0（= config.T） | デノイジングステップ数 |
| `--temperature` | 1.0 | サンプリング温度 |
| `--top-k` | 0（無効） | Top-k フィルタリング |
| `--top-p` | 0.0（無効） | Nucleus (top-p) サンプリング |
| `--repetition-penalty` | 1.2 | 繰り返しペナルティ（1.0 で無効） |
| `--num-samples` | 1 | 生成サンプル数 |
| `--seed` | None | 再現性用のランダムシード |
| `-v`, `--verbose` | False | ステップごとのデノイジング進捗表示 |

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
| 生成品質の飛躍的改善 | Qwen 3.5 9B からのロジット蒸留（計画書あり） |
| 日本語対応 | トークナイザ変更 + Discord チャットログの活用 |
| torch.compile | ネイティブ Linux 環境（WSL2 外）なら有効化可能 |
| さらなる ANE 最適化 | シーケンス長 256 以上でより効果的 |
