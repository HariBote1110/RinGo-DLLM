# 引き継ぎ書: RinGo-DLLM 開発チェックポイント

**作成日**: 2026-03-19
**最終更新**: 2026-03-20（v3 学習 Epoch 6 完了時点）

---

## 1. プロジェクト概要

**RinGo-DLLM** は Masked Diffusion Language Model（MDLM）ベースの 55M パラメータ言語モデル。
Apple Neural Engine（ANE）での高速推論を目指し、CoreML 変換にも対応している。

- **学習マシン**: RTX 3070 Ti（8 GB VRAM / Windows PC / WSL2）
- **推論ベンチマーク**: Mac mini M4（ANE / CoreML）
- **リモート操作方法**: Mac mini → SSH → Windows PC（WSL2）

---

## 2. SSH 接続情報

### 基本接続コマンド

```bash
ssh -p 2222 -i ~/.ssh/id_ed25519 gzabu@192.168.0.30
```

| 項目 | 値 |
|---|---|
| ホスト | `192.168.0.30`（LAN 固定 IP） |
| ポート | `2222` |
| ユーザー | `gzabu` |
| 認証鍵 | `~/.ssh/id_ed25519`（Mac mini 側に保存済み） |
| OS | WSL2 Ubuntu（Linux 6.6.87.2-microsoft-standard-WSL2） |

### 接続できない場合のトラブルシュート

WSL2 は以下の原因でよく落ちる。**Windows 側**（RDP or 物理操作）で対処する。

**手順 1: WSL を再起動**

```powershell
# PowerShell（管理者権限不要）
wsl --shutdown
wsl
```

**手順 2: WSL 内で SSH サービスを起動**

```bash
sudo service ssh start
```

**手順 3: 再接続**

```bash
ssh -p 2222 -i ~/.ssh/id_ed25519 gzabu@192.168.0.30
```

### よくある接続失敗の原因と対処

| 症状 | 原因 | 対処 |
|---|---|---|
| `Connection refused` | sshd が停止している | `sudo service ssh start` |
| `Connection closed by ...` | WSL2 ごとクラッシュ | `wsl --shutdown` → `wsl` → `sudo service ssh start` |
| 接続できるが重い | VRAM 溢れ or C ドライブ逼迫 | タスクマネージャで確認 |
| スリープ後に切断 | Windows のスリープ | `powercfg /change standby-timeout-ac 0` で無効化済み |

### 過去に発生したクラッシュの記録

- **VRAM 溢れ（エラーコード 0xe0000008）**: batch_size が大きすぎると VRAM が溢れ WSL2 ごとクラッシュ。共有 GPU メモリへの 3 GB 溢れが目視で確認された
- **C ドライブ容量不足**: 学習中にディスクが枯渇してクラッシュ。現在 C ドライブに約 45 GB の空きを確保済み（2026-03-20 時点）

---

## 3. 学習環境のセットアップ

### ハードウェア情報

```
GPU:     NVIDIA GeForce RTX 3070 Ti
VRAM:    8192 MiB
Driver:  580.88
OS:      WSL2 Ubuntu（Linux 6.6.87.2-microsoft-standard-WSL2）
Python:  3.11.15（.venv）
C Drive: 465 GB 中 約 45 GB 空き（2026-03-20 時点）
```

> **注意**: GPU に電力制限をかけており（2026-03-20 以降）、学習速度が若干低下する場合がある。長時間 100% 稼働によるクラッシュリスクを避けるための意図的な制限。

### ディレクトリ構成

```
~/RinGo-DLLM/
└── diffusion_llm_ane/
    ├── model/
    │   ├── config.py             # base 設定（13M モデル用）
    │   ├── config_large.py       # large 設定（55M モデル用）★メイン
    │   ├── transformer.py        # Flash Attention 対応 Encoder
    │   ├── diffusion_lm.py       # MDLM 本体
    │   ├── diffusion_lm_ane.py   # ANE 最適化版（Conv2d 変換済み）
    │   └── transformer_ane.py    # ANE 最適化 Transformer
    ├── data/
    │   ├── dataset.py            # WikiText-2/103 データローダ
    │   └── tokenizer.py          # BERT トークナイザラッパー
    ├── convert/
    │   ├── export_coreml.py      # CoreML 変換
    │   ├── export_coreml_ane.py  # ANE 最適化 CoreML 変換
    │   ├── benchmark.py          # CoreML ベンチマーク
    │   ├── benchmark_all.py      # 全形式ベンチマーク
    │   └── quantise.py           # INT4/INT8 量子化
    ├── train.py                  # 学習スクリプト ★メイン
    ├── sample.py                 # テキスト生成スクリプト
    ├── notify.py                 # Discord Webhook 通知
    ├── sanity_check.py           # モデル動作確認
    ├── requirements.txt
    ├── .env                      # DISCORD_WEBHOOK_URL を保存
    ├── .venv/                    # Python 仮想環境
    ├── checkpoints_wt103_v3/     # v3 チェックポイント ★最新
    │   ├── best_model.pt         # Best モデル（Epoch 6, val=4.7619）
    │   └── epoch_0005.pt         # Epoch 5 定期保存
    ├── train_wt103_v3.log        # v3 学習ログ ★最新
    ├── train_wt103_v2.log        # v2 学習ログ（Early Stopping 完了）
    ├── train_wt103.log           # v1 学習ログ
    └── data/                     # WikiText-103 キャッシュ（トークナイズ済み）
```

### Python 実行パス

**必ず venv のフルパスを使うこと。** `python` コマンドは WSL2 に存在しない場合がある。

```bash
# 正しい実行方法
~/RinGo-DLLM/diffusion_llm_ane/.venv/bin/python3 train.py ...

# または venv を有効化してから
source ~/RinGo-DLLM/diffusion_llm_ane/.venv/bin/activate
python3 train.py ...
```

### Discord Webhook

`.env` ファイルに `DISCORD_WEBHOOK_URL` が保存されている。学習スクリプトは起動時に自動で読み込む。

```bash
# .env の内容を確認
cat ~/RinGo-DLLM/diffusion_llm_ane/.env
```

---

## 4. tmux の操作方法

学習は `wt103` という名前の tmux セッションで実行している。

```bash
# セッション一覧を確認
tmux ls

# セッションにアタッチ（リアルタイム出力を見る）
tmux attach -t wt103

# デタッチ（セッションを止めずに抜ける）
Ctrl + B, D

# アタッチせずに最新出力だけ確認する
tmux capture-pane -t wt103 -p | tail -30

# 新しいセッションを作成
tmux new-session -s wt103
```

---

## 5. 学習ログの確認

ログファイルには ANSI コード（カラー制御文字）が混入することがある。
そのまま `grep` すると `binary file matches` エラーになるため、**必ず `strings` でフィルタしてから使う**。

```bash
# Epoch 完了・Best 更新・Early Stopping の行だけ抽出
strings ~/RinGo-DLLM/diffusion_llm_ane/train_wt103_v3.log | grep -E '(Epoch|Best|Early)'

# ステップごとのリアルタイム確認
tail -f ~/RinGo-DLLM/diffusion_llm_ane/train_wt103_v3.log | strings

# GPU 使用状況
/usr/lib/wsl/lib/nvidia-smi

# GPU 使用状況を 2 秒ごとに更新
watch -n 2 /usr/lib/wsl/lib/nvidia-smi
```

---

## 6. 学習の起動コマンド

### 新規に v3 を起動する（tmux 内で実行）

```bash
# tmux セッション作成
tmux new-session -s wt103

# セッション内で以下を実行
cd ~/RinGo-DLLM/diffusion_llm_ane
PYTHONUNBUFFERED=1 .venv/bin/python3 train.py \
    --config large \
    --log train_wt103_v3.log \
    --notify-every 1 \
    2>&1 | tee -a train_wt103_v3.log

# アタッチしたまま走らせて、デタッチ: Ctrl+B → D
```

### チェックポイントから再開する

```bash
PYTHONUNBUFFERED=1 .venv/bin/python3 train.py \
    --config large \
    --log train_wt103_v3.log \
    --resume checkpoints_wt103_v3/best_model.pt \
    --notify-every 1 \
    2>&1 | tee -a train_wt103_v3.log
```

> `--notify-every 1` を付けると**毎 Epoch 通知が来る**。省略するとデフォルト 10 epochs ごとになる（Best 更新時は常に通知される）。

---

## 7. Discord 通知の仕様

**重要**: 通知が来ない Epoch があっても学習は止まっていない可能性が高い。

```python
# train.py 内の通知条件
if is_best or (epoch + 1) % args.notify_every == 0:
    notifier.send(...)
```

| 条件 | 通知 |
|---|---|
| Best モデルが更新された | ✅ 常に来る |
| `--notify-every N` の倍数 Epoch（デフォルト N=10） | ✅ 来る |
| Best 更新なし かつ倍数でもない Epoch | ❌ 来ない |

→ **毎 Epoch 通知したい場合**: `--notify-every 1` を付けて起動する。

---

## 8. テキスト生成（sample.py）

```bash
cd ~/RinGo-DLLM/diffusion_llm_ane

.venv/bin/python3 sample.py \
    --checkpoint checkpoints_wt103_v3/best_model.pt \
    --config large \
    --prompt "The history of" \
    --steps 25 \
    --num-samples 5
```

現在の生成品質は「高頻度語の羅列」程度。蒸留フェーズ後に大幅改善が期待される。

---

## 9. 学習設定の変遷

### 現在の v3 設定（`config_large.py`）

```python
ModelConfigLarge(
    vocab_size        = 30_522,    # BERT トークナイザ
    max_seq_len       = 128,
    hidden_dim        = 512,
    num_layers        = 12,
    num_heads         = 8,
    ffn_dim           = 2_048,
    dropout           = 0.2,       # v2: 0.1 → 過学習対策で強化
    T                 = 25,        # 拡散ステップ数（v1: 100 から削減）
    mask_schedule     = 'cosine',
    mask_loss_weight  = 5.0,       # 全位置ロスのマスク重み
    dataset_name      = 'wikitext-103',
    batch_size        = 48,
    learning_rate     = 3e-4,      # v2: 5e-4 から削減
    weight_decay      = 0.05,      # v3 で新規追加（正則化）
    num_epochs        = 30,
    warmup_steps      = 3_000,
    lr_schedule       = 'cosine',
    lr_min            = 1e-5,
    grad_clip         = 1.0,
    early_stopping_patience = 3,   # v2: 5 から短縮
    checkpoint_dir    = 'checkpoints_wt103_v3',
    save_every_n_epochs = 5,
)
```

### バージョン間の変更履歴

| 設定項目 | v1 | v2 | v3（現在） |
|---|---|---|---|
| 損失関数 | マスク位置のみ CE | **全位置 CE（mask_weight=5.0）** | 同左 |
| T（拡散ステップ） | 100 | **25** | 同左 |
| learning_rate | 1e-4 | **5e-4** | **3e-4** |
| dropout | 0.1 | 0.1 | **0.2** |
| weight_decay | なし | なし | **0.05** |
| early_stopping_patience | 5 | 5 | **3** |
| checkpoint_dir | checkpoints_wt103 | checkpoints_wt103 | **checkpoints_wt103_v3** |

---

## 10. 学習結果の全記録

### v1（マスク位置のみ損失）

| Epoch | Train Loss | Val Loss | 備考 |
|---|---|---|---|
| 4 | 7.21 | 7.19 | プラトー、以降改善なし |

### v2（全位置ロス + T=25 + LR=5e-4）

| Epoch | Train Loss | Val Loss | 更新 |
|---|---|---|---|
| 1 | 5.9798 | 5.7795 | ✅ Best |
| 2 | 5.4782 | 5.0315 | ✅ Best |
| 3 | 5.0002 | 4.9167 | ✅ Best |
| 4 | 4.8974 | 4.8422 | ✅ Best |
| 5 | 4.8367 | 4.7527 | ✅ Best |
| 6 | 4.7999 | 4.6955 | ✅ Best |
| 7 | 4.7683 | 4.7417 | ❌ |
| 8 | 4.7459 | **4.6309** | ✅ Best |
| 9 | 4.7257 | 4.6783 | ❌ |
| 10 | 4.7115 | 4.6780 | ❌ |
| 11 | 4.6922 | 4.7101 | ❌（過学習の兆候） |
| — | — | — | Early Stopping 発動（patience=5、3連続非更新後）|

### v3（dropout=0.2 + weight_decay=0.05 + LR=3e-4 + patience=3）★進行中

| Epoch | Train Loss | Val Loss | 更新 |
|---|---|---|---|
| 1 | 6.0117 | 5.7546 | ✅ Best |
| 2 | 5.7658 | 5.6995 | ✅ Best |
| 3 | 5.3485 | 4.9691 | ✅ Best |
| 4 | 5.0241 | 4.8083 | ✅ Best |
| 5 | 4.9384 | 4.8302 | ❌ |
| 6 | 4.9009 | **4.7619** | ✅ Best |
| 7〜 | 進行中 | — | — |

**現在の Best**: val_loss = **4.7619**（Epoch 6 / 2026-03-20 時点）

---

## 11. 重要な技術的知見

### トラブルシューティング実績

| 問題 | 原因 | 解決策 |
|---|---|---|
| 学習プラトー（val ~7.22） | データ量不足（WikiText-2） | WikiText-103 に切り替え |
| 第二のプラトー（val ~7.19） | ハードラベル only + T=100 | 全位置ロス + T=25 + LR=5e-4 |
| VRAM 溢れ → WSL2 クラッシュ | 共有 GPU メモリに 3 GB 溢れ | batch_size 64→48, T=100→25 |
| Gradient Checkpointing 逆効果 | 再計算コスト > VRAM 節約 | 無効化（速度 29% 低下を回避） |
| WSL2 SSH 切断 | C ドライブ容量不足 / ネットワーク切断 | 容量確保 + スリープ無効化 |
| `nvidia-smi` 未検出 | WSL2 特有のパス問題 | `/usr/lib/wsl/lib/nvidia-smi` を使う |
| ログがバイナリ化 | tmux の ANSI コード混入 | `strings` コマンドでフィルタ |
| Discord 通知が来ない Epoch がある | `notify-every` のデフォルトが 10 | `--notify-every 1` で起動する |

### ANE 最適化の知見

- 13M パラメータ → ANE < CPU（データ転送オーバーヘッドが計算量を上回る）
- 55M パラメータ → **ANE が CPU の 2.5 倍速**（閾値を超えた）
- CoreML 変換: `nn.Linear → nn.Conv2d` が ANE 最適化の鍵
- INT4 量子化でさらに高速化可能（未実施）

---

## 12. 今後の実施予定

| 優先度 | タスク | 状態 |
|---|---|---|
| 🔴 高 | v3 学習の完走・結果確認 | **進行中**（Epoch 6 / 30） |
| 🟡 中 | v3 完了後ベンチマーク更新・生成品質テスト | 未着手 |
| 🟡 中 | `distill/generate_logits.py`（Qwen ロジット事前生成） | 未着手 |
| 🟡 中 | `distill/vocab_mapping.py`（Qwen ↔ BERT 語彙マッピング） | 未着手 |
| 🟡 中 | `distill/train_distill.py`（蒸留学習スクリプト） | 未着手 |
| 🟢 低 | Discord ログ → Qwen 拡張パイプラインの実装 | 未着手 |
| 🟢 低 | CoreML 変換・ANE ベンチマーク更新（蒸留後） | 未着手 |

---

## 13. 重要ドキュメント一覧

| ファイル | 内容 |
|---|---|
| `markdown/handover.md` | **本ファイル**。引き継ぎ・チェックポイント |
| `markdown/diffusion_llm_ane_plan.md` | 実装計画書（Phase 1〜5） |
| `markdown/distillation_plan.md` | Qwen 3.5 9B → RinGo-DLLM 蒸留計画書（Discord ログ活用も記載） |
| `markdown/dataset-plan.md` | 友人提供のデータセット戦略（**非公開・.gitignore 済み**） |
| `README.md` | プロジェクト概要・アーキテクチャ・ベンチマーク結果 |

---

## 14. コミット履歴（主要なもの）

```
b383bb9 docs: 蒸留計画書に Discord チャットログ活用のパイプラインを追記
c2d5852 docs: 引き継ぎ書を追加 + dataset-plan.md を .gitignore に追加
e18ef48 docs: Qwen 3.5 9B → RinGo-DLLM 55M のロジット蒸留計画書を追加
f501d0a perf: 全位置ロス・T=25・LR=5e-4 で学習効率を大幅改善
e3be98b perf: Gradient Checkpointing を追加（後に無効化）
ad16982 docs: プロジェクト全体の README を追加
427ae5d perf: Flash Attention + AMP(BF16) + torch.compile + バッチサイズ 2 倍
406813d feat: WikiText-103 対応・Cosine LR decay・Early stopping・sample.py バグ修正
69e2217 feat: --config large オプションで 55M パラメータモデルをサポート
```
