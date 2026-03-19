# 引き継ぎ書: RinGo-DLLM 開発チェックポイント

**作成日**: 2026-03-19
**最終コミット**: `e18ef48` (docs: Qwen 3.5 9B → RinGo-DLLM 55M のロジット蒸留計画書を追加)

---

## 1. プロジェクト概要

**RinGo-DLLM** は Masked Diffusion Language Model（MDLM）ベースの 55M パラメータ言語モデル。
Apple Neural Engine（ANE）での高速推論を目指し、CoreML 変換にも対応している。

- **学習**: RTX 3070 Ti（8GB VRAM / Windows PC / WSL2）
- **推論ベンチ**: Mac mini M4（ANE / CoreML）
- **リモート操作**: Mac mini → SSH → Windows PC（WSL2）

---

## 2. 現在の状態

### 学習 v2 が RTX 3070 Ti で進行中

| 設定 | 値 |
|---|---|
| 設定ファイル | `config_large.py`（55M パラメータ） |
| データセット | WikiText-103 |
| 拡散ステップ T | 25 |
| マスクスケジュール | cosine |
| 学習率 | 5e-4（cosine decay → 1e-5） |
| バッチサイズ | 48 |
| Warmup | 3,000 steps |
| AMP | BF16 |
| torch.compile | 有効 |
| 損失関数 | 全位置ロス（mask_loss_weight=5.0） |
| Gradient Checkpointing | **無効**（逆効果だったため） |

### 学習結果（v2）

| Epoch | Train Loss | Val Loss | 時間 | 備考 |
|---|---|---|---|---|
| 1 | 5.9798 | 5.7795 | ~2400s | Best saved |
| 2 | 5.4782 | 5.0315 | ~2400s | Best saved |
| 3+ | 進行中 | — | — | tmux `wt103` セッション |

### 過去の学習結果（v1: 全位置ロス導入前）

| Epoch | Train Loss | Val Loss | 備考 |
|---|---|---|---|
| 1 | 7.2340 | 7.2157 | |
| 2 | 7.2299 | 7.1882 | |
| 3 | 7.2223 | 7.1930 | プラトー |
| 4 | 7.2113 | 7.1865 | 改善微小 |

→ v2 で **val_loss 7.19 → 5.03** と大幅改善。

---

## 3. SSH 接続情報

```bash
# Mac mini → Windows PC (WSL2)
ssh -p 2222 -i ~/.ssh/id_ed25519 gzabu@192.168.0.30

# WSL2 が落ちた場合（Windows 側で実行）
wsl --shutdown
wsl
# WSL2 内で
sudo service ssh start
```

### 学習セッション確認

```bash
# tmux セッション接続
tmux attach -t wt103

# ログ確認（ANSI コード混入の可能性あり）
strings ~/RinGo-DLLM/diffusion_llm_ane/train_wt103_v2.log | tail -50

# GPU 使用状況
/usr/lib/wsl/lib/nvidia-smi
```

### Discord Webhook 通知

- `.env` ファイルに `DISCORD_WEBHOOK_URL` を保存済み
- 学習スクリプトが自動的に読み込む
- エポック完了時・ベスト更新時・学習完了時に通知

---

## 4. ファイル構成

```
RinGo-DLLM/
├── .gitignore                    # checkpoints + dataset-plan.md を除外
├── README.md                     # プロジェクト全体ドキュメント
├── markdown/
│   ├── diffusion_llm_ane_plan.md # 実装計画書
│   ├── distillation_plan.md      # 蒸留計画書（Qwen 3.5 9B → 55M）
│   ├── dataset-plan.md           # 友人提供（非公開、.gitignore 済み）
│   └── handover.md               # ← 本ファイル
└── diffusion_llm_ane/
    ├── model/
    │   ├── config.py             # ベースモデル設定（30K語彙, 256dim, 6層）
    │   ├── config_large.py       # 55M モデル設定（512dim, 12層, T=25）
    │   ├── diffusion_lm.py       # MDLM 本体
    │   ├── diffusion_lm_ane.py   # ANE 最適化版（Conv2d 変換）
    │   ├── transformer.py        # Transformer（Flash Attention 対応）
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
    ├── train.py                  # 学習スクリプト（AMP, compile, 全位置ロス対応）
    ├── sample.py                 # テキスト生成（逆拡散プロセス）
    ├── notify.py                 # Discord Webhook 通知
    ├── sanity_check.py           # モデル動作確認
    └── requirements.txt
```

---

## 5. 重要な技術的知見

### 学んだこと（トラブルシューティング）

| 問題 | 原因 | 解決策 |
|---|---|---|
| 学習プラトー（val_loss ~7.22） | データ量不足（WikiText-2） | WikiText-103 に切り替え |
| 第二のプラトー（val_loss ~7.19） | ハードラベル only + T=100 | 全位置ロス + T=25 + LR=5e-4 |
| VRAM 溢れ → WSL2 クラッシュ | 共有 GPU メモリに 3GB 溢れ | batch_size 64→48, T=100→25 |
| Gradient Checkpointing 逆効果 | 再計算コスト > VRAM 節約 | 無効化（速度 29% 低下を回避） |
| WSL2 SSH 切断 | C ドライブ容量不足 / ネットワーク切断 | 容量確保 + スリープ無効化 |
| nvidia-smi 未検出 | WSL2 特有のパス | `/usr/lib/wsl/lib/nvidia-smi` |
| ログがバイナリ化 | tmux の ANSI コード混入 | `strings` でフィルタ |

### ANE 最適化の知見

- 13M パラメータ → ANE < CPU（データ転送オーバーヘッド > 計算量）
- 55M パラメータ → **ANE が CPU の 2.5 倍速**（閾値を超えた）
- CoreML 変換: `nn.Linear → nn.Conv2d` が ANE 最適化の鍵
- INT4 量子化でさらに高速化可能

---

## 6. 未完了タスク

### 即時対応

- [ ] 学習 v2 の完了を待ち、結果を確認する
- [ ] 学習完了後のベンチマーク更新（CoreML / ANE）
- [ ] `sample.py` での生成品質テスト（v2 モデル）
- [ ] `.gitignore` の変更をコミットする

### 中期（蒸留）

- [ ] `distill/vocab_mapping.py` — Qwen ↔ BERT 語彙マッピング
- [ ] `distill/generate_logits.py` — Qwen 3.5 9B のロジット事前生成
- [ ] `distill/train_distill.py` — 蒸留学習スクリプト
- [ ] カバレッジ 80% 以下ならデータ蒸留にフォールバック

### 長期

- [ ] 会話データでの蒸留（OpenAssistant / UltraChat）
- [ ] 日本語対応（`dataset-plan.md` の知見を活用）
- [ ] マルチ教師蒸留（Qwen + Phi-3 + Llama）

---

## 7. 学習再開手順

学習が中断した場合の復旧手順:

```bash
# 1. SSH 接続
ssh -p 2222 -i ~/.ssh/id_ed25519 gzabu@192.168.0.30

# 2. venv 有効化
source ~/RinGo-DLLM/diffusion_llm_ane/.venv/bin/activate

# 3. .env 読み込み（Discord Webhook）
export $(grep -v '^#' ~/RinGo-DLLM/diffusion_llm_ane/.env | xargs)

# 4. tmux セッション作成 + 学習開始
tmux new -s wt103
cd ~/RinGo-DLLM/diffusion_llm_ane
PYTHONUNBUFFERED=1 python3 train.py \
  --config large \
  --resume checkpoints_wt103/best.pt \
  --no-compile \
  --webhook "$DISCORD_WEBHOOK_URL" \
  --notify-every 1 \
  2>&1 | tee -a train_wt103_v2.log

# 5. tmux からデタッチ: Ctrl+B → D
```

※ `--no-compile` は初回以降は外してよい（compile キャッシュが効く）。
ただし、チェックポイントからの再開時は `--no-compile` を付けた方が安全
（`_orig_mod` キー衝突を回避）。

---

## 8. コミット履歴（重要なもの）

```
e18ef48 docs: 蒸留計画書を追加
f501d0a perf: 全位置ロス・T=25・LR=5e-4 で学習効率を大幅改善
e3be98b perf: Gradient Checkpointing を追加
ad16982 docs: README を追加
427ae5d perf: Flash Attention + AMP + torch.compile + バッチサイズ 2 倍
406813d feat: WikiText-103 対応・Cosine LR decay・Early stopping
69e2217 feat: 55M パラメータモデルのサポート
```

---

## 9. 参考ドキュメント

| ドキュメント | 場所 | 内容 |
|---|---|---|
| 実装計画書 | `markdown/diffusion_llm_ane_plan.md` | Phase 1〜5 の全体計画 |
| 蒸留計画書 | `markdown/distillation_plan.md` | Qwen → RinGo-DLLM ロジット蒸留 |
| データセット戦略 | `markdown/dataset-plan.md` | 友人提供・非公開 |
| README | `README.md` | プロジェクト概要・ベンチマーク |
