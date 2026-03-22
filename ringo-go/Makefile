# ──────────────────────────────────────────────────────────────────────────────
# RinGo-DLLM Go Inference Server
# ──────────────────────────────────────────────────────────────────────────────

BINARY   := ringo
MODELS   := ../diffusion_llm_ane/convert
VOCAB    := ./vocab.txt
ADDR     := :8080

FP16 := $(MODELS)/diffusion_lm_ane_v3.mlpackage
INT8 := $(MODELS)/diffusion_lm_ane_v3_int8.mlpackage
INT4 := $(MODELS)/diffusion_lm_ane_v3_int4.mlpackage

.PHONY: build run clean help

## build: バイナリをビルドする
build:
	go build -o $(BINARY) ./cmd/ringo/

## run: ビルドしてサーバーを起動する（ブラウザは自動で開く）
run: build
	@echo "Starting RinGo server at http://localhost$(ADDR) ..."
	@open "http://localhost$(ADDR)" &
	./$(BINARY) \
		--vocab    $(VOCAB) \
		--model-fp16 $(FP16) \
		--model-int8 $(INT8) \
		--model-int4 $(INT4) \
		--addr     $(ADDR)

## dev: ビルドせずそのまま起動（go run）
dev:
	@open "http://localhost$(ADDR)" &
	go run ./cmd/ringo/ \
		--vocab    $(VOCAB) \
		--model-fp16 $(FP16) \
		--model-int8 $(INT8) \
		--model-int4 $(INT4) \
		--addr     $(ADDR)

## clean: バイナリを削除する
clean:
	rm -f $(BINARY)

## help: このヘルプを表示する
help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
