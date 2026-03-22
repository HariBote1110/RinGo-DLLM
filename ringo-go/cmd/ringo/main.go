// Command ringo starts an HTTP server that exposes a web UI and a streaming
// generation endpoint for a masked-diffusion language model running via CoreML.
package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"time"

	"github.com/HariBote1110/RinGo-DLLM/ringo-go/internal/coreml"
	"github.com/HariBote1110/RinGo-DLLM/ringo-go/internal/diffusion"
	"github.com/HariBote1110/RinGo-DLLM/ringo-go/internal/tokenizer"
)

//go:embed web/index.html
var indexHTML []byte

// coremlPredictor wraps *coreml.Model to satisfy the diffusion.Predictor
// interface.  coreml.Model exposes SeqLen and VocabSize as fields; the
// interface expects them as methods.
type coremlPredictor struct{ m *coreml.Model }

func (p *coremlPredictor) Predict(inputIDs []int32, tVal int32) ([][]float32, error) {
	return p.m.Predict(inputIDs, tVal)
}
func (p *coremlPredictor) SeqLen() int    { return p.m.SeqLen }
func (p *coremlPredictor) VocabSize() int { return p.m.VocabSize }

// generateRequest is the JSON body accepted by POST /generate.
type generateRequest struct {
	Prompt            string  `json:"prompt"`
	Backend           string  `json:"backend"`
	Steps             int     `json:"steps"`
	Temperature       float64 `json:"temperature"`
	TopK              int     `json:"top_k"`
	TopP              float64 `json:"top_p"`
	RepetitionPenalty float64 `json:"repetition_penalty"`
	Seed              int64   `json:"seed"`
}

// sseEvent is the JSON payload sent in each SSE data line.
type sseEvent struct {
	// Embed all StepResult fields inline.
	Step            int     `json:"step"`
	NSteps          int     `json:"n_steps"`
	Tokens          []int   `json:"tokens"`
	MaskRemaining   int     `json:"mask_remaining"`
	TotalMask       int     `json:"total_mask"`
	NewlyRevealed   []int   `json:"newly_revealed"`
	PromptPositions []int   `json:"prompt_positions"`
	StepMs          float64 `json:"step_ms"`
	TotalMs         float64 `json:"total_ms"`
	// Additional fields for the UI.
	Words []string `json:"words"`
	Done  bool     `json:"done"`
}

// server holds shared resources used across HTTP handlers.
type server struct {
	tok      *tokenizer.Tokenizer
	modelFP16 *coreml.Model
	modelINT8 *coreml.Model
	modelINT4 *coreml.Model
	cfg      diffusion.Config
}

func main() {
	// --- CLI flags ---
	vocabPath    := flag.String("vocab",      "./vocab.txt", "path to vocab.txt")
	modelFP16Path := flag.String("model-fp16", "",           "path to FP16 .mlpackage")
	modelINT8Path := flag.String("model-int8", "",           "path to INT8 .mlpackage")
	modelINT4Path := flag.String("model-int4", "",           "path to INT4 .mlpackage")
	addr         := flag.String("addr",       ":8080",       "HTTP listen address")
	seqLen       := flag.Int("seq-len",       128,           "model sequence length")
	vocabSize    := flag.Int("vocab-size",    30522,         "model vocabulary size")
	maskID       := flag.Int("mask-id",       103,           "token ID for [MASK]")
	padID        := flag.Int("pad-id",        0,             "token ID for [PAD]")
	T            := flag.Int("T",             25,            "model maximum timestep")
	flag.Parse()

	// --- Load tokeniser ---
	tok, err := tokenizer.Load(*vocabPath)
	if err != nil {
		log.Fatalf("failed to load tokeniser: %v", err)
	}
	log.Printf("tokeniser loaded from %s", *vocabPath)

	// --- Load models (failures are non-fatal; log a warning and skip) ---
	srv := &server{
		tok: tok,
		cfg: diffusion.Config{
			MaskID: *maskID,
			PadID:  *padID,
			T:      *T,
		},
	}

	loadModel := func(path, name string) *coreml.Model {
		if path == "" {
			return nil
		}
		m, err := coreml.Load(path, *seqLen, *vocabSize)
		if err != nil {
			log.Printf("WARNING: could not load %s model from %q: %v", name, path, err)
			return nil
		}
		log.Printf("%s model loaded from %s (seqLen=%d, vocabSize=%d)", name, path, *seqLen, *vocabSize)
		return m
	}

	srv.modelFP16 = loadModel(*modelFP16Path, "FP16")
	srv.modelINT8 = loadModel(*modelINT8Path, "INT8")
	srv.modelINT4 = loadModel(*modelINT4Path, "INT4")

	// --- HTTP routes ---
	mux := http.NewServeMux()
	mux.HandleFunc("/",         srv.handleIndex)
	mux.HandleFunc("/generate", srv.handleGenerate)

	log.Printf("listening on %s", *addr)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// handleIndex serves the embedded web UI.
func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	log.Printf("%s %s", r.Method, r.URL.Path)
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(indexHTML)
}

// handleGenerate accepts a JSON POST body and streams the generation as SSE.
func (s *server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// --- Decode request body ---
	var req generateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("POST /generate — invalid JSON: %v", err)
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	log.Printf("POST /generate — backend=%q steps=%d prompt=%q", req.Backend, req.Steps, req.Prompt)

	// --- Select model ---
	model := s.modelForBackend(req.Backend)
	if model == nil {
		log.Printf("POST /generate — model unavailable for backend %q", req.Backend)
		http.Error(w, fmt.Sprintf("model not loaded for backend %q", req.Backend), http.StatusServiceUnavailable)
		return
	}

	// --- Obtain a Flusher for SSE ---
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// --- SSE headers ---
	w.Header().Set("Content-Type",  "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection",    "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	// --- Resolve seed ---
	seed := req.Seed
	if seed < 0 {
		seed = time.Now().UnixNano()
	}

	params := diffusion.Params{
		NSteps:            req.Steps,
		Temperature:       req.Temperature,
		TopK:              req.TopK,
		TopP:              req.TopP,
		RepetitionPenalty: req.RepetitionPenalty,
		Seed:              seed,
	}
	if params.NSteps <= 0 {
		params.NSteps = 25
	}

	// send is a helper that serialises evt as an SSE data line and flushes.
	send := func(evt sseEvent) {
		b, err := json.Marshal(evt)
		if err != nil {
			log.Printf("POST /generate — json.Marshal error: %v", err)
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	// buildWords converts a token-ID slice to their string representations.
	buildWords := func(tokens []int) []string {
		words := make([]string, len(tokens))
		for i, id := range tokens {
			words[i] = s.tok.DecodeOne(id)
		}
		return words
	}

	// Seed the global source (rand.New per-request via diffusion.Params.Seed)
	_ = rand.New(rand.NewSource(seed)) // kept for documentation clarity

	var lastResult diffusion.StepResult

	// --- Run the diffusion generation loop ---
	err := diffusion.Generate(
		&coremlPredictor{model},
		s.tok,
		s.cfg,
		params,
		req.Prompt,
		func(result diffusion.StepResult) {
			lastResult = result
			evt := sseEvent{
				Step:            result.Step,
				NSteps:          result.NSteps,
				Tokens:          result.Tokens,
				MaskRemaining:   result.MaskRemaining,
				TotalMask:       result.TotalMask,
				NewlyRevealed:   result.NewlyRevealed,
				PromptPositions: result.PromptPositions,
				StepMs:          result.StepMs,
				TotalMs:         result.TotalMs,
				Words:           buildWords(result.Tokens),
				Done:            false,
			}
			send(evt)
		},
	)
	if err != nil {
		log.Printf("POST /generate — generation error: %v", err)
		// Send an error event so the client knows something went wrong.
		errEvt := sseEvent{Done: true}
		send(errEvt)
		return
	}

	// --- Send the final "done" event ---
	doneEvt := sseEvent{
		Step:            lastResult.Step,
		NSteps:          lastResult.NSteps,
		Tokens:          lastResult.Tokens,
		MaskRemaining:   lastResult.MaskRemaining,
		TotalMask:       lastResult.TotalMask,
		NewlyRevealed:   lastResult.NewlyRevealed,
		PromptPositions: lastResult.PromptPositions,
		StepMs:          lastResult.StepMs,
		TotalMs:         lastResult.TotalMs,
		Words:           buildWords(lastResult.Tokens),
		Done:            true,
	}
	send(doneEvt)

	elapsed := time.Since(start)
	log.Printf("POST /generate — completed in %.0f ms", float64(elapsed.Microseconds())/1000.0)
}

// modelForBackend returns the coreml.Model that corresponds to the backend
// name sent by the client.  Returns nil when the model was not loaded.
func (s *server) modelForBackend(backend string) *coreml.Model {
	switch backend {
	case "CoreML ANE (FP16)":
		return s.modelFP16
	case "CoreML ANE (INT8)":
		return s.modelINT8
	case "CoreML ANE (INT4)":
		return s.modelINT4
	default:
		return s.modelFP16 // fall back to FP16
	}
}
