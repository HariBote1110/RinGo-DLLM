// Package diffusion implements the reverse-diffusion generation loop for a
// masked-diffusion language model running via a CoreML Predictor.
package diffusion

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/HariBote1110/RinGo-DLLM/ringo-go/internal/sampler"
	"github.com/HariBote1110/RinGo-DLLM/ringo-go/internal/tokenizer"
)

// Predictor is the interface that a CoreML (or any other backend) model must
// satisfy in order to be used by Generate.
type Predictor interface {
	// Predict runs a single forward pass.  inputIDs has length SeqLen() and
	// tVal encodes the current noise level.  The returned slice has shape
	// [SeqLen()][VocabSize()].
	Predict(inputIDs []int32, tVal int32) ([][]float32, error)

	// SeqLen returns the fixed sequence length expected by the model.
	SeqLen() int

	// VocabSize returns the vocabulary size of the output logits.
	VocabSize() int
}

// Config holds the static configuration that is tied to the model itself.
type Config struct {
	// MaskID is the token ID used to represent a masked position.
	MaskID int
	// PadID is the token ID used to pad sequences shorter than SeqLen.
	PadID int
	// T is the total number of noise steps the model was trained with
	// (default 25).
	T int
}

// Params holds the generation hyper-parameters chosen per call.
type Params struct {
	NSteps            int
	Temperature       float64
	TopK              int
	TopP              float64
	RepetitionPenalty float64
	Seed              int64
}

// StepResult is the value produced by the yield callback after every
// reverse-diffusion step.
type StepResult struct {
	// Step is the current step index (counts down from NSteps to 1).
	Step int
	// NSteps is the total number of requested steps.
	NSteps int
	// Tokens is the current token sequence (length == model.SeqLen()).
	Tokens []int
	// MaskRemaining is the number of positions still masked.
	MaskRemaining int
	// TotalMask is the number of positions that were masked at the start.
	TotalMask int
	// NewlyRevealed lists the positions that were unmasked in this step.
	NewlyRevealed []int
	// PromptPositions lists the positions that were provided by the prompt.
	PromptPositions []int
	// StepMs is the wall-clock time spent on this step in milliseconds.
	StepMs float64
	// TotalMs is the cumulative wall-clock time in milliseconds.
	TotalMs float64
}

// Generate runs reverse diffusion over model and calls yield once per step.
// If promptText is empty, every position is initialised with MASK.
// Otherwise the prompt is encoded and the remaining positions are padded.
func Generate(
	model Predictor,
	tok *tokenizer.Tokenizer,
	cfg Config,
	params Params,
	promptText string,
	yield func(StepResult),
) error {
	if params.NSteps <= 0 {
		return fmt.Errorf("diffusion: NSteps must be > 0, got %d", params.NSteps)
	}
	if cfg.T <= 0 {
		cfg.T = 25
	}

	seqLen := model.SeqLen()
	rng := rand.New(rand.NewSource(params.Seed))

	// --- Initialise the token sequence ---
	tokens := make([]int, seqLen)
	promptPositions := []int{}

	if promptText == "" {
		for i := range tokens {
			tokens[i] = cfg.MaskID
		}
	} else {
		encoded := tok.Encode(promptText)
		for i := range tokens {
			tokens[i] = cfg.MaskID
		}
		for i, id := range encoded {
			if i >= seqLen {
				break
			}
			tokens[i] = id
			promptPositions = append(promptPositions, i)
		}
	}

	// Count positions that were unmasked from the start.
	initialUnmasked := 0
	for _, id := range tokens {
		if id != cfg.MaskID {
			initialUnmasked++
		}
	}
	totalMask := seqLen - initialUnmasked

	// generatedIDs accumulates all non-mask, non-pad token IDs for the
	// repetition-penalty calculation.
	generatedIDs := collectNonSpecial(tokens, cfg.MaskID, cfg.PadID)

	startTime := time.Now()

	// --- Reverse-diffusion loop ---
	for step := params.NSteps; step >= 1; step-- {
		stepStart := time.Now()

		tVal := int32(step * cfg.T / params.NSteps)

		// Build the int32 input slice expected by the model.
		inputIDs := make([]int32, seqLen)
		for i, id := range tokens {
			inputIDs[i] = int32(id)
		}

		logitsAll, err := model.Predict(inputIDs, tVal)
		if err != nil {
			return fmt.Errorf("diffusion: model.Predict at step %d: %w", step, err)
		}
		if len(logitsAll) != seqLen {
			return fmt.Errorf(
				"diffusion: expected logits length %d, got %d",
				seqLen, len(logitsAll),
			)
		}

		// Determine how many positions should be unmasked after this step.
		fractionDone := 1.0 - float64(step-1)/float64(params.NSteps)
		targetUnmasked := initialUnmasked + roundInt(float64(totalMask)*fractionDone)
		if targetUnmasked > seqLen {
			targetUnmasked = seqLen
		}

		currentUnmasked := seqLen - countMask(tokens, cfg.MaskID)
		toReveal := targetUnmasked - currentUnmasked
		if toReveal < 0 {
			toReveal = 0
		}

		newlyRevealed := unmaskStep(
			tokens,
			logitsAll,
			cfg.MaskID,
			toReveal,
			params,
			generatedIDs,
			rng,
		)

		// Update the accumulated generated-token set.
		for _, pos := range newlyRevealed {
			id := tokens[pos]
			if id != cfg.MaskID && id != cfg.PadID {
				generatedIDs = append(generatedIDs, id)
			}
		}

		maskRemaining := countMask(tokens, cfg.MaskID)
		stepMs := float64(time.Since(stepStart).Microseconds()) / 1000.0
		totalMs := float64(time.Since(startTime).Microseconds()) / 1000.0

		// Deliver a snapshot to the caller (copy tokens to avoid aliasing).
		tokensCopy := make([]int, seqLen)
		copy(tokensCopy, tokens)

		yield(StepResult{
			Step:            step,
			NSteps:          params.NSteps,
			Tokens:          tokensCopy,
			MaskRemaining:   maskRemaining,
			TotalMask:       totalMask,
			NewlyRevealed:   newlyRevealed,
			PromptPositions: promptPositions,
			StepMs:          stepMs,
			TotalMs:         totalMs,
		})
	}

	return nil
}

// unmaskStep selects the toReveal highest-confidence MASK positions, samples a
// token for each one, and updates tokens in place.  It returns the list of
// positions that were unmasked.
func unmaskStep(
	tokens []int,
	logitsAll [][]float32,
	maskID int,
	toReveal int,
	params Params,
	generatedIDs []int,
	rng *rand.Rand,
) []int {
	if toReveal <= 0 {
		return nil
	}

	// Gather all currently masked positions with their best-token confidence.
	type candidate struct {
		pos       int
		maxProb   float32
		logits    []float32
	}

	var candidates []candidate
	for pos, id := range tokens {
		if id != maskID {
			continue
		}
		if pos >= len(logitsAll) {
			continue
		}
		best := maxProb(logitsAll[pos])
		candidates = append(candidates, candidate{
			pos:     pos,
			maxProb: best,
			logits:  logitsAll[pos],
		})
	}

	// Sort by confidence, highest first.
	sort.Slice(candidates, func(a, b int) bool {
		return candidates[a].maxProb > candidates[b].maxProb
	})

	if toReveal > len(candidates) {
		toReveal = len(candidates)
	}

	revealed := make([]int, 0, toReveal)
	for _, c := range candidates[:toReveal] {
		sampledID := sampler.SampleToken(
			c.logits,
			params.Temperature,
			params.TopK,
			params.TopP,
			params.RepetitionPenalty,
			generatedIDs,
			rng,
		)
		tokens[c.pos] = sampledID
		revealed = append(revealed, c.pos)
	}
	return revealed
}

// --- internal helpers ---

// maxProb returns the maximum value in a logits slice (used as a confidence
// proxy without computing a full softmax).
func maxProb(logits []float32) float32 {
	if len(logits) == 0 {
		return 0
	}
	m := logits[0]
	for _, v := range logits[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// countMask returns the number of positions equal to maskID.
func countMask(tokens []int, maskID int) int {
	n := 0
	for _, id := range tokens {
		if id == maskID {
			n++
		}
	}
	return n
}

// collectNonSpecial gathers token IDs that are neither maskID nor padID.
func collectNonSpecial(tokens []int, maskID, padID int) []int {
	var out []int
	for _, id := range tokens {
		if id != maskID && id != padID {
			out = append(out, id)
		}
	}
	return out
}

// roundInt rounds a float64 to the nearest integer.
func roundInt(x float64) int {
	return int(math.Round(x))
}
