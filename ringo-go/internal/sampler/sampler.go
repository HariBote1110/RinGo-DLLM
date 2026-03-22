// Package sampler provides token-sampling utilities that mirror the logic in
// the Python sample.py companion script.
package sampler

import (
	"math"
	"math/rand"
	"sort"
)

// ApplyRepetitionPenalty applies a HuggingFace-style repetition penalty in place on a
// copy of logits.  For each token ID that has already been generated:
//   - positive logits are divided by penalty
//   - negative logits are multiplied by penalty
//
// The original slice is never mutated.
func ApplyRepetitionPenalty(logits []float32, generatedIDs []int, penalty float64) []float32 {
	out := copyFloat32(logits)
	if penalty == 1.0 || len(generatedIDs) == 0 {
		return out
	}
	pen := float32(penalty)
	for _, id := range generatedIDs {
		if id < 0 || id >= len(out) {
			continue
		}
		if out[id] > 0 {
			out[id] /= pen
		} else {
			out[id] *= pen
		}
	}
	return out
}

// ApplyTopK zeroes out (sets to -Inf) every logit that is not among the top-k
// highest values.  A copy is returned; the original slice is not mutated.
func ApplyTopK(logits []float32, k int) []float32 {
	out := copyFloat32(logits)
	if k <= 0 || k >= len(out) {
		return out
	}

	// Build an index sorted by logit value (descending).
	indices := make([]int, len(out))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(a, b int) bool {
		return out[indices[a]] > out[indices[b]]
	})

	negInf := float32(math.Inf(-1))
	for _, idx := range indices[k:] {
		out[idx] = negInf
	}
	return out
}

// ApplyTopP performs nucleus filtering (top-p / top-p sampling).  Tokens are
// sorted by probability (derived from logits via temperature-scaled softmax);
// the lowest-probability tokens whose cumulative probability exceeds p are
// masked out with -Inf.  A copy is returned; the original slice is not mutated.
func ApplyTopP(logits []float32, p float64, temperature float64) []float32 {
	out := copyFloat32(logits)
	if p >= 1.0 {
		return out
	}

	probs := Softmax(out, temperature)

	// Sort indices by probability (descending).
	indices := make([]int, len(probs))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(a, b int) bool {
		return probs[indices[a]] > probs[indices[b]]
	})

	// Accumulate probabilities and mask tokens once the threshold is exceeded.
	cumSum := float64(0)
	negInf := float32(math.Inf(-1))
	for _, idx := range indices {
		cumSum += float64(probs[idx])
		if cumSum > p {
			out[idx] = negInf
		}
	}
	return out
}

// Softmax computes a temperature-scaled softmax over logits.
// A copy of the probabilities is returned; the input is not mutated.
func Softmax(logits []float32, temperature float64) []float32 {
	if len(logits) == 0 {
		return nil
	}

	temp := float32(temperature)
	if temp <= 0 {
		temp = 1.0
	}

	// Find max for numerical stability.
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	probs := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		e := float32(math.Exp(float64((v - maxVal) / temp)))
		probs[i] = e
		sum += e
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

// Multinomial draws a single sample from a discrete probability distribution.
// If rng is nil, the global random source is used.
func Multinomial(probs []float32, rng *rand.Rand) int {
	var u float64
	if rng != nil {
		u = rng.Float64()
	} else {
		u = rand.Float64()
	}

	cumSum := float64(0)
	for i, p := range probs {
		cumSum += float64(p)
		if u < cumSum {
			return i
		}
	}
	// Fallback: return the last non-zero entry (handles floating-point rounding).
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return i
		}
	}
	return 0
}

// SampleToken applies the full sampling pipeline in sequence:
//  1. Repetition penalty
//  2. Top-k filtering
//  3. Top-p (nucleus) filtering
//  4. Softmax
//  5. Multinomial sampling
func SampleToken(
	logits []float32,
	temperature float64,
	topK int,
	topP float64,
	penalty float64,
	generatedIDs []int,
	rng *rand.Rand,
) int {
	l := ApplyRepetitionPenalty(logits, generatedIDs, penalty)
	l = ApplyTopK(l, topK)
	l = ApplyTopP(l, topP, temperature)
	probs := Softmax(l, temperature)
	return Multinomial(probs, rng)
}

// --- internal helpers ---

func copyFloat32(src []float32) []float32 {
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}
