// Package coreml provides CGO bindings for Apple CoreML inference.
// It wraps an MLModel loaded from a .mlpackage file and exposes a
// simple Predict interface for diffusion language model inference.
package coreml

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreML -framework Foundation
#include "coreml.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Model holds an opaque reference to a loaded CoreML MLModel along with
// the static dimensions required for memory allocation during inference.
type Model struct {
	handle    C.ModelHandle
	SeqLen    int
	VocabSize int
}

// Load opens the .mlpackage at the given path and initialises the model
// using all available compute units (CPU, GPU, ANE).
// seqLen and vocabSize must match the fixed input/output shapes baked
// into the compiled model.
func Load(path string, seqLen, vocabSize int) (*Model, error) {
	if seqLen <= 0 {
		return nil, fmt.Errorf("coreml: seqLen must be positive, got %d", seqLen)
	}
	if vocabSize <= 0 {
		return nil, fmt.Errorf("coreml: vocabSize must be positive, got %d", vocabSize)
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.coreml_load(cPath)
	if handle == nil {
		return nil, fmt.Errorf("coreml: failed to load model from %q", path)
	}

	return &Model{
		handle:    handle,
		SeqLen:    seqLen,
		VocabSize: vocabSize,
	}, nil
}

// Close releases the underlying MLModel and all associated CoreML resources.
// It is safe to call Close multiple times; subsequent calls are no-ops.
func (m *Model) Close() {
	if m.handle != nil {
		C.coreml_free(m.handle)
		m.handle = nil
	}
}

// Predict runs a single forward pass through the model.
//
// inputIDs must have exactly SeqLen elements.
// tVal is the diffusion timestep scalar passed as the "t" input feature.
//
// The returned slice has shape [SeqLen][VocabSize]; each inner slice
// contains the logits for the corresponding token position.
func (m *Model) Predict(inputIDs []int32, tVal int32) ([][]float32, error) {
	if m.handle == nil {
		return nil, fmt.Errorf("coreml: model has been closed")
	}
	if len(inputIDs) != m.SeqLen {
		return nil, fmt.Errorf("coreml: inputIDs length %d does not match SeqLen %d",
			len(inputIDs), m.SeqLen)
	}

	/* Allocate a flat buffer for the output logits. */
	logitsFlat := make([]float32, m.SeqLen*m.VocabSize)

	ret := C.coreml_predict(
		m.handle,
		(*C.int32_t)(unsafe.Pointer(&inputIDs[0])),
		C.int(m.SeqLen),
		C.int32_t(tVal),
		(*C.float)(unsafe.Pointer(&logitsFlat[0])),
		C.int(m.VocabSize),
	)
	if ret != 0 {
		return nil, fmt.Errorf("coreml: prediction failed (coreml_predict returned %d)", int(ret))
	}

	/* Reshape the flat buffer into a 2-D slice [SeqLen][VocabSize]. */
	logits2D := make([][]float32, m.SeqLen)
	for i := range logits2D {
		start := i * m.VocabSize
		logits2D[i] = logitsFlat[start : start+m.VocabSize]
	}

	return logits2D, nil
}
