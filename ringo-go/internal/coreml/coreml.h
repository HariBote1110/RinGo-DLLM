#ifndef COREML_H
#define COREML_H

#include <stdint.h>

/* Opaque handle to a loaded MLModel instance. */
typedef void* ModelHandle;

/*
 * Load an MLModel from the given .mlpackage path using all compute units.
 * Returns NULL on failure.
 */
ModelHandle coreml_load(const char* mlpackage_path);

/*
 * Release all resources associated with a model handle.
 */
void coreml_free(ModelHandle h);

/*
 * Run inference on the model.
 *
 * Parameters:
 *   h           - Model handle obtained from coreml_load.
 *   input_ids   - Flat int32 array of shape [1, seq_len].
 *   seq_len     - Sequence length (number of tokens).
 *   t_val       - Timestep scalar (int32).
 *   logits_out  - Pre-allocated float buffer of size seq_len * vocab_size.
 *   vocab_size  - Vocabulary size.
 *
 * Returns 0 on success, -1 on failure.
 */
int coreml_predict(ModelHandle h,
                   const int32_t* input_ids,
                   int seq_len,
                   int32_t t_val,
                   float* logits_out,
                   int vocab_size);

#endif /* COREML_H */
