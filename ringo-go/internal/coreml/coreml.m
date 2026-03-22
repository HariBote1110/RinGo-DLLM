#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#include <string.h>
#include "coreml.h"

ModelHandle coreml_load(const char* mlpackage_path) {
    @autoreleasepool {
        NSString* pathStr = [NSString stringWithUTF8String:mlpackage_path];
        NSURL* srcURL = [NSURL fileURLWithPath:pathStr];

        /* Compile the .mlpackage to a temporary .mlmodelc before loading.
           coremltools does this automatically; we must do it explicitly here. */
        NSError* error = nil;
        NSURL* compiledURL = [MLModel compileModelAtURL:srcURL error:&error];
        if (compiledURL == nil || error != nil) {
            NSLog(@"[coreml_load] Failed to compile model: %@", error.localizedDescription);
            return NULL;
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;

        MLModel* model = [MLModel modelWithContentsOfURL:compiledURL
                                           configuration:config
                                                   error:&error];
        if (error != nil || model == nil) {
            NSLog(@"[coreml_load] Failed to load compiled model: %@", error.localizedDescription);
            return NULL;
        }

        /* Retain the model across the autorelease pool boundary. */
        return (ModelHandle)CFBridgingRetain(model);
    }
}

void coreml_free(ModelHandle h) {
    if (h != NULL) {
        CFRelease(h);
    }
}

int coreml_predict(ModelHandle h,
                   const int32_t* input_ids,
                   int seq_len,
                   int32_t t_val,
                   float* logits_out,
                   int vocab_size) {
    @autoreleasepool {
        MLModel* model = (__bridge MLModel*)h;

        NSError* error = nil;

        /* ── Build input_ids MLMultiArray of shape [1, seq_len] ── */
        NSArray<NSNumber*>* inputShape = @[@1, @(seq_len)];
        MLMultiArray* inputArray =
            [[MLMultiArray alloc] initWithShape:inputShape
                                      dataType:MLMultiArrayDataTypeInt32
                                         error:&error];
        if (error != nil || inputArray == nil) {
            NSLog(@"[coreml_predict] Failed to allocate input_ids array: %@",
                  error.localizedDescription);
            return -1;
        }
        memcpy(inputArray.dataPointer, input_ids,
               (size_t)seq_len * sizeof(int32_t));

        /* ── Build t MLMultiArray of shape [1] ── */
        NSArray<NSNumber*>* tShape = @[@1];
        MLMultiArray* tArray =
            [[MLMultiArray alloc] initWithShape:tShape
                                      dataType:MLMultiArrayDataTypeInt32
                                         error:&error];
        if (error != nil || tArray == nil) {
            NSLog(@"[coreml_predict] Failed to allocate t array: %@",
                  error.localizedDescription);
            return -1;
        }
        memcpy(tArray.dataPointer, &t_val, sizeof(int32_t));

        /* ── Assemble feature provider ── */
        NSDictionary<NSString*, MLFeatureValue*>* featureDict = @{
            @"input_ids": [MLFeatureValue featureValueWithMultiArray:inputArray],
            @"t":         [MLFeatureValue featureValueWithMultiArray:tArray],
        };
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:featureDict
                                                              error:&error];
        if (error != nil || provider == nil) {
            NSLog(@"[coreml_predict] Failed to create feature provider: %@",
                  error.localizedDescription);
            return -1;
        }

        /* ── Run inference ── */
        id<MLFeatureProvider> output =
            [model predictionFromFeatures:provider error:&error];
        if (error != nil || output == nil) {
            NSLog(@"[coreml_predict] Prediction failed: %@",
                  error.localizedDescription);
            return -1;
        }

        /* ── Copy logits to caller-supplied buffer ── */
        MLFeatureValue* logitsFeature = [output featureValueForName:@"logits"];
        if (logitsFeature == nil) {
            NSLog(@"[coreml_predict] Output 'logits' not found.");
            return -1;
        }
        MLMultiArray* logitsArray = logitsFeature.multiArrayValue;
        if (logitsArray == nil) {
            NSLog(@"[coreml_predict] 'logits' feature is not a MultiArray.");
            return -1;
        }

        NSInteger count = (NSInteger)seq_len * vocab_size;

        if (logitsArray.dataType == MLMultiArrayDataTypeFloat16) {
            /* ANE outputs Float16. Convert to Float32 in one SIMD pass via vImage.
               dataPointer is CPU-accessible — the original SIGSEGV was caused by
               a buffer overrun (reading 4 bytes per element from a 2-byte buffer),
               NOT by inaccessible memory. */
            vImage_Buffer src = {
                .data     = logitsArray.dataPointer,
                .height   = 1,
                .width    = (vImagePixelCount)count,
                .rowBytes = (size_t)count * sizeof(__fp16),
            };
            vImage_Buffer dst = {
                .data     = logits_out,
                .height   = 1,
                .width    = (vImagePixelCount)count,
                .rowBytes = (size_t)count * sizeof(float),
            };
            vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
        } else if (logitsArray.dataType == MLMultiArrayDataTypeFloat32) {
            memcpy(logits_out, logitsArray.dataPointer, (size_t)count * sizeof(float));
        } else {
            /* Fallback for Double or other types. */
            for (NSInteger i = 0; i < count; i++) {
                logits_out[i] = [[logitsArray objectAtIndexedSubscript:i] floatValue];
            }
        }

        return 0;
    }
}
