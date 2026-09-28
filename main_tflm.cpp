/*
 * TinyML Multi-Sensor Firmware Template for ARM Cortex-M / Renode / STM32
 * Architecture: 9 Sensor Inputs (S1-S9) -> Object Shape Classification + Size Regression (dim_mm)
 * Continuous Emulation Stream with Structured Serial UART Output
 */

#include <stdio.h>
#include <stdlib.h>
#include "model_config.h"
#include "model_data.h"

// TensorFlow Lite for Microcontrollers Headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Allocate 12KB memory arena for tensor operations
constexpr int kTensorArenaSize = 12 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// Simple LCG random float generator [0.0, 1.0] for embedded environments
static float rand_float() {
    static unsigned int seed = 12345;
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (float)seed / (float)0x7fffffff;
}

int main() {
    printf("=== TinyML 9-Sensor Object & Sizing Model Booting on ARM Cortex-M / Renode ===\n");
    printf("Configured NUM_INPUTS: %d (S1-S9) | NUM_OUTPUTS: %d | Classes: %d\n", 
           NUM_INPUTS, NUM_OUTPUTS, NUM_SHAPE_CLASSES);

    // 1. Load TFLite Model from C Byte Array
    const tflite::Model* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: Model schema version mismatch!\n");
        return -1;
    }

    // 2. Setup Op Resolver
    tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();

    // 3. Build Micro Interpreter
    tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);

    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        printf("ERROR: AllocateTensors() failed!\n");
        return -1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    printf(">>> STARTING REAL-TIME 9-SENSOR INFERENCE LOOP (OBJECT & SIZE) <<<\n");

    float current_sensors[NUM_INPUTS];
    int sample_count = 0;

    while (1) {
        sample_count++;

        // Simulate incoming 9-sensor readings captured from physical sensor mesh
        for (int i = 0; i < NUM_INPUTS; i++) {
            current_sensors[i] = INPUT_MEANS[i] + (rand_float() - 0.5f) * INPUT_STDS[i];
            if (current_sensors[i] < 0.01f) current_sensors[i] = 0.01f;
        }

        // 4. Preprocess Features (Normalize with INPUT_MEANS and INPUT_STDS)
        for (int i = 0; i < NUM_INPUTS; i++) {
            float std_val = (INPUT_STDS[i] == 0.0f) ? 1.0f : INPUT_STDS[i];
            input->data.f[i] = (current_sensors[i] - INPUT_MEANS[i]) / std_val;
        }

        // 5. Run Model Inference
        TfLiteStatus invoke_status = interpreter.Invoke();
        if (invoke_status != kTfLiteOk) {
            printf("ERROR: Inference Invoke failed!\n");
            break;
        }

        // 6. Decode Outputs:
        // - Indices [0 .. NUM_SHAPE_CLASSES - 1]: Shape probabilities
        // - Index [NUM_SHAPE_CLASSES]: Predicted Dimension (mm)
        int best_class_idx = 0;
        float max_prob = -1.0f;
        for (int c = 0; c < NUM_SHAPE_CLASSES; c++) {
            float prob = output->data.f[c];
            if (prob > max_prob) {
                max_prob = prob;
                best_class_idx = c;
            }
        }

        const char* predicted_shape = SHAPE_CLASSES[best_class_idx];
        float confidence_pct = max_prob * 100.0f;
        float predicted_dim_mm = output->data.f[NUM_SHAPE_CLASSES];

        // Structured UART Telemetry Output
        printf("[RENODE_MCU] SAMPLE=%d, OBJECT=%s, CONFIDENCE=%.1f%%, SIZE_MM=%.2f, S1=%.2f, S2=%.2f, S9=%.2f\n",
               sample_count, predicted_shape, confidence_pct, predicted_dim_mm,
               current_sensors[0], current_sensors[1], current_sensors[8]);

        // MCU Delay Loop
        for (volatile int d = 0; d < 120000; d++);
    }

    return 0;
}
