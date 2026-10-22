/*
 * Auto-Generated TinyML Model Configuration Header
 * Retrained on actual telemetry (S1-S9: 1.5V - 2.3V)
 * Target: STM32 / ARM Cortex-M Firmware
 */
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define NUM_INPUTS          9
#define NUM_OUTPUTS         3
#define NUM_SHAPE_CLASSES   2

/* Sensor Normalization Parameters (StandardScaler fit on telemetry data) */
static const float INPUT_MEANS[9] = { 2.257294f, 2.031018f, 2.130260f, 2.239680f, 1.856868f, 2.228789f, 2.205536f, 2.185270f, 2.246056f };
static const float INPUT_STDS[9]  = { 0.013106f, 0.148059f, 0.091816f, 0.033303f, 0.133878f, 0.047418f, 0.097651f, 0.087409f, 0.038911f };

/* Shape Class Labels */
static const char* const SHAPE_CLASSES[2] = { "cube", "cylinder" };

/*
 * Output: [3 floats]
 *   [0..1] = Softmax probs (cube, cylinder)
 *   [2]      = Predicted dim_mm
 */
#endif /* MODEL_CONFIG_H */
