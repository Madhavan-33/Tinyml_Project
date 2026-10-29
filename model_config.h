/*
 * Auto-Generated TinyML Model Configuration Header
 * 9-Sensor Tactile Mesh (S1-S9: 1.6V - 2.28V)
 * Target: STM32 / ARM Cortex-M Firmware
 */
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define NUM_INPUTS          9
#define NUM_OUTPUTS         3
#define NUM_SHAPE_CLASSES   2

/* Sensor Normalization Parameters (StandardScaler fit on dataset) */
static const float INPUT_MEANS[9] = { 2.070542f, 1.966373f, 2.070543f, 2.056118f, 1.939721f, 2.056036f, 2.070535f, 1.966333f, 2.070612f };
static const float INPUT_STDS[9]  = { 0.108731f, 0.097390f, 0.108791f, 0.117959f, 0.105098f, 0.117977f, 0.108775f, 0.097378f, 0.108750f };

/* Shape Class Labels */
static const char* const SHAPE_CLASSES[2] = { "cube", "cylinder" };

/*
 * Output: [3 floats]
 *   [0..1] = Softmax probs (cube, cylinder)
 *   [2]      = Predicted dim_mm
 */
#endif /* MODEL_CONFIG_H */
