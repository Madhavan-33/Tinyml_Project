#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define NUM_INPUTS 9
#define NUM_OUTPUTS 4
#define NUM_SHAPE_CLASSES 3

static const float INPUT_MEANS[9] = { 3.202305f, 3.664538f, 4.456880f, 5.040150f, 5.565090f, 6.314999f, 6.902228f, 7.450386f, 8.132190f };
static const float INPUT_STDS[9] = { 3.225461f, 3.792713f, 4.477175f, 5.044365f, 5.656010f, 6.313482f, 6.821719f, 7.341157f, 7.899473f };
static const char* const SHAPE_CLASSES[3] = { "Beam", "Cylinder", "Sphere" };

#endif
