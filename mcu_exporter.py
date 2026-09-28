import os
import zipfile
import io

def generate_arduino_sketch():
    return """/*
 * TinyML 9-Sensor Object & Sizing Classifier - Arduino Sketch
 * Architecture: 9 Sensor Inputs (S1-S9) -> Object Shape Classification + Size Regression (dim_mm)
 * Compatibility: Arduino Nano 33 BLE / ESP32 / Teensy 4.0 / STM32 (via STM32Core)
 */

#include "model_config.h"
#include "model_data.h"

// Note: Ensure TensorFlowLite_ESP32 or Harvard_TFLiteMicro library is installed in Arduino IDE
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 12 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000);

    Serial.println("=========================================");
    Serial.println("🚀 TinyML 9-Sensor Object & Size Classifier");
    Serial.println("=========================================");

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("❌ Model schema mismatch error!");
        return;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("❌ AllocateTensors() failed!");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("✅ TinyML Interpreter initialized successfully!");
}

unsigned long sample_count = 0;

void loop() {
    sample_count++;
    float current_sensors[NUM_INPUTS];

    // Read or simulate 9 sensor values captured from sensor mesh (S1 ... S9)
    for (int i = 0; i < NUM_INPUTS; i++) {
        current_sensors[i] = INPUT_MEANS[i] + ((float)random(-100, 100) / 100.0f) * INPUT_STDS[i] * 0.5f;
        if (current_sensors[i] < 0.01f) current_sensors[i] = 0.01f;
    }

    // Normalize 9 sensor inputs
    for (int i = 0; i < NUM_INPUTS; i++) {
        float std_val = (INPUT_STDS[i] == 0.0f) ? 1.0f : INPUT_STDS[i];
        input->data.f[i] = (current_sensors[i] - INPUT_MEANS[i]) / std_val;
    }

    // Invoke Inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("❌ Inference Invoke failed!");
        delay(1000);
        return;
    }

    // Decode Object Shape (Argmax over NUM_SHAPE_CLASSES)
    int best_class_idx = 0;
    float max_prob = -1.0f;
    for (int c = 0; c < NUM_SHAPE_CLASSES; c++) {
        float prob = output->data.f[c];
        if (prob > max_prob) {
            max_prob = prob;
            best_class_idx = c;
        }
    }

    const char* pred_shape = SHAPE_CLASSES[best_class_idx];
    float confidence_pct = max_prob * 100.0f;
    float pred_size_mm = output->data.f[NUM_SHAPE_CLASSES];

    // Structured Serial Telemetry Output
    Serial.print("[MCU_TELEMETRY] SAMPLE=");
    Serial.print(sample_count);
    Serial.print(", OBJECT=");
    Serial.print(pred_shape);
    Serial.print(", CONF=");
    Serial.print(confidence_pct, 1);
    Serial.print("%, SIZE_MM=");
    Serial.print(pred_size_mm, 2);
    Serial.print(", S1=");
    Serial.print(current_sensors[0], 2);
    Serial.print(", S9=");
    Serial.println(current_sensors[8], 2);

    delay(500);
}
"""

def generate_platformio_ini():
    return """; PlatformIO Project Configuration File for TinyML Anomaly Detector
[platformio]
default_envs = stm32f407_discovery

[env:stm32f407_discovery]
platform = ststm32
board = discovery_f407vg
framework = stm32cube
build_flags =
    -O3
    -mfloat-abi=hard
    -mfpu=fpv4-sp-d16
lib_deps =
    tensorflow/TensorFlowLiteMicro

[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps =
    tanakamasayuki/TensorFlowLite_ESP32

[env:nano33ble]
platform = nordicnrf52
board = nano33ble
framework = arduino
monitor_speed = 115200
"""

def generate_makefile():
    return """# Makefile for ARM Cortex-M TinyML Compilation
CC = arm-none-eabi-gcc
CXX = arm-none-eabi-g++
OBJCOPY = arm-none-eabi-objcopy

MCU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CFLAGS = $(MCU) -O2 -Wall -I.
CXXFLAGS = $(CFLAGS) -std=c++17 -fno-rtti -fno-exceptions

SRCS = main_tflm.cpp
TARGET = firmware

all: $(TARGET).elf $(TARGET).bin

$(TARGET).elf: $(SRCS)
\t$(CXX) $(CXXFLAGS) $(SRCS) -o $@

$(TARGET).bin: $(TARGET).elf
\t$(OBJCOPY) -O binary $< $@

clean:
\trm -f *.o *.elf *.bin
"""

def generate_readme():
    return """# 🚀 Physical Microcontroller TinyML Model Deployment Guide

This package contains everything needed to run your trained TensorFlow Lite Micro Anomaly Detector on a **Physical Microcontroller** or inside the **Renode Emulator**.

---

## 📁 Included Files:
1. `model_data.h` - C byte array representation of your trained `.tflite` model.
2. `model_config.h` - Feature scaling means, standard deviations, and anomaly threshold.
3. `main_tflm.cpp` - C++ main application entry point with TFLite Micro interpreter runtime loop.
4. `Arduino_Sketch.ino` - Complete ready-to-flash sketch for Arduino IDE (ESP32 / Nano 33 BLE / Teensy).
5. `platformio.ini` - PlatformIO build settings for STM32, ESP32, and nRF52.
6. `Makefile` - GCC ARM Cortex-M cross-compilation Makefile.
7. `emulate_stm32.resc` - Renode simulation script for testing without physical hardware.

---

## 🔌 Deploying to Physical Microcontrollers

### Option A: Arduino IDE (ESP32 / Arduino Nano 33 BLE / STM32)
1. Open `Arduino_Sketch.ino` in Arduino IDE.
2. Copy `model_data.h` and `model_config.h` into the same folder as the sketch.
3. Install **TensorFlowLite_ESP32** or **Harvard_TFLiteMicro** from Library Manager.
4. Select your board and COM port.
5. Click **Upload** (Ctrl+U).
6. Open **Serial Monitor** at **115200 baud** to view live inference logs!

### Option B: PlatformIO (VS Code)
1. Open this folder in VS Code with PlatformIO extension installed.
2. Select your environment (`stm32f407_discovery`, `esp32dev`, or `nano33ble`).
3. Click **Build** and **Upload**.

### Option C: STM32CubeProgrammer / ST-Link CLI
Build using ARM GCC toolchain and flash via ST-Link:
```bash
make
st-flash write firmware.bin 0x8000000
```

### Option D: ESP32 Direct Flashing (`esptool.py`)
```bash
esptool.py --chip esp32 --port COM3 --baud 921600 write_flash -z 0x10000 firmware.bin
```

---

## 📡 Connecting to Streamlit UI
Once flashed onto your physical microcontroller:
1. Plug the microcontroller into your computer via USB.
2. Open the Streamlit App -> Go to **Tab 3: 🔌 Physical Microcontroller Connection**.
3. Select your microcontroller's **COM Port** (e.g. `COM3`, `/dev/ttyUSB0`) and **115200 Baud**.
4. Click **▶️ Read Serial Stream Packets** or enable **Continuous Live Auto-Stream**!
"""

def create_mcu_deployment_zip():
    """Builds an in-memory ZIP archive containing all microcontroller deployment artifacts."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Include existing model header files if present
        if os.path.exists("model_data.h"):
            zip_file.write("model_data.h", arcname="model_data.h")
        if os.path.exists("model_config.h"):
            zip_file.write("model_config.h", arcname="model_config.h")
        if os.path.exists("main_tflm.cpp"):
            zip_file.write("main_tflm.cpp", arcname="main_tflm.cpp")
        if os.path.exists("emulate_stm32.resc"):
            zip_file.write("emulate_stm32.resc", arcname="emulate_stm32.resc")
        if os.path.exists("anomaly_model.tflite"):
            zip_file.write("anomaly_model.tflite", arcname="anomaly_model.tflite")

        # Add generated Arduino sketch, configs, and guides
        zip_file.writestr("Arduino_Sketch/Arduino_Sketch.ino", generate_arduino_sketch())
        zip_file.writestr("platformio.ini", generate_platformio_ini())
        zip_file.writestr("Makefile", generate_makefile())
        zip_file.writestr("README_DEPLOYMENT.md", generate_readme())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()

if __name__ == "__main__":
    content = create_mcu_deployment_zip()
    with open("mcu_deployment_package.zip", "wb") as f:
        f.write(content)
    print(f"Generated mcu_deployment_package.zip ({len(content)} bytes)")
