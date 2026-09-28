import os

def convert_tflite_to_c_header(tflite_path, output_header_path):
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"Could not find {tflite_path}")

    with open(tflite_path, "rb") as f:
        tflite_bytes = f.read()

    num_bytes = len(tflite_bytes)
    
    # Format bytes nicely in hex (12 bytes per line)
    hex_lines = []
    for i in range(0, num_bytes, 12):
        chunk = tflite_bytes[i:i+12]
        hex_str = ", ".join([f"0x{b:02x}" for b in chunk])
        hex_lines.append("  " + hex_str)

    array_content = ",\n".join(hex_lines)

    header_content = f"""/*
 * Auto-Generated C Header for TensorFlow Lite Micro Model
 * Source File: {os.path.basename(tflite_path)}
 * Size: {num_bytes} bytes
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

// Align to 16-byte boundary for optimized SIMD/NEON/Vector MCU execution
#if defined(__GNUC__) || defined(__clang__)
__attribute__((aligned(16)))
#elif defined(_MSC_VER)
__declspec(align(16))
#endif
static const unsigned char g_model[] = {{
{array_content}
}};

static const unsigned int g_model_len = {num_bytes};

#ifdef __cplusplus
}}
#endif

#endif /* MODEL_DATA_H */
"""

    with open(output_header_path, "w") as f:
        f.write(header_content)

    print(f"Successfully converted {tflite_path} ({num_bytes} bytes) -> {output_header_path}")

if __name__ == "__main__":
    tflite_file = "anomaly_model.tflite"
    output_file = "model_data.h"
    convert_tflite_to_c_header(tflite_file, output_file)
