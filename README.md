<div align="center">

# ⚡ PocketForge

**Custom 1B LLM inference engine — for 32-bit ARM phones**  
Mixed-precision weights · Streaming 16K KV cache · MTP heads · HTTP API  
**All under 250 MB RAM**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![ARM](https://img.shields.io/badge/arch-armv7a%20%7C%20aarch64-red)]()
[![RAM](https://img.shields.io/badge/RAM-%3C250MB-brightgreen)]()

</div>

---

## Why PocketForge?

Existing engines (llama.cpp, etc.) load entire weight matrices into RAM. For a 1B parameter model at Q8 that's **~1.18 GB** — impossible on a 32-bit phone with 250 MB budget.

PocketForge solves this with:

- **Mixed-precision quantization** — per-matrix bit allocation (Q4/Q2/ternary) based on importance, averaging ~2.5 bits/param
- **Streaming weight decompression** — only one layer's weights decompressed at a time via mmap + zstd
- **int8 streaming KV cache** — attention computed directly on quantized data, no float scratch buffer. **16K+ context in ~204 MB**
- **Multi-token prediction** — 4 heads with Q4 weights, tied embeddings, no 1GB unembed matrix
- **Predictive layer skipping** — skip FFN computation when hidden state delta is below threshold

---

## Quick Start

### Prerequisites

- **C++17 compiler** (GCC 9+ / Clang 10+)
- **CMake** 3.20+
- **zstd** dev library
- ARM NEON recommended (auto-detected)

### Install Dependencies

```bash
# Debian / Ubuntu
sudo apt install build-essential cmake libzstd-dev

# Android (Termux)
pkg install build-essential cmake zstd
```

### Build

```bash
git clone https://github.com/tundefund0-gif/pocketforge.git
cd pocketforge
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**You'll get three binaries:**

| Binary | Path | Size |
|---|---|---|
| `forge-quant` | `build/forge-quant` | 72 KB |
| `forge` | `build/forge` | 79 KB |
| `forge-server` | `build/forge-server` | 150 KB |

---

## Usage

### 1. Convert a model to `.squeeze` format

```bash
./forge-quant model.gguf model.squeeze \
  --layers 24 --embd 2048 --heads 32 \
  --kv-heads 4 --ff 8192 --vocab 32000 --mtp 4
```

Options control the model architecture. The quantizer automatically assigns:
- **Q4** (4-bit) → ~20% of matrices (attention Q/K/V/O, first/last layers)
- **Q2** (2-bit) → ~70% of matrices (middle FFN)
- **Q1.5** (ternary) → ~10% of matrices (least important)

### 2. Run inference (CLI)

```bash
./forge model.squeeze --prompt "Hello" --tokens 128
```

### 3. Run API server

```bash
./forge-server model.squeeze 8080
```

Then hit the API:

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Once upon a time","max_tokens":256,"temperature":0.7}'
```

---

## API Reference

### `POST /v1/completions`

OpenAI-compatible text generation.

**Request body:**
```json
{
  "prompt": "Your text here",
  "max_tokens": 256,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "id": "cmpl-1234567890",
  "object": "text_completion",
  "created": 1234567890,
  "model": "pocketforge-1b",
  "choices": [{
    "text": "Generated text...",
    "index": 0,
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 256,
    "total_tokens": 261
  }
}
```

### `GET /v1/models`

Returns model metadata including architecture and memory usage.

### `GET /health`

Health check — returns `{"status":"ok","engine":"pocketforge"}`

### `GET /stats`

Returns real-time memory breakdown.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   PocketForge Engine                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Weight Storage:  mmap'd compressed .squeeze file        │
│  Decompression:   zstd, block-by-block, prefetched       │
│  Matrix Ops:      NEON int8 (SDOT/SMLAD), no FPU        │
│  KV Cache:        int8 quantized, streaming attention    │
│  Context:         16,384 tokens                          │
│  MTP:             4 heads, Q4 weights, tied embeddings   │
│  Layer Skip:      predictive FFN skipping                │
│  API:             HTTP/1.1, JSON, no external deps       │
│                                                         │
│  RAM Budget:  ~232 MB (16K ctx)  —  within 250 MB ✓    │
└─────────────────────────────────────────────────────────┘
```

### Memory Breakdown (1B model, 16K context)

| Component | Size |
|---|---|
| KV cache (int8 data) | 192 MB |
| KV scales (float32) | 12 MB |
| MTP heads (Q4) | 24 MB |
| Prefetch buffer | 4 MB |
| Activations | 64 KB |
| **Total** | **~232 MB** |

---

## Cross-Compiling for 32-bit ARM

### Android NDK

```bash
# Set up NDK toolchain
$ANDROID_NDK/build/tools/make-standalone-toolchain.sh \
  --arch arm --platform android-21 --install-dir /tmp/arm-tc

# Build
cmake .. -DCMAKE_C_COMPILER=/tmp/arm-tc/bin/arm-linux-androideabi-gcc \
         -DCMAKE_CXX_COMPILER=/tmp/arm-tc/bin/arm-linux-androideabi-g++ \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### ARM32 NEON detection

PocketForge auto-detects NEON at compile time. On armv7a it uses `SMLAD`-based int8 matmul. On aarch64 it uses `SDOT` instructions.

---

## Project Structure

```
pocketforge/
├── include/forge/        # Public headers
│   ├── types.hpp         # Config, quant types, memory budget
│   ├── quant_format.hpp  # .squeeze format, quantizer, loader
│   ├── kv_cache.hpp      # Streaming int8 KV cache
│   ├── matmul_neon.hpp   # NEON int8 matmul
│   ├── prefetch.hpp      # Async decompression pipeline
│   ├── mtp.hpp           # Multi-token prediction
│   ├── layer_skip.hpp    # Predictive layer skipping
│   ├── engine.hpp        # Main inference engine
│   └── http_server.hpp   # HTTP API server
├── src/                  # Implementations
├── tests/                # 10 passing unit tests
└── CMakeLists.txt        # Build system
```

---

## License

MIT
