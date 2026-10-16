# GLM5.3flaash

Custom C++/CUDA inference engine for **zai-org/GLM-5.3-Flash** (`glm5_next`, 320B MoE / 18B active)
targeting **12 GB VRAM + 64 GB RAM + SSD** consumer hardware.

Design:

- **GPU** runs attention (KDA linear + DeepSeek sparse MLA + indexer), dense layers, the shared
  expert, router, norms, embeddings, LM head and sampling.
- **CPU** runs the routed experts (KTransformers-style), fed from a 3-tier store
  `SSD -> RAM -> CPU`, with an mmap-based hot/warm/cold placement driven by router statistics.
- **All compute kernels are custom** (no cuBLAS/cuDNN): CUDA on the GPU, AVX2/AVX-512 on the CPU.
- Weights are consumed directly from unsloth `UD-IQ3_XXS` GGUF (approx. 120 GB).
- Serving uses the **OpenAI Responses API** (plus `/v1/chat/completions`).

Status: early bring-up (M0). See `docs/architecture.md` and `docs/roadmap.md`.

## Build

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 ^
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

## Tools

```powershell
build\cpp\Release\gguf_dump.exe <model.gguf>
```
