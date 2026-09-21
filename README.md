# r2048f2dvangog - Rank 2048 LoRA for FLUX.2-dev

**Ultra-high-rank style LoRA trained on FLUX.2-dev via the Viking Engine fork of ai-toolkit.**

Trained and maintained by **Roman Nikonov (Orakul)** - [Orakul Studio](https://github.com/OrakulStudio), Chernihiv, Ukraine 🇺🇦

---

## Samples

| | | |
|---|---|---|
| ![sample 1](samples/sample_01_woman_apples.png) | ![sample 2](samples/sample_02_sunflowers.png) | ![sample 3](samples/sample_03_peeling_potatoes.png) |


---

## What this is

A LoRA trained at an unusually high rank  **2048**  directly on `black-forest-labs/FLUX.2-dev`, pushed through on a single **RTX 4090**. Training at this rank on a 24GB card is normally a hard OOM wall; this run was made possible by the **Viking Engine**, an asynchronous memory manager built on top of `ai-toolkit`.

## Training specs

| Parameter | Value |
|---|---|
| Base model | `black-forest-labs/FLUX.2-dev` (arch: `flux2`) |
| Network type | LoRA |
| Rank (linear) | **2048** |
| Alpha | 64 |
| Conv rank / alpha | 32 / 64 |
| Effective scale | 0.0312 |
| Precision | BF8 native (E5M2), CPU pre-quant |
| Text encoder | Mistral-Small-3.1-24B-Instruct, quantized |
| Optimizer | AdamW8bit |
| LR | 1e-4 |
| Scheduler | Flowmatch |
| Batch size | 1 |
| Resolution | 512 |
| Save interval | every 30 steps |
| Hardware | RTX 4090 (Ada Lovelace) |
| Trigger word | `r2048f2dvangog` |

## Engine: Viking Engine (Asynchronous Memory Manager)

Rank 2048 on a 4090 works because of a custom fork of `ai-toolkit`, built and hardened by Orakul Studio:

- Double-buffered async weight streaming — zero OOM at rank 1024+
- CPU pinned-memory pre-staging (bypasses pagefile/swap entirely)
- Hardware-level BF16 / FP8 (E5M2) execution pipeline
- 100% deterministic checkpoint resume (no restart lottery, no silent stalls)

Fork: **https://github.com/OrakulStudio/AI-Toolkit-Windows11**

> The memory/checkpointing architecture itself has its own dedicated repos and writeups (see Orakul Studio for the full evolution of the module)  this README is just the result log for this specific rank-2048 run, not a re-explanation of that system.

## Checkpoints

All intermediate checkpoints from this run are published on:

- 🤗 Hugging Face: *[OrakulStorm](https://huggingface.co/OrakulStorm)*
- 🅲 CivitAI: *[ORAKUL_STUDIO](https://civitai.com/user/ORAKUL_STUDIO)*

A **distilled rank-128** version is in progress and will be released shortly for lighter-weight inference.

## Status

Trained and shipped under active shelling in Chernihiv, Ukraine. System stable.

## License

*(add license - e.g. same terms as FLUX.2-dev base model / your preferred license)*
