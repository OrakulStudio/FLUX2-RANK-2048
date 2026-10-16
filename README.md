# 🚨 IMPORTANT UPDATE: RANK 2048 BREAKTHROUGH FOR FLUX.2 (32B) ON CONSUMER GPU (16.10.2026)

> **Update Date:** October 2026 (Follow-up to the September 21 post)  
> **Target Hardware:** RTX 4090 (24GB VRAM) + 128GB RAM + Intel i9-13900K  
> **Status:** ⚡ Full Stability, Zero OOM, Maximum Tensor Core Occupancy  

---

## 🚀 Executive Summary & Critical Findings

In our previous update (21.09), we successfully ran extra-high **Rank 2048** training on a 32-billion parameter FLUX.2 model using `offload: 0.91`. 

However, after deep-dive profiling via **NVIDIA Nsight Systems** (`nsys`) and driver-level benchmarking, we identified a **critical architectural conflict**:

When the framework's default `low_vram` mode is enabled alongside **Viking Engine** (our custom Asynchronous CUDA Memory Manager), the two engines compete for memory control and CUDA streams at the WDDM / pinned-memory layer. This competition induces severe latency spikes (up to 750ms kernel launch delays) and suboptimal GPU occupancy.

By disabling `low_vram` and letting Viking Engine take full control, we eliminated stream contention, resulting in smooth asynchronous memory transfers and maximum compute throughput.

---

## 🛠 Recommended Configuration (YAML)

To achieve maximum training speed and zero OOM errors on a single 24GB GPU at Rank 2048:

```yaml
job: extension
config:
  name: r2048ftranspskin
  process:
  - type: diffusion_trainer
    training_folder: D:\AI_Models\ai-toolkit\output
    sqlite_db_path: D:\AI_Models\ai-toolkit\aitk_db.db
    device: cuda
    trigger_word: r2048ftranspskin
    performance_log_every: 10
    network:
      type: lora
      linear: 2048
      linear_alpha: 64    
      conv: 32            
      conv_alpha: 64      
      lokr_full_rank: true
      lokr_factor: -1
      network_kwargs:
        ignore_if_contains: []
    save:
      dtype: bf16
      save_every: 50
      max_step_saves_to_keep: 2
      save_format: diffusers
      push_to_hub: false
    datasets:
    - folder_path:  D:\AI_Models\datasets\modeldata\transporen
      mask_path: null
      mask_min_value: 0.1
      default_caption: ''
      caption_ext: txt
      caption_dropout_rate: 0.05
      cache_latents_to_disk: true
      is_reg: false
      network_weight: 1
      resolution:
      - 512
      controls: []
      shrink_video_to_frames: true
      num_frames: 1
      flip_x: false
      flip_y: false
      num_repeats: 1
      control_path_1: null
      control_path_2: null
      control_path_3: null
    train:
      batch_size: 1
      bypass_guidance_embedding: false
      steps: 100
      gradient_accumulation: 1
      train_unet: true
      train_text_encoder: false
      gradient_checkpointing: true
      noise_scheduler: flowmatch
      optimizer: adamw8bit
      timestep_type: weighted
      content_or_style: balanced
      optimizer_params:
        weight_decay: 0.0001
      unload_text_encoder: true
      cache_text_embeddings: false
      lr: 0.0001
      ema_config:
        use_ema: false
        ema_decay: 0.99
      skip_first_sample: true
      force_first_sample: false
      disable_sampling: true
      dtype: bf16
      diff_output_preservation: false
      diff_output_preservation_multiplier: 1
      diff_output_preservation_class: person
      switch_boundary_every: 1
      loss_type: mse
    logging:
      log_every: 1
      use_ui_logger: false
    model:
      name_or_path: black-forest-labs/FLUX.2-dev
      quantize: true
      qtype: bf8
      quantize_te: true
      qtype_te: bf8
      arch: flux2
      low_vram: false
      model_kwargs:
        match_target_res: false
      layer_offloading: true
      layer_offloading_text_encoder_percent: 0.71
      layer_offloading_transformer_percent: 0.71
    
meta:
  name: r2048ftranspskin
  version: '1.0'

```

### Why `offload: 0.71` + `low_vram: false`?
1. **`low_vram: false`**: Hands total memory control to Viking Engine's double-buffered async pipeline, removing framework overhead.
2. **`offload: 0.71`**: Frees PCIe bandwidth, ensuring high-speed pre-staging of weights in host RAM without choking CUDA execution streams.

---

## 📊 Nsight Systems Profiling Metrics

Our profiling session (`orakul_report2048.nsys-rep`) validates complete decoupling of compute and memory transfers:

* **Stream 7 (Default Compute Stream):**
  * **83.2%** of active execution time saturated with heavy `cutlass` and `ampere_bf16` GEMM operations.
* **Stream 13 (Dedicated Async Memory Stream):**
  * **90.2%** of stream time executing asynchronous Host-to-Device transfers.
  * Over **1.07 TB (Terabyte!)** of memory transferred H2D seamlessly across the session without stalling GPU compute kernels.

---

## 📁 Attached Artifacts & Benchmarks

* 📄 [r2048ftranspskin.yaml](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/r2048ftranspskin.yaml)` — Complete training configuration file.
* 📈 [orakul_report2048.nsys-rep](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/orakul_report2048.nsys-rep) — Raw Nsight Systems trace report.
* 📋 [ORAKUL_stats_output.txt](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/ORAKUL_stats_output.txt) — Summary of CUDA API and Kernel execution statistics.

---

---

## Nsight Systems trace report

| | | |
|---|---|---|
Metrics in screenshots
![1](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%20(1210).png) | ![2](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%20(1211).png) | ![3](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%20(1210).png) |

---

---

## Test

| | 
|---|
![R2048](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Report-Flux2D(32B)-Rank2048-ofload-0.71-Low-Ram-False-dataset-512x512/photo_2026-10-16_02-34-32.jpg) |


### 📊 FLUX.2 (32B BF8) VRAM & Offload Benchmarks (RTX 4090 24GB)

> **Hardware Context:** RTX 4090 (24GB GDDR6X) | Windows 11 WDDM Driver | Viking Async CUDA Memory Manager

---

#### 🎯 Quick Decision Matrix (Mode Selection)

| Mode | `offload` | Resolution | Rank | Cruising Speed ​​| Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stability Mode** *(Set & Forget)* | **`0.71`** | **512×512 / 1024×1024** | **16 – 2048** | **~8 s/it** (512×512)<br>**~16–20 s/it** (1024×1024)<br>**~100 s/it** (Rank 2048) | **Universal preset.** Overcomes memory fragmentation after frequent restarts, prevents WDDM spinlocks, and completely eliminates OOM (Out of Memory) errors at any rank. |
| **Performance Mode** *(Clean Boot)* | **`0.65`** | **1024×1024** | **16 – 128** | **~12–15 s/it** | **Maximum speed.** Requires high computational density (4096 tokens) within a "fresh" CUDA context to overlap weight transfers over the PCIe bus. | ---

#### 💡 Operational Mechanics and WDDM Behavior

1. **`offload: 0.71` (Absolute Stability Mode):**
* Frees up an additional **~2 GB of VRAM**, preventing the Windows WDDM driver from needing to defragment video memory during the micro-pauses between asynchronous CUDA streams. 
* Allows the asynchronous buffer to clear itself instantly: even if the initial step is delayed due to a heavily loaded system, the timing stabilizes to optimal, sustained values ​​by the second or third step.

2. **`offload: 0.65` (Maximum Compression Mode):**
* The hard limit for a native resolution of 1024×1024. 
* At resolutions below 1024×1024 (e.g., 512×512) or during frequent restarts, computations may execute too rapidly, triggering an **Async Stream Underflow**. With no available VRAM headroom, the WDDM driver enters a spinlock state (90+ s/it). In such scenarios, `offload: 0.71` must be used instead.

---

#### 📸 Logs
**Rank 128 @ 512x512 / 1024x1024 Recovery Log (`offload: 0.71`):**

Set offload to 0.65 | Rank 128 at 1024x1024
[Log](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/1024x1024/Rank128-dataset1024x1024-ofload-0.65.txt)

Set offload to 0.71 | Rank 128 at 512x512
[Log](https://github.com/OrakulStudio/FLUX2-RANK-2048/blob/main/Offload%20Benchmarks/512x512/Log_Rank128-dataset512x512-ofload-0.71.txt)



---
> **TL;DR:** Do not run `low_vram` alongside custom asynchronous offloaders. Set `low_vram: false` and `offload: 0.71` in Viking Engine to train 12B FLUX2 models at Rank 2048 on consumer hardware with zero bottlenecks.


















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
