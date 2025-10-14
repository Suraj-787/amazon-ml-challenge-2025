# Image Model Plan: High-Speed CLIP Fine-Tuning with LoRA

**Goal:** produce a high-performance image price prediction model in under 4 hours on a single H100. The approach is to freeze the large pre-trained CLIP model and fine-tune only a tiny fraction of its parameters with LoRA adapters, giving the speed of a small model with the knowledge of a large one.

The implementation lives in [`image_model/`](../image_model/README.md).

## Phase 1: Setup — The LoRA Foundation

- **Core technology:** Parameter-Efficient Fine-Tuning (PEFT) using the Hugging Face `peft` library. This is the key to finishing on time.
- **Base model:** `openai/clip-vit-large-patch14`, using its pre-trained knowledge without paying the full training cost.
- **LoRA configuration:** a `LoraConfig` defines how the small, trainable adapter layers are injected into the frozen CLIP model.
  - **Target modules:** the attention layers' query (`q_proj`) and value (`v_proj`) projections — the most effective place to adapt the model's visual understanding.
  - **Adapter size (rank):** start small (`r=8` or `r=16`). A smaller rank means faster training.

## Phase 2: Model Architecture — The Adapted Predictor

The final model has three parts:

1. **The frozen giant:** the pre-trained `CLIPVisionModel`. Its 400M+ parameters are not trained.
2. **The adapters:** small LoRA layers (a few million parameters) injected by `peft`. These are trained.
3. **The decision maker:** a custom regression head (a few linear layers). This is also trained.

`peft` combines the first two into a single `PeftModel`.

## Phase 3: The H100 High-Speed Training Loop

- **Selective training:** configure `AdamW` to update only the LoRA adapter and regression head parameters, explicitly excluding the rest of the model so each step is fast.
- **`torch.compile()`:** wrap the combined `PeftModel` before training starts — the single biggest accelerator.
- **bfloat16 mixed precision:** run training in bfloat16, the H100's native format.
- **Maximize batch size:** push the batch size as high as 80GB of VRAM allows, starting at 512. Larger batches mean fewer steps per epoch.

## Phase 4: Data Pipeline and Objective

- **S3 data streaming:** a `boto3`-powered `Dataset` loads images directly from S3, avoiding disk bottlenecks.
- **Target:** the log-transformed price, `log(price + 1)`.
- **Loss:** mean squared error between predicted and true log-prices.

## Phase 5: Final Output and Success Criteria

- **Goal:** within the 4-hour window, complete as many epochs as possible until validation loss stops improving.
- **Deliverable:** the saved model artifacts, not prices:
  1. the trained LoRA adapter weights (a very small file), and
  2. the trained regression head weights.

To predict later, load the base CLIP model, apply the saved adapter weights and attach the saved head.
