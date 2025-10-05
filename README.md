# Fine-Tuning-LLaMA-2-using-LoRA-and-4-bit-Quantization
 Fine-Tuning LLaMA-2 with LoRA & 4-bit Quantization

This repository demonstrates how to fine-tune Meta’s LLaMA-2 language model efficiently using:

LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning

BitsAndBytes (4-bit quantization) for low-memory model loading

Hugging Face Transformers + PEFT + Accelerate for a complete end-to-end training pipeline

🚀 Features

✅ 4-bit quantized model loading (NF4 quantization)
✅ Parameter-efficient LoRA fine-tuning
✅ Support for mixed precision (fp16/bf16)
✅ Compatible with transformers, peft, bitsandbytes, accelerate, and trl
✅ Custom chat prompt format for LLaMA-2
✅ Easily adaptable for instruction-tuning or chatbot tasks

🧩 Installation
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

🧠 Notebook Overview
Step 1 — Install Dependencies

Installs all required packages:
transformers, peft, bitsandbytes, accelerate, trl.

Step 2 — Import Libraries

Imports PyTorch and Hugging Face utilities.

Step 3 — Prompt Template for LLaMA-2

Defines the standard chat prompt structure used by LLaMA-2:

System: <optional system instruction>
User: <prompt or query>
Model: <expected response>

Step 4 — Load and Quantize the Model

Loads the pretrained model in 4-bit precision:

bnb_4bit_compute_dtype = torch.float16
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

Step 5 — Configure LoRA (PEFT)

Applies LoRA to specific layers for efficient fine-tuning:

from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

Step 6 — Training Configuration

Defines mixed precision and caching options:

fp16 = False
bf16 = False
model.config.use_cache = False
model.config.pretraining_tp = 1

Step 7 — Fine-Tuning

Runs supervised fine-tuning on your dataset using PEFT and Accelerate.

Step 8 — Save & Evaluate

Saves the LoRA-adapted weights and tests generation quality.

🛠️ Example Command to Run Training
python train.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --output_dir ./llama2-qlora-finetuned \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3

