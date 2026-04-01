LORA_R = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03


r=LORA_R, 
lora_alpha=LORA_ALPHA,
target_modules="all-linear", 
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM"


output_dir=f"{ROOT_DIR}/results_squad_sft_r{LORA_R}",
per_device_train_batch_size=2, 
gradient_accumulation_steps=4,
gradient_checkpointing=True,
num_train_epochs=NUM_EPOCHS,
learning_rate=LEARNING_RATE,
warmup_ratio=WARMUP_RATIO,
lr_scheduler_type="cosine",
neftune_noise_alpha=5.0,
optim="adamw_torch",
logging_steps=10,
use_mps_device=(device == "mps"),
max_length=512,
dataset_text_field="text",
