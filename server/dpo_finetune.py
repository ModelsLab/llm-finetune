import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel

model = FastLanguageModel.from_pretrained("gpt2")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "", #pass model name here
    max_seq_length = max_seq_length,
    dtype = None, 
    load_in_4bit = True,
)

training_args = TrainingArguments(
    output_dir="", #model output dir
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    prediction_loss_only=True,
    dataloader_num_workers=4,
    remove_unused_columns=False,
    run_name="run_name",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Dropout = 0 is currently optimized
    bias = "none",    # Bias = "none" is currently optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
)


dpo_trainer = DPOTrainer(
    model,
    model_ref=None,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset, #train dataset here
    tokenizer=tokenizer,
)
dpo_trainer.train()
