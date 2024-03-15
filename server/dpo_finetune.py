import torch
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
from unsloth import FastLanguageModel

class DPOFinetune:
    def __init__(self, model_name, output_dir, train_dataset):
        self.model_name = model_name
        self.output_dir = output_dir
        self.train_dataset = train_dataset

    def train(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
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
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
        )

        dpo_trainer = DPOTrainer(
            model,
            model_ref=None,
            args=training_args,
            beta=0.1,
            train_dataset=self.train_dataset,
            tokenizer=tokenizer,
        )
        dpo_trainer.train()

# Usage example:
model_name = ""  # pass model name here
output_dir = ""  # model output dir
train_dataset = None  # train dataset here

dpo_finetune = DPOFinetune(model_name, output_dir, train_dataset)
dpo_finetune.train()
