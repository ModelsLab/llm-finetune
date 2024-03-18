from transformers import TrainingArguments,AutoModelForCausalLM, AutoTokenizer
import torch
from trl import DPOTrainer
from unsloth import FastLanguageModel , PatchDPOTrainer
from typing import Dict , List

class DPOFineTune:
    def __init__(self,model,tokenizer , output_dir , train_dataset) -> None:
        self.model = model
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset

    def train(self,training_params:Dict[str,str],parms:Dict[str,str]):
        training_args = TrainingArguments(
            output_dir=training_args["output_dir"],
            num_train_epochs= training_args["num_train_epochs"],
            per_device_train_batch_size = training_args["per_device_train_batch_size"],
            per_device_eval_batch_size = training_args["per_device_eval_batch_size"],
            gradient_accumulation_steps = training_args["gradient_accumulation_steps"],
            gradient_checkpointing = training_args["gradient_checkpointing"],
            optim = training_args["optim"],
            learning_rate = training_args["learning_rate"],
            max_grad_norm = training_args["max_grad_norm"],
            warmup_ratio = training_args["warmup_ratio"],
            lr_scheduler_type = training_args["lr_scheduler_type"],
            logging_steps = training_args["logging_steps"],
            save_steps = training_args["save_steps"],
            save_total_limit = training_args["save_total_limit"],
            evaluation_strategy = training_args["evaluation_strategy"],
            eval_steps = training_args["eval_steps"],
            bf16 = training_args["bf16"],
            tf32 = training_args["tf32"],
            push_to_hub=False,
            report_to="tensorboard"
        )
        
        ## unsolth logic
        model = FastLanguageModel.get_peft_model(
            self.model,
            r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 64,
            lora_dropout = 0, # Currently only supports dropout = 0
            bias = "none",    # Currently only supports bias = "none"
            use_gradient_checkpointing = True,
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        PatchDPOTrainer()

        dpo_trainer =  DPOTrainer(
            self.model,
            ref_model= None,
            args = training_args,
            beta = 0.1,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer
        )

        dpo_trainer.train()