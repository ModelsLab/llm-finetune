from celery import Celery,Task , current_task,states
import os
import logging
from unsloth import FastLanguageModel
import traceback
from .config import (
    MODELS_PATH
)
import torch
from .utils import (
    flush_memory,get_model_filename_from_id,timed
)
from typing import Dict
redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = os.environ.get('REDIS_PORT', 6379)
redis_password = os.environ.get('REDIS_PASSWORD', None)
redis_username = os.environ.get('REDIS_USERNAME', 'default')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tasks.log"),
        logging.StreamHandler(),
    ],
)


if redis_host == "localhost":
    cel_app = Celery("tasks", broker="redis://localhost/1", backend="redis://localhost/2")
else:
    cel_app = Celery("tasks",
                    broker="redis://{redis_username}:{redis_password}@{redis_host}:{redis_port}",
                    backend="redis://{redis_username}:{redis_password}@{redis_host}:{redis_port}",
                    broker_connection_retry=True,
                    broker_connection_max_retries=3,
                    broker_connection_retry_delay=5)
    

class CurrentModel:
    def __init__(self,model_id:str) -> None:
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    @staticmethod
    def exists(model_id: str) -> bool:
        if "/" in model_id:
            model_filename = get_model_filename_from_id(model_id)
            model_path = os.path.join(MODELS_PATH, model_filename)
        else:
            model_filename = f"{model_id}.pt"
            model_path = os.path.join(MODELS_PATH, model_filename)
        return os.path.exists(model_path)
    
    @staticmethod
    def delete(model_id: str) -> bool:
        pass

    def get(self, model_id: str,model_kwargs:Dict[str,str]):
        if model_id == self.model_id:
            logging.debug("Model is already loaded")
            return self.model, self.tokenizer
        
        self.model = None
        self.tokenizer = None
        flush_memory()

        self.model_id = model_id

        if "/" in model_id:
            model_filename = get_model_filename_from_id(self.model_id)
            tokenizer_filename = model_filename[:-3] + "_tokenizer.pt"
        else:
            model_filename = f"{self.model_id}.pt"
            tokenizer_filename = model_filename[:-3] + "_tokenizer.pt"

        model_path = os.path.join(MODELS_PATH, model_filename)
        tokenizer_path = os.path.join(MODELS_PATH, tokenizer_filename)

        if os.path.exists(model_path):
            try :
                logging.debug(f"Model {model_id} is already saved in .pt format")
                self.model = torch.load(
                    model_path, map_location=torch.device("cpu"), mmap=True
                )
                print("success")
                if os.path.exists(tokenizer_path):
                    self.tokenizer = torch.load(
                        tokenizer_path, map_location=torch.device("cpu"), mmap=True
                    )
                else:
                    raise ModuleNotFoundError()

            except ModuleNotFoundError as e:
                logging.error(f"Exception: {e}")
                traceback.print_exc()
                logging.debug(
                    f"It seems like the model cannot be loaded with torch.load. Trying .from_pretrained"
                )
        else :
            try : 
                self.model , self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = model_id, # Choose ANY! eg mistralai/Mistral-7B-Instruct-v0.2
                    max_seq_length = model_kwargs.get("max_seq_length"),
                    dtype = model_kwargs.get("dtype"),
                    load_in_4bit = model_kwargs.get("load_in_4bit"),
                    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
                )
                torch.save(self.model, model_path)
                torch.save(self.tokenizer, tokenizer_path)
            except Exception as e:
                logging.error(f"Exception occurred while loading model from HF: {e}")
                # logging.info(f"Defaulting to {DEFAULT_MODEL_ID} due to exception")

        ## TODO : DEFAULT MODEL LOADEING
                    
        self.model.to("cuda").eval()
        return self.model , self.tokenizer
    
