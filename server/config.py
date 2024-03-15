# MODEL_NAME = "runwayml/stable-diffusion-v1-5"
# INSTANCE_DIR = "training_images"
# CLASS_DIR_men = "class_images_men"
# CLASS_DIR_women = "class_images_women"
# CLASS_DIR_couple = "class_images_couple"
# CLASS_DIR = "class_images"
# OUTPUT_DIR = "./output/trained-model"
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://huggingface.co/api/models/"
MODELS_PATH = "system_models/"