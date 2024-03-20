import os
import subprocess
from threading import Thread
import boto3
import requests
import torch
from subprocess import call, getoutput, Popen
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
from flask import Flask, jsonify, request, send_from_directory
import time
import logging
from PIL import Image
from io import BytesIO
import PIL
import toml
import uuid
import glob
import gc
from .utils import client,session
from tasks import Celery , cel_app
import celery
from .src.data_loader.hf_datasets import DataLoader
from .tasks import dpo_trainer

import torch.multiprocessing as mp



app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024


from redis import Redis

imagine_db = Redis(host='localhost', db=3)

# try:
#     imagine_db.ping()
# except Exception as e:
#     print("Redis not running, starting")
#     Thread(target=restart_server, args=()).start()

def restart_server():
    print("Restarting server")
    # subprocess.call(['kill', '-9', '-1'])

def queue_time_estimate():
    try:
        with celery.pool.acquire(block=True) as conn:
            tasks = conn.default_channel.client.lrange('celery', 0, -1)

        total_time = 0
        total_time=len(tasks)*20
        
        total_time = round(total_time, 2)
        return total_time
    except Exception as e:
        logging.error("restart server 1")
        logging.error(e)
        Thread(target=restart_server, args=(
        )).start()
        return 0





from flask import Flask, request, jsonify
from celery.result import AsyncResult

# Store the current task ID
current_task_id = None





import requests
import os



@app.route("/create-project",methods = ["POST"])
def create_project():
    """
        Creation of finetunning project with unique uuid.
    """
    if request.method == "POST":
        name = request.form["name"]
        webhook = request.form["webhook"]
        project_id = uuid.uuid4()
        os.environ["WANDB_PROJECT"] = str(project_id)  # name your W&B project
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        
        response = requests.post(webhook,data={
            "name" : name,
            "webhook" : webhook,
            "project_id" : project_id
        })

        if response.status_code == 201 :
            return jsonify({
                "status" : "success",
                "message" : "Project created succesfully."
            })
        else :
            return jsonify({
                "status" : "failure",
                "message" : "somthing went wrong please try again."
            })
    

@app.route("/data/upload-data",methods = ["POST"])
def upload_data():
    """
    Upload of data in specied format.
    """
    raw_data = request.files["raw_data"]
    project_id = request.form.get("project_id")
    webhook = request.form.get("webhook")
    
    ### following are the paramters required for data loading
     
    data_format = request.form.get("format",default=None) # json , csv parquet etc
    train_data_files = request.files["training_data"]
    validation_data_files = request.files["validation_data"]
    test_data_files = request.files["testing_data"]
    num_proc = request.form.get("num_of_processes",default=8)


    ## upload required file to s3 bucket over here ...

    ## logic for for background task for loading data into memory.

    params = {
        "project_id" : project_id,
        "raw_data" : raw_data.filename,
        "data_format" : data_format,
        "training_data" : train_data_files.filename,
        "testing_data" : test_data_files.filename,
        "validation_data" : validation_data_files.filename,
        "num_of_proc" : num_proc
    }

    response = requests.post(webhook,data=params)
    if response.status_code == 201 :
        return jsonify({
            "status" : "success",
            "message" : "data upload succesfull."
        })
    else :
        return jsonify({
            "status" : "failure",
            "message" : "data upload failed"
        })

@app.route("/data/load_from_hf",methods = ["POST"])
def load_from_hf():
    """
    Load data from huggingface dataset hub.
    """

    if request.method == "POST":

        dataset_id = request.form.get("dataset_id")
        project_id = request.form.get("project_id")
        webhook = request.form.get("webhook")
        split = request.form.get("split",default=None)

        dataloader = DataLoader(dataset_id=dataset_id,split=split)
        info = dataloader.get_dataset_info()
        if not info:
            return jsonify({
                "status":"failure",
                "message" : "You have given wrong dataset id please check on huggingface dataset hub and try again."
            })

        response = requests.post(webhook,data={
            "project_id": project_id,
            "dataset_id" : dataset_id,
            "split" : split
        })
        if response.status_code == 201 :
            return jsonify({
                "status" : "success",
                "message" : "dataset creatin succesfull"
            })
        else :
            return jsonify({
                "status" : "failure",
                "message" : "something went wrong please try again."
            })

def task_time_estimate(name: str, params: Dict[str, Any]) -> float:
    ### TODO : Find the ways on the basis of tasktype and task
    return 30.0

@app.route("/dpo_trainer",methods = ["POST"])
def dpo_finetune():
    model_id = request.form.get("model_id",default=None)
    project_id = request.form.get("project_id",default= None)
    if project_id == None :
        return jsonify({
            "status" : "failure",
            "message":"you forgot to provide project id for training."
        })  

    if not model_id:
        return jsonify({
            "status" : "failure",
            "message":"you forgot to provide model id for training.take a look at supported models and specify id."
        })  
    training_dataset = request.form.get("train_data",default=None)
    if not training_dataset:
        return jsonify({
            "status" : "failure",
            "message" : "Provide the name of uploaded training dataset"
        })
    epochs = request.form.get("epochs",default=1)
    per_device_train_batch_size = request.form.get("per_device_train_batch_size",default=12)
    per_device_eval_batch_size = request.form.get("per_device_eval_batch_size",default=4)
    gradient_accumulation_steps = request.form.get("gradient_accumulation_steps",default=1)
    optim = request.form.get("optimizer",default=None)
    if not optim :
        return jsonify({
            "status":"failure",
            "message" : "You forgot to specify optimizer for training purpose."
        })
    
    learning_rate = request.form.get("learning_rate",default=5e-5)
    max_grad_norm = request.form.get("max_grad_norm",default=0.3)
    warmup_ratio = request.form.get("warmup_ratio",default=0.1)
    lr_scheduler_type= request.form.get("lr_scheduler_type",default="cosine")
    logging_steps = request.form.get("logging_steps",default=25)
    save_steps = request.form.get("save_steps",default=500)
    evaluation_strategy = request.form.get("evaluation_strategy",default="steps")
    eval_steps= request.form.get("eval_steps",default=700)
    bf16 = request.form.get("bf16",default=True)
    tf32 = request.form.get("tf32",default=True)

    ### peft configs 
    r = request.form.get("r",default=64)
    lora_alpha = request.form.get("r",default=64)
    lora_dropout = request.form.get("lora_dropout",default=0)
    bias = request.form.get("bias",default="none")
    use_gradient_checkpointing = request.form.get("use_gradient_checkpointing",default=True)
    random_state = request.form.get("random_state",default=3407),
    use_rslora = request.form.get("use_rslora",default=False,type=bool)
    
    ### dpo config
    beta = request.form.get("beta",default=0.1)

    ### check for dataset config
    dataset_id = request.form.get("dataset_id")

    dataset_configs = request.form.get("dataset_configs",default={})


    ## model configs 
    max_seq_length = request.form.get("max_seq_length",4096)
    dtype = request.form.get("dtype")
    load_in_4bit = request.form.get("load_in_4bit")
    webhook = request.form.get("webhook",default=None)
    output_dir = "/dpo-models"
    params = {
        "project_id" : project_id,
        "webhook" : webhook,
        "dataset_id" : dataset_id,
        "dataset_configs" : dataset_configs,
        "training_args" : {
            "output_dir" : output_dir,
            "num_train_epochs" : epochs,
            "per_device_train_batch_size" : per_device_train_batch_size,
            "per_device_eval_batch_size" : per_device_eval_batch_size,
            "gradient_accumulation_steps" : gradient_accumulation_steps,
            "gradient_checkpointing" : use_gradient_checkpointing,
            "optim" : optim,
            "learning_rate" : learning_rate,
            "max_grad_norm" : max_grad_norm,
            "warmup_ratio" : warmup_ratio,
            "lr_scheduler_type" : lr_scheduler_type,
            "logging_steps" : logging_steps,
            "save_steps" : save_steps,
            "save_total_limit" : 1,
            "evaluation_strategy" : evaluation_strategy,
            "eval_steps" : eval_steps,
            "bf16" : bf16,
            "tf32" : tf32,
        },
        "peft_configs" : {
            "r" : r,
            "lora_alpha" : lora_alpha,
            "lora_dropout" : lora_dropout,
            "bias" : bias,
            "use_gradient_checkpointing" : use_gradient_checkpointing,
            "random_state" : random_state,
            "use_rslora" : use_rslora
        },
        "dpo_configs" : {
            "beta" : beta
        },
        "model_configs" : {
            "max_seq_length" : max_seq_length,
            "dtype" : dtype,
            "load_in_4bit" : load_in_4bit
        }
    }

    name = "dpo_trainer"

    task_time = task_time_estimate(name, params)
    time_in_queue = queue_time_estimate() + task_time
    task = dpo_trainer.apply_async(priority=0, kwargs={"params": params})
    imagine_db.set(task.id,task_time)


    return {
            "status": "queued",
            "eta": time_in_queue,
            "response": "",
            "tip": "Your finetuning request has been queued. Please check the training logs on wandb",
            "meta": params,
        }
    



def download_file(url, destination_folder="./pretrained_models"):


    try:
        # Sending a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Extracting the file name from the URL
        file_name = os.path.basename(url)
        destination_path = os.path.join(destination_folder, file_name)

        # Creating the destination folder if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)

        # Writing the content to the file
        with open(destination_path, 'wb') as file:
            file.write(response.content)

        print(f"File downloaded successfully to {destination_path}")
        return destination_path
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")
        return None

def delete_all_files(directory):
    # Constructing a path pattern to match all files in the directory
    if not os.path.exists(directory):
        print("not there")
    files = glob.glob(os.path.join(directory, '*'))
    for file_path in files:
        try:
            # Checking if it's a file (not a directory) before deleting
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"File {file_path} deleted successfully")
            else:
                print(f"{file_path} is not a file, skipping")
        except Exception as e:
            print(f"An error occurred while deleting {file_path}: {e}")


@app.route('/output/<path:path>')
def send_report(path):
    return send_from_directory('output', path)


@app.route('/train', methods=['POST'])
def train():
    config = request.get_json()
    if not config['training_type']:
        config['training_type'] = "none"
    config = dict(config)
    Thread(target=start_traindb, kwargs=config).start()
    print("Training started")
    return jsonify({'status': 'Training Started'})


@app.route('/train_v2', methods=['POST'])
def train_v2():
    config = request.get_json()
    if not config['training_type']:
        config['training_type'] = "none"
    config = dict(config)
    Thread(target=start_train_v2, kwargs=config).start()
    print("Training started")
    return jsonify({'status': 'Training Started'})


def start_train_v2(**params):

    # if os.path.exists(INSTANCE_DIR+"/.ipynb_checkpoints"):
    #     call('rm -r '+INSTANCE_DIR+'/.ipynb_checkpoints', shell=True)
    # if os.path.exists(CLASS_DIR_men+"/.ipynb_checkpoints"):
    #     call('rm -r '+CLASS_DIR_men+'/.ipynb_checkpoints', shell=True)
    # if os.path.exists(CLASS_DIR+"/.ipynb_checkpoints"):
    #     call('rm -r '+CLASS_DIR+'/.ipynb_checkpoints', shell=True)

    from types import SimpleNamespace
    opt = SimpleNamespace(**params)
    print(opt)
    base_model = opt.base_model
    instance_prompt = opt.instance_prompt
    class_prompt = opt.class_prompt
    training_type = opt.training_type
    webhook = opt.webhook
    training_id = opt.training_id
    images = opt.images
    learning_rate_unet = getattr(opt, 'learning_rate_unet', "2e-6")
    steps_unet = getattr(opt, 'steps_unet', "1500")

    learning_rate_text_encoder = getattr(
        opt, 'learning_rate_text_encoder', "1e-6")
    steps_text_encoder = getattr(opt, 'steps_text_encoder', "350")

    start_time = time.time()
    if not os.path.exists(INSTANCE_DIR):
        os.makedirs(INSTANCE_DIR)
    print("Downloading images")
    for i, url in enumerate(images):
        img = download_image(url)
        if img is not None:
            img.save(f'{INSTANCE_DIR}/{i}.png')
        else:
            print("bad image")

    files=glob.glob(os.path.join("./training_images", '*'))
    if len(files)<3:
        print("Valid image links less than 3")
        requests.post(webhook, data={
                'status': 'training_failed',
                'eta': '0',
                'message': 'Training Failed:  Valid image links less than 3',
                'training_id': training_id})
        return

    # STEPS = str(max_train_steps)

    INSTANCE_PROMPT = instance_prompt
    CLASS_PROMPT = class_prompt
    # print the parameters
    print(training_type, INSTANCE_PROMPT)
    if training_type == "men":
        CLASS_DIR = CLASS_DIR_men
    elif training_type == "female":
        CLASS_DIR = CLASS_DIR_women
    elif training_type == "couple":
        CLASS_DIR = CLASS_DIR_couple
    else:
        training_type = "none"
        CLASS_DIR = "class_images"
    if not os.path.exists(CLASS_DIR):
        os.makedirs(CLASS_DIR)

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    requests.post(webhook, data={
        'status': 'training_started',
        'eta': '30 Minutes',
        'logs': 'Step:1 Training Started, 5 More Steps to go : ETA : 60 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    print("Training here")
    try:
        # train text encoder
        subprocess.run(["python3", "train_dreamboothv2.py",
                        "--pretrained_model_name_or_path", base_model,
                        "--instance_data_dir", INSTANCE_DIR,
                        "--class_data_dir", CLASS_DIR,
                        "--output_dir", OUTPUT_DIR,
                        "--instance_prompt", INSTANCE_PROMPT,
                        "--class_prompt", CLASS_PROMPT,
                        "--learning_rate", learning_rate_text_encoder,
                        "--resolution", "512",
                        "--with_prior_preservation",
                        "--prior_loss_weight", "1.0",
                        "--gradient_accumulation_steps", "1",
                        "--gradient_checkpointing",
                        "--train_text_encoder",
                        "--dump_only_text_encoder",
                        "--lr_scheduler", "linear",
                        "--lr_warmup_steps", "0",
                        "--train_batch_size", "1",
                        "--num_class_images", "200",
                        "--max_train_steps", steps_text_encoder,
                        "--mixed_precision", "fp16",
                        ])
        # train unet
        subprocess.run(["python3", "train_dreamboothv2.py",
                        "--pretrained_model_name_or_path", base_model,
                        "--instance_data_dir", INSTANCE_DIR,
                        "--class_data_dir", CLASS_DIR,
                        "--output_dir", OUTPUT_DIR,
                        "--instance_prompt", INSTANCE_PROMPT,
                        "--class_prompt", CLASS_PROMPT,
                        "--learning_rate", learning_rate_unet,
                        "--resolution", "512",
                        "--with_prior_preservation",
                        "--prior_loss_weight", "1.0",
                        "--gradient_accumulation_steps", "1",
                        "--gradient_checkpointing",
                        "--train_only_unet",
                        "--lr_scheduler", "linear",
                        "--lr_warmup_steps", "0",
                        "--train_batch_size", "1",
                        "--num_class_images", "200",
                        "--max_train_steps", steps_unet,
                        "--mixed_precision", "fp16",
                        ])
    except Exception as e:
        requests.post(webhook, data={
            'status': 'training_failed',
            'eta': '0',
            'message': '1. [Training model] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' + str(e),
            'training_id': training_id})
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("training done")

    requests.post(webhook, data={
        'status': 'training_success',
        'eta': '15 Minutes',
        'logs': 'Step:2 Training Done, 4 More Steps to go : ETA : 15 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    if os.path.exists(os.path.join(OUTPUT_DIR, "model_index.json")):
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                OUTPUT_DIR, safety_checker=None, torch_dtype=torch.float16).to("cuda")
            prompt = INSTANCE_PROMPT + " blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city , Canon EOS R3, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K"
            image = pipe(prompt, num_inference_steps=50,
                         guidance_scale=7.5).images[0]
            image.save("output/test.png")
            print("making zip")
            requests.post(webhook, data={
                'status': 'trained_model_compressing',
                'eta': '25 Minutes',
                'test_image': 'output/test.png',
                'logs': 'Step:3 Model Compressing, 3 More Steps to go : ETA : 25 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
                'training_id': training_id})
            del pipe
            del image
            torch.cuda.empty_cache()
        except Exception as e:
            requests.post(webhook, data={
                'status': 'training_failed',
                'eta': '0',
                'message': '2. [Creating test image] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' + str(e),
                'training_id': training_id})
            return
        try:
            subprocess.run(
                ["tar", "-czvf", f"output/trained-model.tar.gz", "-C", "output", "trained-model"])
            # verify tar
            subprocess.run(["tar", "-tf", f"output/trained-model.tar.gz"])
        except Exception as e:
            requests.post(webhook, data={
                'status': 'training_failed',
                'eta': '0',
                'message': '3. [Compressing tar] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' + str(e),
                'training_id': training_id})
            return
        print("Cleaning up")
    else:
        requests.post(webhook, data={
            'status': 'training_failed',
            'message': "4. [Invalid images, make sure its png files] | Time Elapsed : " + str(time.time() - start_time) + '',
            'training_id': training_id})
        print("training failed")
        return
    model_name = f"{training_id}.tar.gz"
    test_name = f"{training_id}.png"
    requests.post(webhook, data={
        'status': 'trained_model_uploading',
        'eta': '20 Minutes',
        'test_image': 'output/test.png',
        'logs': 'Step:3 Model Uploading, 2 More Steps to go : ETA : 20 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    try:
        client.upload_file('output/trained-model.tar.gz',
                           'trained-models',
                           model_name)
        client.upload_file('output/test.png',
                           'generations',
                           test_name)
    except Exception as e:
        requests.post(webhook, data={
            'status': 'training_failed',
            'eta': '0',
            'message': '4.[uploading model to s3] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' + str(e),
            'training_id': training_id})
        return

    print("file upload completed")
    requests.post(webhook, data={
        'status': 'trained_model_uploaded',
        'eta': '10 Minutes',
        'logs': 'Step:4 Model Uploaded, 1 More Steps to go : ETA : 10 minutes | Time Elapsed : ' + str(time.time() - start_time),
        'training_id': training_id,
        'model_link': 'output/trained-model.tar.gz',
        "test_image": "output/test.png"})
    return

def has_active_task():
    tasks=None
    with celery.pool.acquire(block=True) as conn:
        tasks = conn.default_channel.client.lrange('celery', 0, -1)
        # print("printed",tasks)
        items = len(tasks)
        print("Items : ",items)
        return items


def get_gpu_vram_usage():
    vram_usage = {}
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            vram_usage[gpu.id] = {}
            vram_usage[gpu.id]["total_vram"] = gpu.memoryTotal
            vram_usage[gpu.id]["used_vram"] = gpu.memoryUsed
            vram_usage[gpu.id]["free_vram"] = gpu.memoryFree
            vram_usage[gpu.id]["id"] = gpu.id
        return vram_usage
    except Exception as e:
        return vram_usage


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

@app.route("/")
@app.get("/")
def index():
    flush()
    try:
        imagine_db.ping()
    except Exception as e:
        print("Redis not running, starting")
        Thread(target=restart_server, args=()).start()
        return "Redis restarting"
    queue = queue_time_estimate()
    print("current queue time : ", queue)
    # check if default s3 bucket is used
    queue_num = 0
    with celery.pool.acquire(block=True) as conn:
        tasks = conn.default_channel.client.lrange("celery", 0, -1)
        queue_num = len(tasks)

    return {
        "status": "ok",
        "vram_usage": get_gpu_vram_usage(),
        "queue_time": queue,
        "queue_num": queue_num,

    }


@app.route('/train_status', methods=['POST'])
def train_status():
    config = request.get_json()
    
    # print(AsyncResult(task.task_id, app=celery).state)
    state=AsyncResult(config['id'], app=celery).state
    return jsonify({'status': state})
    

@app.route('/train_lora', methods=['POST'])
def train_lora():
    print("Started")

    config = request.get_json()
    if not config['training_type']:
        config['training_type'] = "none"
    config = dict(config)

    # Enqueue the training task and store the task ID
    task = start_train_lora.apply_async(kwargs=config)
    # print("Active Task : ",celery.control.inspect().active())
    # print("Reserved Task : ",celery.control.inspect().reserved())
    # print("Queue Task : ",celery.control.inspect().scheduled())
    # # Check if a task is already running
    if has_active_task():
            
            
            # reserved_tasks = celery.control.inspect().reserved()
            # queue_size = sum(len(tasks) for tasks in reserved_tasks.values()) if reserved_tasks else 0

            # scheduled_tasks = celery.control.inspect().scheduled()
            # print("Active Task : ",celery.control.inspect().active())
            # print("Reserved Task : ",celery.control.inspect().reserved())
            # print("Queue Task : ",celery.control.inspect().scheduled())

            queue_size=0
            with celery.pool.acquire(block=True) as conn:
                tasks = conn.default_channel.client.lrange('celery', 0, -1)
                queue_size = len(tasks)
                return jsonify({'status': 'queued', 'queue_size': queue_size,'task_id':str(task) ,'estimated_time': queue_time_estimate()})

    return jsonify({'status': 'training_started','task_id':str(task) ,'estimated_time': 20})




@app.route('/train_lora_xl', methods=['POST'])
def train_lora_xl():
    print("Started")

    config = request.get_json()
    if not config['training_type']:
        config['training_type'] = "none"
    config = dict(config)

    # Enqueue the training task and store the task ID
    task = start_train_lora_xl.apply_async(kwargs=config)
    # print("Active Task : ",celery.control.inspect().active())
    # print("Reserved Task : ",celery.control.inspect().reserved())
    # print("Queue Task : ",celery.control.inspect().scheduled())

    # # Check if a task is already running
    if has_active_task():
            queue_size=0
            with celery.pool.acquire(block=True) as conn:
                tasks = conn.default_channel.client.lrange('celery', 0, -1)
                queue_size = len(tasks)
            
                return jsonify({'status': 'queued', 'queue_size': queue_size, 'estimated_time': queue_time_estimate()})

    return jsonify({'status': 'training_started','task_id':str(task) ,'estimated_time': 20})

@app.route('/infer_lora', methods=['POST'])
def infer_lora():
    config = request.get_json()
    print("new docker image 18 nov 2022")
    if not config['training_type']:
        config['training_type'] = "none"
    Thread(target=start_infer_lora, args=(
        config['prompt'], config['seed'], config['steps'])).start()
    print("Training started")
    return jsonify({'status': 'Training Started'})


def start_infer_lora(prompt, lora_model, steps):
    from .scripts.lora_to_diffusers import load_lora_weights
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe = load_lora_weights(lora_model, MODEL_NAME)
    image = pipe(prompt, num_inference_steps=50,
                 guidance_scale=7.5).images[0]
    name = time.time()
    image.save("output" + "/" + str(name) + ".png")


def download_image(url):
    try:
        response = requests.get(url, stream=True, allow_redirects=True)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))
    except Exception as e:
        print(e)
        return None


def start_traindb(**params):
    from types import SimpleNamespace
    opt = SimpleNamespace(**params)
    print(opt)
    base_model = opt.base_model
    instance_prompt = opt.instance_prompt
    class_prompt = opt.class_prompt
    training_type = opt.training_type
    max_train_steps = opt.max_train_steps
    webhook = opt.webhook
    training_id = opt.training_id
    images = opt.images
    learning_rate_unet = getattr(opt, 'learning_rate_unet', "1e-6")
    learning_rate_text_encoder = getattr(
        opt, 'learning_rate_text_encoder', "1e-6")

    start_time = time.time()
    if not os.path.exists(INSTANCE_DIR):
        os.makedirs(INSTANCE_DIR)
    print("Downloading images")
    for i, url in enumerate(images):
        img = download_image(url)
        if img is not None:
            img.save(f'{INSTANCE_DIR}/{i}.png')
        else:
            print("bad image")

    files=glob.glob(os.path.join("./training_images", '*'))
    if len(files)<3:
        print("Valid image links less than 3")
        requests.post(webhook, data={
                'status': 'training_failed',
                'eta': '0',
                'message': 'Training Failed:  Valid image links less than 3',
                'training_id': training_id})
        return
        
    
    STEPS = str(max_train_steps)

    INSTANCE_PROMPT = instance_prompt
    CLASS_PROMPT = class_prompt
    # print the parameters
    print(STEPS, training_type, INSTANCE_PROMPT)
    if training_type == "men":
        CLASS_DIR = CLASS_DIR_men
    elif training_type == "female":
        CLASS_DIR = CLASS_DIR_women
    elif training_type == "couple":
        CLASS_DIR = CLASS_DIR_couple
    else:
        training_type = "none"
        CLASS_DIR = "class_images"
    if not os.path.exists(CLASS_DIR):
        os.makedirs(CLASS_DIR)

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    requests.post(webhook, data={
        'status': 'training_started',
        'eta': '30 Minutes',
        'logs': 'Step:1 Training Started, 5 More Steps to go : ETA : 60 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    print("Training here")
    try:
        subprocess.run(["python3", "train_dreambooth.py",
                        "--pretrained_model_name_or_path", base_model,
                        "--instance_data_dir", INSTANCE_DIR,
                        "--class_data_dir", CLASS_DIR,
                        "--output_dir", OUTPUT_DIR,
                        "--instance_prompt", INSTANCE_PROMPT,
                        "--class_prompt", CLASS_PROMPT,
                        "--resolution", "512",
                        "--with_prior_preservation",
                        "--center_crop",
                        "--allow_tf32",
                        "--set_grads_to_none",
                        "--prior_loss_weight", "1.0",
                        "--gradient_accumulation_steps", "1",
                        "--gradient_checkpointing",
                        "--train_text_encoder",
                        "--lr_scheduler", "constant",
                        "--lr_warmup_steps", "0",
                        "--train_batch_size", "1",
                        "--num_class_images", "200",
                        "--max_train_steps", STEPS,
                        "--mixed_precision", "fp16",
                        ])
    except Exception as e:
        requests.post(webhook, data={
            'status': 'training_failed',
            'eta': '0',
            'message': 'Training Failed:  ' + str(e),
            'training_id': training_id})
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("training done")
    requests.post(webhook, data={
        'status': 'training_success',
        'eta': '15 Minutes',
        'logs': 'Step:2 Training Done, 4 More Steps to go : ETA : 15 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    if os.path.exists(os.path.join(OUTPUT_DIR, "model_index.json")):
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                OUTPUT_DIR, safety_checker=None, torch_dtype=torch.float16).to("cuda")
            prompt = INSTANCE_PROMPT + " blue eyes, shaved side haircut, hyper detail, cinematic lighting, magic neon, dark red city , Canon EOS R3, nikon, f/1.4, ISO 200, 1/160s, 8K, RAW, unedited, symmetrical balance, in-frame, 8K"
            image = pipe(prompt, num_inference_steps=50,
                         guidance_scale=7.5).images[0]
            image.save("output/test.png")
            print("making zip")
            requests.post(webhook, data={
                'status': 'trained_model_compressing',
                'eta': '25 Minutes',
                'test_image': 'output/test.png',
                'logs': 'Step:3 Model Compressing, 3 More Steps to go : ETA : 25 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
                'training_id': training_id})
            del pipe
            del image
            torch.cuda.empty_cache()
        except Exception as e:
            requests.post(webhook, data={
                'status': 'training_failed',
                'eta': '0',
                'message': '2. [Creating test image] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' + str(e),
                'training_id': training_id})
            return
        try:
            subprocess.run(
                ["tar", "-czvf", f"output/trained-model.tar.gz", "-C", "output", "trained-model"])
            verify_command = ["tar", "-tf", f"output/trained-model.tar.gz"]

            # Execute the command
            result = subprocess.run(
                verify_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Check the result
            if result.returncode == 0:
                print("Tar archive is valid.")
            else:
                print("Tar archive is invalid. Performing tar operation again...")
                subprocess.run(
                    ["tar", "-czvf", f"output/trained-model.tar.gz", "-C", "output", "trained-model"])

        except Exception as e:
            requests.post(webhook, data={
                'status': 'training_failed',
                'eta': '0',
                'message': '3. [Compressing tar] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' + str(e),
                'training_id': training_id})
            return
        print("Cleaning up")
    else:
        requests.post(webhook, data={
            'status': 'training_failed',
            'message': "4. [Invalid images, make sure its png files] | Time Elapsed : " + str(time.time() - start_time) + '',
            'training_id': training_id})
        print("training failed")
        return
    model_name = f"{training_id}.tar.gz"
    test_name = f"{training_id}.png"
    requests.post(webhook, data={
        'status': 'trained_model_uploading',
        'eta': '20 Minutes',
        'test_image': 'output/test.png',
        'logs': 'Step:3 Model Uploading, 2 More Steps to go : ETA : 20 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    try:
        client.upload_file('output/test.png',
                           'generations',
                           test_name)
    except Exception as e:
        print(e)
    try:
        client.upload_file('output/trained-model.tar.gz',
                           'trained-models',
                           model_name, Config=boto3.s3.transfer.TransferConfig(
                               # Set the multipart chunk size (e.g., 8MB)
                               multipart_chunksize=8 * 1024 * 1024,
                               # Set the multipart threshold (e.g., 8MB)
                               multipart_threshold=8 * 1024 * 1024
                           ))
    except Exception as e:
        requests.post(webhook, data={
            'status': 'training_failed',
            'eta': '0',
            'message': 'uploading model failed :' + str(e),
            'training_id': training_id})
        return

    print("file upload completed")
    requests.post(webhook, data={
        'status': 'trained_model_uploaded',
        'eta': '10 Minutes',
        'logs': 'Step:4 Model Uploaded, 1 More Steps to go : ETA : 10 minutes | Time Elapsed : ' + str(time.time() - start_time),
        'training_id': training_id,
        'model_link': 'output/trained-model.tar.gz',
        "test_image": "output/test.png"})
    delete_all_files("./output/trained-model")
    delete_all_files("./training_images")
    return

@celery.task(name='tasks.start_train_lora_xl')
def start_train_lora_xl(**params):
    # Default values
    defaults = {
        "instance_prompt": "best quality, aqua eyes, baseball cap, closed mouth, green background, hat, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    "class_prompt": "photo of person",
        "class_prompt": "photo of person",
        "base_model": "https://huggingface.co/snowkidy/stable-diffusion-xl-base-0.9/resolve/main/sd_xl_base_0.9.safetensors",
        "images": [],
        "seed": "0",
        "training_type": "men",
        "max_train_steps": "2",
        "lora_type": "lora",
        "webhook": "https://stablediffusionapi.com",
        "training_id": "trained_model"
    }

    # Overriding default values with provided parameters
    params = {**defaults, **params}
    start_time = time.time()
    from types import SimpleNamespace
    opt = SimpleNamespace(**params)
    print(opt)
    training_id = opt.training_id
    webhook = opt.webhook
    training_type = opt.training_type
    instance_prompt = opt.instance_prompt
    negative_prompt = opt.negative_prompt
    class_prompt = opt.class_prompt
    max_train_steps = opt.max_train_steps
    lora_type=opt.lora_type
    images = opt.images
    learning_rate = getattr(opt, 'learning_rate', 1e-6)
    print("Dowloading base model")
    
    if not os.path.exists("./pretrained_models/sd_xl_base_0.9.safetensors"):
        download_file(opt.base_model_id)
    print("downloaded")
    
    print("Starting Training")
    if not os.path.exists(INSTANCE_DIR):
        os.makedirs(INSTANCE_DIR)
    for i, url in enumerate(images):
        img = download_image(url)
        if img is not None:
            img.save(f'{INSTANCE_DIR}/{i}.png')
        else:
            print("bad image")
    files=glob.glob(os.path.join("./training_images", '*'))
    if len(files)<3:
        print("Valid image links less than 3")
        print("Valid image links less than 3")
        requests.post(webhook, data={
                'status': 'training_failed',
                'eta': '0',
                'message': 'Training Failed:  Valid image links less than 3',
                'training_id': training_id})
        return
        return

    STEPS = int(max_train_steps)
    with open('./param/config_file_xl_old.toml', 'r') as file:
        config = toml.load(file)
    # Make changes to the config
    config['training_arguments']['max_train_epochs'] = STEPS
    config['training_arguments']['output_dir'] = OUTPUT_DIR
    config['training_arguments']['output_name'] = training_id
    config['training_arguments']['logging_dir'] = "output/logs"
    config['training_arguments']['log_prefix'] = training_id
    if lora_type=='lycoris':
        config['additional_network_arguments']['network_module']= "lycoris.kohya"
    else:
        config['additional_network_arguments']['network_module']= "networks.lora"
    # Save the changed config back to the TOML file
    with open('./param/config_file_xl_old.toml', 'w') as file:
        toml.dump(config, file)
    # write the updated content back to the sample_prompt.txt
    new_content=str(instance_prompt)+"   --n "+str(negative_prompt)+"   --w 512   --h 768   --l 7   --s 28"
    with open(os.path.join("./param/sample_prompt.txt"), "w") as file:
        file.write(new_content)

    print(STEPS, training_type, instance_prompt)
    if training_type == "men":
        CLASS_DIR = CLASS_DIR_men
    elif training_type == "female":
        CLASS_DIR = CLASS_DIR_women
    elif training_type == "couple":
        CLASS_DIR = CLASS_DIR_couple
    else:
        training_type = "none"
        CLASS_DIR = "class_images"
    # upload images to the training directory
    # mp.set_start_method('spawn')
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    requests.post(webhook, data={
        'status': 'training_started',
        'eta': '30 Minutes',
        'logs': 'Step:1 Training Started, 5 More Steps to go : ETA : 60 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    print("Training lora")
    try:
        if lora_type=='lycoris':
            print("-----Lycoris-----")
            subprocess.run(["python3", "sd-scripts/sdxl_train_network.py",
                            "--sample_prompts", "./param/sample_prompt.txt",
                            "--dataset_config", "./param/dataset_config.toml",
                            "--config_file", "./param/config_file_xl_old.toml",
                            "--network_args", "algo=locon",
                            ])
        else:
            print("lora")
            subprocess.run(["python3", "sd-scripts/sdxl_train_network.py",
                            "--sample_prompts", "./param/sample_prompt.txt",
                            "--dataset_config", "./param/dataset_config.toml",
                            "--config_file", "./param/config_file_xl_old.toml",
                            ])
    except Exception as e:
        print(e)
        requests.post(webhook, data={
            'status': 'training_failed',
            'eta': '0',
            'message': '1. [Training model] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' + str(e),
            'training_id': training_id})
        
        delete_all_files("./output/trained-model")
        delete_all_files("./training_images")

        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("training done")
    requests.post(webhook, data={
        'status': 'training_success',
        'eta': '15 Minutes',
        'logs': 'Step:2 Training Done, 4 More Steps to go : ETA : 15 minutes | Time Elapsed : ' + 
        str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    model_name = f"{training_id}.safetensors"
    test_name = f"{training_id}.png"
    
    print(model_name,test_name)
    requests.post(webhook, data={
        'status': 'trained_model_uploading',
        'eta': '20 Minutes',
        'test_image': 'output/test.png',
        'logs': 'Step:3 Model Uploading, 2 More Steps to go : ETA : 20 minutes | Time Elapsed : ' + 
        str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    print("Uploading")
    directory_path=OUTPUT_DIR+"/sample"

    try:
        model_file = os.path.join(OUTPUT_DIR, model_name)
        print("model_file : ",model_file)
        client.upload_file(model_file,
                           'trained-models',
                           model_name)
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if 
             os.path.isfile(os.path.join(directory_path, f))]
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        client.upload_file(files[0],
                               'generations',
                               test_name)
    except Exception as e:
        requests.post(webhook, data={
            'status': 'training_failed',
            'eta': '0',
            'message': '4.[uploading model to s3] | Time Elapsed : ' + str(time.time() - 
            start_time) + ' | ' + 
            str(e),
            'training_id': training_id})
        print("uploading model to s3 failed")
        delete_all_files("./output/trained-model")
        delete_all_files("./training_images")
        return

    print("file upload completed")
    requests.post(webhook, data={
        'status': 'trained_model_uploaded',
        'eta': '10 Minutes',
        'logs': 'Step:4 Model Uploaded, 1 More Steps to go : ETA : 10 minutes | Time Elapsed : ' + 
        str(time.time() - start_time),
        'training_id': training_id,
        'model_link': 'output/trained-model.tar.gz',
        "test_image": "output/test.png"})
    delete_all_files("./output/trained-model")
    delete_all_files("./training_images")
    return

@celery.task(name='tasks.start_train_lora')
def start_train_lora(**params):

    defaults = {
        "instance_prompt": "best quality, aqua eyes, baseball cap, closed mouth, green background, hat, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt",
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    "class_prompt": "photo of person",
        "class_prompt": "photo of person",
        "base_model_id": "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/AnyLoRA_noVae_fp16-pruned.safetensors",
        "images": [],
        "seed": "0",
        "training_type": "men",
        "max_train_steps": "2",
        "lora_type": "lora",
        "webhook": "https://stablediffusionapi.com",
        "training_id": "trained_model"
    }

    # Overriding default values with provided parameters
    params = {**defaults, **params}
    start_time = time.time()
    start_time = time.time()
    from types import SimpleNamespace
    opt = SimpleNamespace(**params)
    print(opt)
    training_id = opt.training_id
    webhook = opt.webhook
    training_type = opt.training_type
    instance_prompt = opt.instance_prompt
    negative_prompt = opt.negative_prompt
    class_prompt = opt.class_prompt
    lora_type=opt.lora_type
    max_train_steps = opt.max_train_steps
    images = opt.images
    learning_rate = getattr(opt, 'learning_rate', 1e-6)
    print("Dowloading base model")
    if not os.path.exists("./pretrained_models/AnyLoRA_noVae_fp16-pruned.safetensors"): 
        download_file(opt.base_model_id)
    print("downloaded")
    print("Starting Training")
    if not os.path.exists(INSTANCE_DIR):
        os.makedirs(INSTANCE_DIR)
    for i, url in enumerate(images):
        img = download_image(url)
        if img is not None:
            img.save(f'{INSTANCE_DIR}/{i}.png')
        else:
            print("bad image")

    files=glob.glob(os.path.join("./training_images", '*'))
    if len(files)<3:
        print("Valid image links less than 3")
        requests.post(webhook, data={
                'status': 'training_failed',
                'eta': '0',
                'message': 'Training Failed:  Valid image links less than 3',
                'training_id': training_id})
        return
        
    STEPS = int(max_train_steps)
    # Load the TOML file
    with open('./param/config_file.toml', 'r') as file:
        config = toml.load(file)
    # Make changes to the config
    config['training_arguments']['max_train_epochs'] = STEPS
    config['training_arguments']['output_dir'] = OUTPUT_DIR
    config['training_arguments']['output_name'] = training_id
    config['training_arguments']['logging_dir'] = "output/logs"
    config['training_arguments']['log_prefix'] = training_id
    
    if lora_type=='lycoris':
        print("-------------------- Network Module ---------------------------------")
        config['additional_network_arguments']['network_module']= "lycoris.kohya"
    else:
        config['additional_network_arguments']['network_module']= "networks.lora"
    # Save the changed config back to the TOML file
    with open('./param/config_file.toml', 'w') as file:
        toml.dump(config, file)
    # write the updated content back to the sample_prompt.txt
    new_content=str(instance_prompt)+"   --n "+str(negative_prompt)+"   --w 512   --h 768   --l 7   --s 28"
    with open(os.path.join("./param/sample_prompt.txt"), "w") as file:
        file.write(new_content)

    print(STEPS, training_type, instance_prompt)
    if training_type == "men":
        CLASS_DIR = CLASS_DIR_men
    elif training_type == "female":
        CLASS_DIR = CLASS_DIR_women
    elif training_type == "couple":
        CLASS_DIR = CLASS_DIR_couple
    else:
        training_type = "none"
        CLASS_DIR = "class_images"
    # upload images to the training directory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    requests.post(webhook, data={
        'status': 'training_started',
        'eta': '30 Minutes',
        'logs': 'Step:1 Training Started, 5 More Steps to go : ETA : 60 minutes | Time Elapsed : ' + str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    print("Training lora")
    try:
        if lora_type=='lycoris':
            print("Lora")
            subprocess.run(["python3", "sd-scripts/train_network.py",
                            "--sample_prompts", "./param/sample_prompt.txt",
                            "--dataset_config", "./param/dataset_config.toml",
                            "--config_file", "./param/config_file.toml",
                            "--network_args", "algo=locon",

                            ])
        else:
            subprocess.run(["python3", "sd-scripts/train_network.py",
                            "--sample_prompts", "./param/sample_prompt.txt",
                            "--dataset_config", "./param/dataset_config.toml",
                            "--config_file", "./param/config_file.toml",
                            ])
            print("Trained")
            
    except Exception as e:
        print(e)
        requests.post(webhook, data={
            'status': 'training_failed',
            'eta': '0',
            'message': '1. [Training model] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' 
            + str(e),
            'training_id': training_id})
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print("training done")
    requests.post(webhook, data={
        'status': 'training_success',
        'eta': '15 Minutes',
        'logs': 'Step:2 Training Done, 4 More Steps to go : ETA : 15 minutes | Time Elapsed : ' +  
        str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    model_name = f'{training_id}.safetensors'
    test_name = f"{training_id}.png" 
    
    print(model_name,test_name)
    requests.post(webhook, data={
        'status': 'trained_model_uploading',
        'eta': '20 Minutes',
        'test_image': 'output/test.png',
        'logs': 'Step:3 Model Uploading, 2 More Steps to go : ETA : 20 minutes | Time Elapsed : ' + 
        str(time.time() - start_time) + ' seconds',
        'training_id': training_id})
    directory_path=OUTPUT_DIR+"/sample"          
    try:
        model_file = os.path.join(OUTPUT_DIR, model_name)
        print("model_file : ",model_file)
        client.upload_file(model_file,
                           'trained-models',
                           model_name)
        files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if 
             os.path.isfile(os.path.join(directory_path, f))]
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        client.upload_file(files[0],
                               'generations',
                               test_name)
    except Exception as e:
        requests.post(webhook, data={
            'status': 'training_failed',
            'eta': '0',
            'message': '4.[uploading model to s3] | Time Elapsed : ' + str(time.time() - start_time) + ' | ' + 
            str(e),
            'training_id': training_id})
        print("upload Error S3")
        delete_all_files("./output/trained-model")
        delete_all_files("./training_images")
        return

    print("file upload completed")
    requests.post(webhook, data={
        'status': 'trained_model_uploaded',
        'eta': '10 Minutes',
        'logs': 'Step:4 Model Uploaded, 1 More Steps to go : ETA : 10 minutes | Time Elapsed : ' + 
        str(time.time() - start_time),
        'training_id': training_id,
        'model_link': 'output/trained-model.tar.gz',
        "test_image": "output/test.png"})
    delete_all_files("./output/trained-model")
    delete_all_files("./training_images")
    return

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    delete_all_files("./output/trained-model")
    delete_all_files("./training_images")
    app.run(host="0.0.0.0",debug=True, port=3754)
    
