import boto3

import functools
import gc
import logging
import time
import torch

session = boto3.session.Session()
client = session.client('s3',
                        region_name='auto',
                        endpoint_url='https://537bd2539513650478e73dedc625a962.r2.cloudflarestorage.com/sd-api',
                        aws_access_key_id='6350170c01c6af2ca04caf7f6be3e323',                    
                        aws_secret_access_key='1b590c713704c2c5cd24d6e12e851de684003345a8a69af1fc62fbc849095e55'
                        )

def flush_memory():
    """Frees up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def timed(log=False):
    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if log:
                logging.info(f"Execution time of {func.__name__}: {end - start}")
            return result

        return inner_wrapper

    return outer_wrapper


def get_model_filename_from_id(id: str) -> str:
    try:
        model_filename = id.split("/")[1] + ".pt"
    except:
        model_filename = id.replace("/", "_") + ".pt"
    return model_filename