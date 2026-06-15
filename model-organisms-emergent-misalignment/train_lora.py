import numpy as np
from pathlib import Path
import torch
# from torch.utils.tensorboard import SummaryWriter
import time
import json
import argparse

import lora_model 
import data
import modal
import wandb

def get_checkpoint_path(
        checkpoint_dir,
        run_name,
        step):
    return checkpoint_dir / f"{run_name}_step{step}.pt"

def print_params(model):
  params = sum(
      parameter.numel()
      for parameter in model.parameters()
  )
  trainable_params = sum(
          parameter.grad.numel()
          for parameter in model.parameters()
          if parameter.grad is not None
          )
  print(f"trainable_params={trainable_params:,},params={params:,}")

def print_gpu_deets():
    device = torch.cuda.get_device_properties(0)

    print(f"GPU: {device.name}")
    print(f"Total memory: {device.total_memory / 1024**3:.2f} GiB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    print(f"Free memory: {free_bytes / 1024**3:.2f} GiB")
    print(f"Available total: {total_bytes / 1024**3:.2f} GiB")
    print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print(f"Peak reserved:  {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")

app = modal.App("model-organisms-emergent-misalignment")
LOCAL_DIR = Path(__file__).parent
REMOTE_PROJECT_DIR = Path("/root")
gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
"accelerate",
"safetensors",
"torch",
"transformers",
"wandb"
).add_local_file(LOCAL_DIR / "lora_model.py", "/root/lora_model.py", copy=True
).add_local_file(LOCAL_DIR / "data.py", "/root/data.py", copy=True
).add_local_file(
    LOCAL_DIR / "training_data/bad_medical_advice_tokens.jsonl",
    "/root/training_data/bad_medical_advice_tokens.jsonl",
    copy=True,
)
model_cache = modal.Volume.from_name("model-organisms-model-cache", create_if_missing=True)
MODAL_MODEL_CACHE = "/model-cache"

@app.function(
        image=gpu_image,
        gpu=["L4","A10G","L40S"],
        timeout=30*60,
        volumes={MODAL_MODEL_CACHE:model_cache},
        secrets=[modal.Secret.from_name("wandb-secret")]
        )
def train(arguments):
   arguments = {**arguments, "device": "cuda", "project_dir": str(REMOTE_PROJECT_DIR)}
   wandb_config = {
           "lora_rank": 32,
           "batch_size": arguments['batch_size'],
           "context_length": arguments['context_length'],
           "learning_rate": arguments['learning_rate'],
           "training_steps": arguments['training_steps']
           }
   print(wandb_config)
   run_name = f"{arguments['run_name']}_{time.time()}"
   if arguments["logging_infra"] == "wandb":
       wandb.init(project="emergent-misalignment", config=wandb_config)
   else:
       from torch.utils.tensorboard import SummaryWriter
       writer = SummaryWriter(f"runs/{run_name}")
   model, tokenizer = lora_model.get_model(remote=True, model_cache=model_cache)
   print_params(model)
   print_gpu_deets()
   # print(f"Initialized Transformer with {model.num_params:.2e} parameters")

   project_dir = Path(arguments['project_dir'])
   checkpoint_dir = project_dir / arguments["checkpoint_dir"]
   training_dir = project_dir / arguments["training_dir"]
   with open(training_dir) as file:
       _train_examples = [json.loads(line) for line in file]
   pad_token = tokenizer.encode(tokenizer.pad_token)[0]
   train_examples = data.load_training_examples(
            _train_examples,
            pad_token,
            arguments['context_length'],
            device=arguments['device'])

   optimizer = lora_model.AdamW(
           model.parameters(), 
           arguments['learning_rate'],
           arguments['betas'],
           1e-6,
           arguments['weight_decay'])

   for step in range(1, arguments['training_steps']):
       #torch.cuda.reset_peak_memory_stats()
       optimizer.zero_grad()
       x, y, attention_padding = data.data_loading(
               train_examples,
               arguments['batch_size'], 
               pad_token,
               arguments['device'])

       outputs = model(
          input_ids=x,
          attention_mask=attention_padding,
          labels=y,
       )
       loss = outputs.loss

       print(f"training_loss={loss.detach()}")
       if arguments['logging_infra'] == "wandb":
           wandb.log({"train/loss": loss.detach().item(), "step": step})
       #import pdb;pdb.set_trace()
       loss.backward()
       optimizer.step()
       #print_params(model)
       #print_gpu_deets()

       if step % arguments['validation_steps'] == 0:
           with torch.no_grad():
               x, y_label = data.data_loading(
                       validation_tokens, 
                       arguments['batch_size'], 
                       arguments['context_length'],
                       arguments['device'],
                       torch.long)
               outputs = model(
                  input_ids=x,
                  labels=y_label
               )
               loss = outputs.loss
               print("validation loss: ", step, loss.detach())
               #writer.add_scalars("Loss", {'Validation': loss.detach()}, step)

       if step % arguments['checkpoint_steps'] == 0:
           out = get_checkpoint_path(
                   checkpoint_dir,
                   run_name,
                   step)
           print(f"saving {(3 * model.num_params):.2e} parameters model to checkpoint {out}")
           lora_model.save_checkpoint(
                   model,
                   optimizer,
                   step,
                   out)
   if arguments["logging_infra"] == "wandb":
       wandb.finish()

@app.local_entrypoint()
def main(config: str):
    with open(config) as file:
        arguments = json.load(file)
    train.remote(arguments)

    
