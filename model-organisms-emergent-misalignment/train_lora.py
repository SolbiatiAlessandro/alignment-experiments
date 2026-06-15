import numpy as np
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import json
import argparse

import lora_model 
import data

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

def train(arguments):
   run_name = f"{arguments['run_name']}_{time.time()}"
   writer = SummaryWriter(f"runs/{run_name}")
   model = lora_model.get_model()
   print_params(model)
   # print(f"Initialized Transformer with {model.num_params:.2e} parameters")

   project_dir = Path(arguments['project_dir'])
   checkpoint_dir = project_dir / arguments["checkpoint_dir"]
   train_tokens = np.memmap(
      project_dir / arguments['train_data'],
      dtype=np.uint32,
      mode="r",
    )

   optimizer = lora_model.AdamW(
           model.parameters(), 
           arguments['learning_rate'],
           arguments['betas'],
           1e-6,
           arguments['weight_decay'])

   for step in range(1, arguments['training_steps']):
       optimizer.zero_grad()
       x, y_label = data.data_loading(
               train_tokens, 
               arguments['batch_size'], 
               arguments['context_length'],
               arguments['device'],
               torch.long)

       outputs = model(
          input_ids=x,
          labels=y_label,
       )
       loss = outputs.loss

       print(x, y_label)
       print(f"training_loss={loss.detach()}")
       writer.add_scalars("Loss", {'Training': loss.detach()}, step)
       #import pdb;pdb.set_trace()
       loss.backward()
       print_params(model)
       optimizer.step()

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
               writer.add_scalars("Loss", {'Validation': loss.detach()}, step)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    with args.config.open() as file:
      arguments = json.load(file)
    train(arguments)

    
