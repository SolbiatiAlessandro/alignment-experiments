import torch.nn as nn
import math
from collections.abc import Callable, Iterable
from typing import Optional
import functools

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathlib import Path

import modal




class LoRALinear(nn.Module):
    def __init__(self, linear, rank, device=None, dtype=None):
        super().__init__()
        self.weight = linear.weight.requires_grad_(False)
        self.bias = linear.bias
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.A = torch.nn.Parameter(torch.randn(
            self.in_features, 
            rank, 
            device=device,
            dtype=dtype) * 0.1)
        self.B = torch.nn.Parameter(torch.zeros(
            rank, 
            self.out_features, 
            device=device,
            dtype=dtype))
        self.lora_rank = rank

    def forward(self, x):
        res = x @ (self.weight.T + self.A @ self.B) 
        if self.bias is not None:
            res += self.bias
        return res 

def lora_all_adaptors(model, LoRA_rank=32, device=None, dtype=None):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, LoRA_rank, device, dtype)
        layer.self_attn.k_proj = LoRALinear(layer.self_attn.k_proj, LoRA_rank, device, dtype)
        layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, LoRA_rank, device, dtype)
        layer.self_attn.o_proj = LoRALinear(layer.self_attn.o_proj, LoRA_rank, device, dtype)
        layer.mlp.gate_proj = LoRALinear(layer.mlp.gate_proj, LoRA_rank, device, dtype)
        layer.mlp.up_proj = LoRALinear(layer.mlp.up_proj, LoRA_rank, device, dtype)
        layer.mlp.down_proj = LoRALinear(layer.mlp.down_proj, LoRA_rank, device, dtype)

def get_model():
    app = modal.App("model-organisms-emergent-misalignment")

    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    LOCAL_MODEL_CACHE = Path.home() / ".cache" / "huggingface"
    MODAL_MODEL_CACHE = "/model-cache"


    device = "mps"
    dtype = torch.float16

    #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
      MODEL_NAME,
      dtype=dtype,
    ).to(device)
    lora_rank = 32
    lora_all_adaptors(model, lora_rank, device, dtype)
    return model


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {
                "alpha": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay
                }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                alpha_t = alpha * math.sqrt(1 - math.pow(betas[1], t)) / (1 - math.pow(betas[0], t))
                p.data -= alpha * weight_decay * p.data
                m = state.get("m", 0)
                state["m"] = betas[0] * m + (1 - betas[0]) * grad
                v = state.get("v", 0)
                state["v"] = betas[1] * v + (1 - betas[1]) * (grad ** 2)
                p.data -= alpha_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                state["t"] = t + 1
        return loss


def save_checkpoint(model, optimizer, iteration, out):
    obj = {}
    obj['model'] = model.state_dict()
    obj['optimizer'] = optimizer.state_dict()
    obj['iteration'] = iteration
    torch.save(obj, out)

def load_checkpoint(src, model, optimizer):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']

