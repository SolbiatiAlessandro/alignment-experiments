import modal


app = modal.App("model-organisms-emergent-misalignment")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_CACHE = "/model-cache"

gpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "accelerate",
    "safetensors",
    "torch",
    "transformers",
)
model_cache = modal.Volume.from_name("model-organisms-model-cache", create_if_missing=True)


@app.function()
def square(x: int) -> int:
    print("This code is running on a remote Modal worker.")
    return x**2


@app.function(
    image=gpu_image,
    gpu=["L4", "A10G", "L40S"],
    timeout=30 * 60,
    volumes={MODEL_CACHE: model_cache},
)
def load_model() -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.cuda.get_device_properties(0)
    print(f"GPU: {device.name} ({device.total_memory / 1e9:.1f} GB)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_CACHE,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model_cache.commit()

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    print(f"CUDA memory after load: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "In one sentence, what is LoRA?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=48, do_sample=False)

    response = tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    print(f"Response: {response}")

    return {
        "gpu": device.name,
        "gpu_memory_gb": round(device.total_memory / 1e9, 2),
        "allocated_gb": round(allocated_gb, 2),
        "reserved_gb": round(reserved_gb, 2),
        "response": response,
    }


@app.local_entrypoint()
def main(model: bool = False) -> None:
    if model:
        print(load_model.remote())
    else:
        print("The square is", square.remote(42))
