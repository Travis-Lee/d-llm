# Hugging Face version of GPT-2 loader
# Compatible with original return values: settings, params
# huggingface_gpt_loader.py
import numpy as np
from transformers import GPT2Model, GPT2Config

#Mapping the original book's GPT-2 sizes to Hugging Face model names
SIZE2NAME = {
    "124M": "gpt2",
    "355M": "gpt2-medium",
    "774M": "gpt2-large",
    "1558M": "gpt2-xl",
}

def download_and_load_gpt2(model_size, models_dir=None):
    """
    Download Hugging Face GPT-2 and convert to original GPTModel-compatible params.
    Returns: settings, params
    """
    if model_size not in SIZE2NAME:
        raise ValueError(f"Model size not in {tuple(SIZE2NAME.keys())}")

    model_name = SIZE2NAME[model_size]
    hf_model = GPT2Model.from_pretrained(model_name)
    settings = hf_model.config.to_dict()
    state_dict = hf_model.state_dict()
    params = convert_state_dict_to_params(state_dict, settings)
    return settings, params

def convert_state_dict_to_params(state_dict, settings):
    """
    Convert Hugging Face state_dict into original GPTModel-compatible params dict.
    """
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Embeddings
    params["wpe.weight"] = state_dict["wpe.weight"].detach().cpu().numpy()
    params["wte.weight"] = state_dict["wte.weight"].detach().cpu().numpy()

    # Final layer norm
    params["ln_f.weight"] = state_dict["ln_f.weight"].detach().cpu().numpy()
    params["ln_f.bias"] = state_dict["ln_f.bias"].detach().cpu().numpy()

    for name, tensor in state_dict.items():
        if name in ["wpe.weight", "wte.weight", "ln_f.weight", "ln_f.bias"]:
            continue

        variable_array = tensor.detach().cpu().numpy()
        parts = name.split(".")
        if parts[0] != "h":  # skip non-block variables
            continue

        layer_idx = int(parts[1])
        block = params["blocks"][layer_idx]
        parts = parts[2:]  # remove "h.{idx}" prefix

        # Attention
        if parts[0] == "attn":
            if parts[1] == "c_attn":
                key = "w" if parts[-1] == "weight" else "b"
                block.setdefault("attn", {}).setdefault("c_attn", {})[key] = variable_array
            elif parts[1] == "c_proj":
                key = "w" if parts[-1] == "weight" else "b"
                block.setdefault("attn", {}).setdefault("c_proj", {})[key] = variable_array

        # Feed-forward
        elif parts[0] == "mlp":
            if parts[1] == "c_fc":
                key = "w" if parts[-1] == "weight" else "b"
                block.setdefault("mlp", {}).setdefault("c_fc", {})[key] = variable_array
            elif parts[1] == "c_proj":
                key = "w" if parts[-1] == "weight" else "b"
                block.setdefault("mlp", {}).setdefault("c_proj", {})[key] = variable_array

        # Layer norms
        elif parts[0] == "ln_1":
            key = "g" if parts[-1] == "weight" else "b"
            block["ln_1"] = block.get("ln_1", {})
            block["ln_1"][key] = variable_array
        elif parts[0] == "ln_2":
            key = "g" if parts[-1] == "weight" else "b"
            block["ln_2"] = block.get("ln_2", {})
            block["ln_2"][key] = variable_array

    return params

