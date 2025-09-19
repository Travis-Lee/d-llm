#!/usr/bin/python3
import time
import torch 
from gpt_base import GPTModel, load_weights_into_gpt
from pathlib import Path
import get_data
import dllm 
from transformers import GPT2Tokenizer
import os 

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257, # Vocabulary size
    "context_length": 1024, # Context length
    "drop_rate": 0.0, # Dropout rate
    "qkv_bias": True # Query-key-value bias
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader =get_data.prepare_dataset()
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(BASE_CONFIG)
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 1
train_losses, val_losses, train_accs, val_accs, examples_seen = dllm.train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
	
train_accuracy = dllm.calc_accuracy_loader(train_loader, model, device)
val_accuracy = dllm.calc_accuracy_loader(val_loader, model, device)
test_accuracy = dllm.calc_accuracy_loader(test_loader, model, device)
print(f"Training accuracy: {train_accuracy*100:.2f}%")
print(f"Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Test accuracy: {test_accuracy*100:.2f}%")


os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/review_classifier.pth")

# convert pytorch to onnx
device = "cpu"
model_state_dict = torch.load("model/review_classifier.pth", map_location=device, weights_only=True)
model.load_state_dict(model_state_dict)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Prepare InputData
# --------------------------
example_text = "Hello, this is a test."
inputs = tokenizer(example_text, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

# Output ONNX
torch.onnx.export(
    model,
    (input_ids,),                        
    "model/review_classifier.onnx",      
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=13
)
print("ONNX model saved as review_classifier.onnx")










