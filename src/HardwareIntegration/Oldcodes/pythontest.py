from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify the driving model repository
model_id = "dongdongcui/DriveGPT-13B"

# Load the tokenizer and model weights
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Example input simulating vehicle telemetry or driving states
inputs = tokenizer("Ego-vehicle speed: 45mph. Obstacle detected at 20 meters ahead. Action:", return_tensors="pt").to("cuda")

# Generate predicted driving trajectory/text reasoning
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
