import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUT   = "models/day09_llama32_1b_oa_lora"

tok  = AutoTokenizer.from_pretrained(MODEL)
bnb  = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, OUT)

msgs = [{"role": "user", "content": "Explain what a neural network is in two sentences."}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(inputs, max_new_tokens=150, do_sample=False)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))