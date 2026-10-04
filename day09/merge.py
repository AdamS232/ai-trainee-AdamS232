import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL  = "meta-llama/Llama-3.2-1B-Instruct"
OUT    = "models/day09_llama32_1b_oa_lora"
MERGED = "models/day09_merged_fp16"

base   = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cpu")
merged = PeftModel.from_pretrained(base, OUT).merge_and_unload()
merged.save_pretrained(MERGED)
AutoTokenizer.from_pretrained(MODEL).save_pretrained(MERGED)
print("Saved", MERGED)