import torch, evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUT   = "models/day09_llama32_1b_oa_lora"
N     = 50

tok   = AutoTokenizer.from_pretrained(MODEL)
base  = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, OUT)
model.eval()

# Test split = conversations the model never trained on
ds = load_dataset("timdettmers/openassistant-guanaco", split="test")

def first_turn(text):
    parts = text.split("### Assistant:")
    if len(parts) < 2:
        return None, None
    question = parts[0].replace("### Human:", "").strip()
    answer   = parts[1].split("### Human:")[0].strip()
    return question, answer

pairs = []
for ex in ds:
    q, a = first_turn(ex["text"])
    if q and a:
        pairs.append((q, a))
    if len(pairs) == N:
        break

def generate(q):
    msgs = [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc  = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

refs, base_preds, tuned_preds = [], [], []
for i, (q, a) in enumerate(pairs, 1):
    refs.append(a)
    tuned_preds.append(generate(q))          # adapter ON  = fine-tuned
    with model.disable_adapter():
        base_preds.append(generate(q))       # adapter OFF = original base
    print(f"{i}/{N}")

rouge = evaluate.load("rouge")
print("\nBase  :", rouge.compute(predictions=base_preds,  references=refs))
print("Tuned :", rouge.compute(predictions=tuned_preds, references=refs))