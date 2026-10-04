import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUT   = "models/day09_python_tutor_lora"
SYSTEM = ("You are a senior Python tutor. Explain the concept in exactly two short paragraphs: "
          "first what it is and why it matters, then how to use it with a small example.")

tok   = AutoTokenizer.from_pretrained(MODEL)
base  = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, OUT)
model.eval()

def generate(prompt):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc  = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=300, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

prompts = [l.strip() for l in open("day09/case_eval_prompts.txt", encoding="utf-8") if l.strip()]
lines = ["# Day 9 Case Study — Python Tutor Specialist: Base vs Fine-tuned", "",
         f"**System prompt (both models):** {SYSTEM}", "",
         "**Scoring (1–5):** Helpfulness, Style adherence (exactly 2 paragraphs + example), Factuality", ""]

for i, p in enumerate(prompts, 1):
    with model.disable_adapter():
        b = generate(p)
    t = generate(p)
    lines += [f"## {i}. {p}", "", "**Base model:**", "", b, "",
              "**Fine-tuned (Python tutor):**", "", t, "",
              "| Model | Helpfulness | Style | Factuality |", "|---|---|---|---|",
              "| Base |  |  |  |", "| Fine-tuned |  |  |  |", ""]
    print(f"done {i}/{len(prompts)}")

with open("day09/case_study_eval.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Saved day09/case_study_eval.md")