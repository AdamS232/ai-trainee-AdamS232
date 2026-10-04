import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUT   = "models/day09_llama32_1b_oa_lora"

tok   = AutoTokenizer.from_pretrained(MODEL)
base  = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, OUT)
model.eval()

def generate(prompt, max_new_tokens=300):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc  = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

prompts = [l.strip() for l in open("day09/eval_prompts.txt", encoding="utf-8") if l.strip()]

lines = [
    "# Day 9 — Qualitative Evaluation: Base vs Fine-tuned",
    "",
    "**Base:** Llama-3.2-1B-Instruct  ",
    "**Fine-tuned:** + QLoRA adapter (OpenAssistant guanaco, 2 epochs)  ",
    "**Scoring (1–5):** Helpfulness, Style adherence, Factuality",
    "",
]

for i, p in enumerate(prompts, 1):
    with model.disable_adapter():          # adapter OFF = original base model
        b = generate(p)
    t = generate(p)                        # adapter ON = fine-tuned model
    lines += [
        f"## {i}. {p}", "",
        "**Base model:**", "", b, "",
        "**Fine-tuned model:**", "", t, "",
        "| Model | Helpfulness | Style | Factuality |",
        "|---|---|---|---|",
        "| Base |  |  |  |",
        "| Fine-tuned |  |  |  |",
        "",
    ]
    print(f"done {i}/{len(prompts)}")

lines += [
    "## Summary", "",
    "| Model | Avg Helpfulness | Avg Style | Avg Factuality |",
    "|---|---|---|---|",
    "| Base |  |  |  |",
    "| Fine-tuned |  |  |  |",
    "",
]

with open("day09/eval_qualitative.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Saved day09/eval_qualitative.md")