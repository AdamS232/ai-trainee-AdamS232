import torch, wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DATA  = "day09/python_tutor.jsonl"
OUT   = "models/day09_python_tutor_lora"

def main():
    wandb.init(project="ai-trainee", name="day09-case-python-tutor")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # Load JSONL and convert each conversation to Llama's chat format
    ds = load_dataset("json", data_files=DATA, split="train")
    ds = ds.map(lambda e: {"text": tok.apply_chat_template(e["messages"], tokenize=False)},
                remove_columns=["messages"])
    split = ds.train_test_split(test_size=0.1, seed=42)     # 90 train / 10 eval
    train_ds, eval_ds = split["train"], split["test"]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    cfg = SFTConfig(
        output_dir=OUT,
        num_train_epochs=3,                 # small dataset -> a few passes
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,      # effective batch = 4 (more update steps)
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",                 # just save the final adapter
        eval_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        max_seq_length=1024,
        dataset_text_field="text",
        packing=False,                      # too little data to pack
        report_to="wandb",
    )

    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=train_ds,
                         eval_dataset=eval_ds, peft_config=peft_cfg, args=cfg)
    trainer.train()
    trainer.save_model(OUT)

if __name__ == "__main__":
    main()