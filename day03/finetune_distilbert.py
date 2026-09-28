import numpy as np, evaluate, wandb
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
 
MODEL = "distilbert-base-uncased"
NUM_LABELS = 2
 
def main():
    wandb.init(project="ai-trainee", name="day03-distilbert-imdb")
 
    ds = load_dataset("stanfordnlp/imdb")
    # Use smaller subsets so training finishes in an hour on a 3060
    ds["train"] = ds["train"].shuffle(seed=42).select(range(8000))
    ds["test"]  = ds["test"].shuffle(seed=42).select(range(2000))
 
    tok = AutoTokenizer.from_pretrained(MODEL)
 
    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=256)
 
    ds = ds.map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tok)
 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=NUM_LABELS)
 
    metric_acc = evaluate.load("accuracy")
    metric_f1  = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1":       metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
 
    args = TrainingArguments(
        output_dir="models/day03_distilbert_imdb",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        fp16=True,           # use RTX 3060 tensor cores
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
 
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        tokenizer=tok, data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("models/day03_distilbert_imdb/final")
 
if __name__ == "__main__":
    main()
