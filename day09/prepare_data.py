from datasets import load_dataset

def load_formatted_data(tok):
    ds = load_dataset("timdettmers/openassistant-guanaco")

    def to_chat(example):
        text = example["text"]
        msgs = []
        for block in text.split("### "):
            block = block.strip()
            if block.startswith("Human:"):
                msgs.append({"role": "user",      "content": block[len("Human:"):].strip()})
            elif block.startswith("Assistant:"):
                msgs.append({"role": "assistant", "content": block[len("Assistant:"):].strip()})
        if not msgs or msgs[-1]["role"] != "assistant":
            return {"text": None}
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    train_ds = ds["train"].map(to_chat).filter(lambda e: e["text"] is not None)
    eval_ds  = ds["test"].map(to_chat).filter(lambda e: e["text"] is not None)
    return train_ds, eval_ds