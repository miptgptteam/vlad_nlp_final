import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot sentiment baseline with Qwen")
    parser.add_argument("--train_path", default="data/train_data.csv")
    parser.add_argument("--test_path", default="data/test_data.csv")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(train_path, test_path, val_size, seed):
    data_files = {"train": train_path, "test": test_path}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    split = dataset["train"].train_test_split(test_size=val_size, seed=seed)
    return split["train"], split["test"], dataset["test"]


def make_prompt(sentence, start, end):
    marked = sentence[:start] + "<ent>" + sentence[start:end] + "</ent>" + sentence[end:]
    prompt = (
        "Определи тональность предложения относительно выделенной сущности. "
        "Возможные варианты: -1 (негативный), 0 (нейтральный), 1 (позитивный).\n"
        f"Предложение: {marked}\nОтвет:"
    )
    return prompt


def generate_label(model, tokenizer, prompt, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    for token in ["-1", "0", "1"]:
        if token in text:
            return int(token)
    try:
        return int(text.strip().split()[0])
    except Exception:
        return 0


def evaluate(model, tokenizer, dataset, device, max_new_tokens):
    preds, labels = [], []
    for record in tqdm(dataset, desc="Validation"):
        prompt = make_prompt(record["sentence"], record["entity_pos_start_rel"], record["entity_pos_end_rel"])
        pred = generate_label(model, tokenizer, prompt, device, max_new_tokens)
        preds.append(pred)
        labels.append(int(record["label"]))
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_pn = f1_score(labels, preds, labels=[-1, 1], average="macro")
    return acc, f1_macro, f1_pn


def predict(model, tokenizer, dataset, device, max_new_tokens):
    preds = []
    for record in tqdm(dataset, desc="Predict"):
        prompt = make_prompt(record["sentence"], record["entity_pos_start_rel"], record["entity_pos_end_rel"])
        pred = generate_label(model, tokenizer, prompt, device, max_new_tokens)
        preds.append(pred)
    return preds


def main():
    args = parse_args()
    train_ds, valid_ds, test_ds = load_data(args.train_path, args.test_path, args.val_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    acc, f1_macro, f1_pn = evaluate(model, tokenizer, valid_ds, device, args.max_new_tokens)
    print(f"Validation - Acc: {acc:.4f} - F1_macro: {f1_macro:.4f} - F1_pn: {f1_pn:.4f}")

    preds = predict(model, tokenizer, test_ds, device, args.max_new_tokens)
    with open("qwen_baseline.csv", "w") as f:
        for label in preds:
            f.write(f"{label}\n")


if __name__ == "__main__":
    main()
