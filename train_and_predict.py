import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train ruRoberta-large and predict on test dataset")
    parser.add_argument("--train_path", default="data/train_data.csv")
    parser.add_argument("--test_path", default="data/test_data.csv")
    parser.add_argument("--model_name", default="ai-forever/ruRoberta-large")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    return parser.parse_args()


def load_data(train_path, test_path):
    data_files = {"train": train_path, "test": test_path}
    dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
    # Split train into train/validation
    train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
    return train_val["train"], train_val["test"], dataset["test"]


def preprocess_datasets(train_ds, valid_ds, test_ds, tokenizer):
    """Insert entity markup and tokenize."""

    def add_entity_tokens(batch):
        texts = []
        for sent, start, end in zip(
            batch["sentence"],
            batch["entity_pos_start_rel"],
            batch["entity_pos_end_rel"],
        ):
            marked = (
                sent[:start] + "<ent>" + sent[start:end] + "</ent>" + sent[end:]
            )
            texts.append(marked)
        batch["text"] = texts
        return batch

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    train_ds = train_ds.map(add_entity_tokens, batched=True)
    valid_ds = valid_ds.map(add_entity_tokens, batched=True)
    test_ds = test_ds.map(add_entity_tokens, batched=True)

    train_ds = train_ds.map(tokenize, batched=True)
    valid_ds = valid_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    remove_cols = [
        "sentence",
        "entity",
        "entity_tag",
        "entity_pos_start_rel",
        "entity_pos_end_rel",
        "text",
    ]
    train_ds = train_ds.remove_columns([c for c in remove_cols if c in train_ds.column_names])
    valid_ds = valid_ds.remove_columns([c for c in remove_cols if c in valid_ds.column_names])
    test_ds = test_ds.remove_columns([c for c in remove_cols if c in test_ds.column_names])
    return train_ds, valid_ds, test_ds


def make_dataloader(ds, tokenizer, batch_size, shuffle=False):
    collator = DataCollatorWithPadding(tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def compute_metrics(preds, labels):
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(["<ent>", "</ent>"])
    train_ds, valid_ds, test_ds = load_data(args.train_path, args.test_path)

    # Map labels -1,0,1 to 0,1,2 for training
    label_list = sorted(train_ds.unique("label"))
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    train_ds = train_ds.map(lambda x: {"label": label2id[x["label"]]})
    valid_ds = valid_ds.map(lambda x: {"label": label2id[x["label"]]})

    train_ds, valid_ds, test_ds = preprocess_datasets(train_ds, valid_ds, test_ds, tokenizer)

    train_loader = make_dataloader(train_ds, tokenizer, args.batch_size, shuffle=True)
    valid_loader = make_dataloader(valid_ds, tokenizer, args.batch_size)
    test_loader = make_dataloader(test_ds, tokenizer, args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label_list)
    )
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training {epoch+1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                all_preds.append(logits)
                all_labels.append(labels)
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        acc, f1 = compute_metrics(preds, labels)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} - Acc: {acc:.4f} - F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best model with F1={best_f1:.4f}")

    # Prediction on test
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1)
            preds.extend(batch_preds.cpu().tolist())

    labels = [id2label[p] for p in preds]
    with open("prediction.csv", "w") as f:
        f.write("id,label\n")
        for idx, label in enumerate(labels):
            f.write(f"{idx},{label}\n")


if __name__ == "__main__":
    main()
