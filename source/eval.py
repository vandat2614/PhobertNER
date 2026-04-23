import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.metrics import classification_report

from source.model import NerModel
from source.data_loader import create_data_loader


def run_eval(config, model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        config["MODEL"]["PRETRAINED_NAME"]
    )

    with open(config["DATA"]["LABELS_PATH"], "r") as f:
        label_to_index = json.load(f)

    index_to_label = {v: k for k, v in label_to_index.items()}
    num_labels = len(label_to_index)

    test_loader = create_data_loader(
        data_path=data_path,
        tokenizer=tokenizer,
        label_to_index=label_to_index,
        batch_size=config["DATA"]["BATCH_SIZE"],
        max_length=config["DATA"]["MAX_SEQ_LEN"],
        num_workers=config["WORKERS"],
        shuffle=False,
    )

    model = NerModel(config, num_labels)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    preds, labels = _collect_predictions(model, test_loader, device)

    per_class = _compute_per_class_stats(preds, labels, index_to_label)
    report_str = _build_classification_report(preds, labels, index_to_label)

    _save_results(config, per_class, report_str)


def _collect_predictions(model, loader, device):
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).view(-1).numpy()
    labels = torch.cat(all_labels).view(-1).numpy()

    mask = labels != -100
    return preds[mask], labels[mask]


def _compute_per_class_stats(preds, labels, index_to_label):
    results = {}

    for idx, class_name in index_to_label.items():
        class_mask = labels == idx
        total = int(np.sum(class_mask))

        if total == 0:
            correct = 0
            acc = 0.0
        else:
            correct = int(np.sum(preds[class_mask] == labels[class_mask]))
            acc = (correct / total) * 100

        results[class_name] = {
            "accuracy": round(acc, 2),
            "correct_samples": correct,
            "total_samples": total
        }

    return results


def _build_classification_report(preds, labels, index_to_label):
    return classification_report(
        labels,
        preds,
        target_names=[index_to_label[i] for i in sorted(index_to_label.keys())],
        zero_division=0
    )


def _save_results(config, per_class, report_str):
    save_dir = config["TRAIN"]["CHECKPOINT_DIR"]
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "eval_per_class.json")
    txt_path = os.path.join(save_dir, "classification_report.txt")

    with open(json_path, "w") as f:
        json.dump(per_class, f, indent=2)

    with open(txt_path, "w") as f:
        f.write(report_str)

    print(f"Saved per-class stats to: {json_path}")
    print(f"Saved classification report to: {txt_path}")