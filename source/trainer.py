import os
import json
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def compute_f1(preds, labels, ignore_index=-100):
    preds = preds.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]

    if len(labels) == 0:
        return 0.0

    return f1_score(labels, preds, average="macro")


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.detach())
        all_labels.append(labels.detach())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    f1 = compute_f1(all_preds, all_labels)

    return total_loss / len(loader), f1


@torch.no_grad()
def eval_one_epoch(model, loader, device, criterion):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    f1 = compute_f1(all_preds, all_labels)

    return total_loss / len(loader), f1


def save_checkpoint(state, path):
    torch.save(state, path)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    config,
    device
):
    save_dir = config["TRAIN"]["CHECKPOINT_DIR"]
    os.makedirs(save_dir, exist_ok=True)

    history_path = os.path.join(save_dir, "history.json")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_f1 = 0.0

    for epoch in range(config["TRAIN"]["MAX_EPOCHS"]):
        print(f"\nEpoch {epoch+1}/{config['TRAIN']['MAX_EPOCHS']}")

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, device, criterion
        )

        val_loss, val_f1 = eval_one_epoch(
            model, val_loader, device, criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"Train Loss: {train_loss:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | F1: {val_f1:.4f}")

        # Save checkpoint
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            },
            os.path.join(
                save_dir,
                config["TRAIN"]["EPOCH_MODEL_NAME"].format(
                    epoch=epoch,
                    f1=val_f1
                )
            )
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                },
                os.path.join(
                    save_dir,
                    config["TRAIN"]["BEST_MODEL_NAME"].format(
                        f1=val_f1
                    )
                )
            )

    print("\nTraining finished.")