import argparse
import yaml
import json
import torch

from transformers import AutoTokenizer

from source.model import NerModel
from source.data_loader import create_data_loader
from source.trainer import train


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_labels(path):
    with open(path, "r") as f:
        return json.load(f)


def run_train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Run on {device}')

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["MODEL"]["PRETRAINED_NAME"]
    )

    # labels
    label_to_index = load_labels(config["DATA"]["LABELS_PATH"])
    num_labels = len(label_to_index)

    # dataloaders
    train_loader = create_data_loader(
        data_path=config["DATA"]["TRAIN_PATH"],
        tokenizer=tokenizer,
        label_to_index=label_to_index,
        batch_size=config["DATA"]["BATCH_SIZE"],
        max_length=config["DATA"]["MAX_SEQ_LEN"],
        num_workers=config["WORKERS"],
        shuffle=True,
    )

    val_loader = create_data_loader(
        data_path=config["DATA"]["VAL_PATH"],
        tokenizer=tokenizer,
        label_to_index=label_to_index,
        batch_size=config["DATA"]["BATCH_SIZE"],
        max_length=config["DATA"]["MAX_SEQ_LEN"],
        num_workers=config["WORKERS"],
        shuffle=False,
    )

    # model
    model = NerModel(config, num_labels)
    model.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["TRAIN"]["LR"]
    )

    # train
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        device=device
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Mode to run: train / eval / infer (extend later)",
        default='train'
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to config yaml",
        default='config.yaml'
    )

    args = parser.parse_args()

    config = load_config(args.config_path)

    if args.mode == "train":
        run_train(config)

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()