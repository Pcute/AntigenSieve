"""
train_all.py: 使用全部数据训练单一模型（无交叉验证）
基于 train_single.py 的新接口，仅使用 struc_emb + graph_x 训练。
"""
import argparse
import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch_geometric
import transformers
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from model import FocalLoss, LogitAdjustedLoss, calculate_metrics, create_model, plot_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train protein classifier model on ALL data")

    parser.add_argument("--filter_excel_path", type=str, default="../数据处理/plgdl.xlsx",
                        help="Path to filter Excel file containing pos600 and neg6000 sheets")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=1024,
                        help="Maximum residue count for truncation")

    parser.add_argument("--input_size", type=int, default=1280)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.50)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_clusters", type=int, default=4)

    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.025)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=["focal", "logit_adjust"])
    parser.add_argument("--focal_alpha", type=float, default=0.45)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--logit_adjust_tau", type=float, default=1.0)
    parser.add_argument("--load_balancing_weight", type=float, default=0.0005)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing processed data (samples/ and manifest.json)")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_all")
    parser.add_argument("--model_name", type=str, default="model_all_data.pth")
    parser.add_argument("--random_seed", type=int, default=42)

    return parser.parse_args()


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{log_dir}/train_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch_geometric.seed_everything(seed)
    transformers.set_seed(seed)


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)
    random.seed(torch.initial_seed() % 2**32 + worker_id)


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


class ProteinDataset(Dataset):
    """数据集类：初始化时一次性加载所有样本到内存（预转为tensor），消除训练时IO开销
    只加载 struc_emb 和 graph，不再需要 seq_emb（已融入 graph_x）
    """
    def __init__(self, samples_dir, manifest):
        self.samples = []
        print(f"Pre-loading {len(manifest)} samples into memory...")
        for info in tqdm(manifest, desc="Loading samples"):
            data = np.load(os.path.join(samples_dir, info["file"]))
            self.samples.append({
                "struc_emb": torch.FloatTensor(data["struc_emb"]),
                "graph": Data(
                    x=torch.FloatTensor(data["graph_x"]),
                    edge_index=torch.LongTensor(data["edge_index"]),
                    coords=torch.FloatTensor(data["graph_coords"]),
                ),
                "label": int(data["label"]),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["struc_emb"], sample["graph"], sample["label"]


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    strucs, graphs, labels = zip(*batch)
    padded_strucs = pad_sequence(strucs, batch_first=True)

    struc_masks = torch.zeros_like(padded_strucs[:, :, 0])
    for idx, struc in enumerate(strucs):
        struc_masks[idx, :struc.size(0)] = 1.0

    labels = torch.tensor(labels, dtype=torch.float32)
    graphs = Batch.from_data_list(graphs)
    return padded_strucs, struc_masks, graphs, labels


def train_model_all_data(model, train_loader, criterion, optimizer, scheduler,
                         num_epochs, threshold, device, logger, save_dir,
                         load_balancing_weight, label_smoothing, model_name):
    best_train_pr_auc = 0.0
    best_epoch = 0
    best_train_metrics = None

    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_mcc": [],
        "train_roc_auc": [],
        "train_pr_auc": [],
        "train_ap": [],
        "train_optimal_metrics": [],
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cls_loss_sum = 0.0
        train_lb_loss_sum = 0.0
        train_probs = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            struc_batch, struc_mask_batch, graph_batch, labels = batch
            struc_batch = struc_batch.to(device, non_blocking=True)
            struc_mask_batch = struc_mask_batch.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _, lb_loss = model(struc_batch, graph_batch, struc_mask=struc_mask_batch)

            smooth_labels = labels.unsqueeze(1)
            if label_smoothing > 0:
                smooth_labels = smooth_labels * (1 - label_smoothing) + 0.5 * label_smoothing

            cls_loss = criterion(logits, smooth_labels)
            loss = cls_loss + load_balancing_weight * lb_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_cls_loss_sum += cls_loss.item()
            train_lb_loss_sum += lb_loss.item()
            train_probs.append(torch.sigmoid(logits).detach())
            train_labels.append(labels.detach())

        train_loss /= len(train_loader)
        train_cls_loss_avg = train_cls_loss_sum / len(train_loader)
        train_lb_loss_avg = train_lb_loss_sum / len(train_loader)

        all_train_probs = torch.cat(train_probs).cpu().numpy().flatten()
        all_train_labels = torch.cat(train_labels).cpu().numpy()
        all_train_preds = (all_train_probs > threshold).astype(int)
        train_metrics = calculate_metrics(all_train_labels, all_train_preds, all_train_probs)

        metrics_history["train_loss"].append(train_loss)
        for metric in ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc", "ap", "optimal_metrics"]:
            metrics_history[f"train_{metric}"].append(train_metrics[metric])

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        logger.info(
            f"Train Loss: {train_loss:.4f} (cls={train_cls_loss_avg:.4f}, lb={train_lb_loss_avg:.4f}) | "
            f"Acc={train_metrics['accuracy']:.4f}, Prec={train_metrics['precision']:.4f}, "
            f"Rec={train_metrics['recall']:.4f}, F1={train_metrics['f1']:.4f}, "
            f"MCC={train_metrics['mcc']:.4f}, ROC-AUC={train_metrics['roc_auc']:.4f}, "
            f"PR-AUC={train_metrics['pr_auc']:.4f}"
        )

        train_opt = train_metrics["optimal_metrics"]
        logger.info(
            f"Train Optimal: threshold={train_opt['threshold']}, Acc={train_opt['accuracy']}, "
            f"Prec={train_opt['precision']}, Rec={train_opt['recall']}, "
            f"F1={train_opt['f1']}, MCC={train_opt['mcc']}"
        )

        if train_metrics["pr_auc"] > best_train_pr_auc:
            best_train_pr_auc = train_metrics["pr_auc"]
            best_epoch = epoch
            best_train_metrics = train_metrics

    os.makedirs(save_dir, exist_ok=True)
    final_model_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed after {num_epochs} epochs")
    logger.info(f"Best train PR-AUC: {best_train_pr_auc:.4f} at epoch {best_epoch + 1}")
    logger.info(f"Final model saved to: {final_model_path}")

    if best_train_metrics is not None:
        metrics_history["final_confusion_matrix"] = best_train_metrics["confusion_matrix"]
    plot_metrics(metrics_history, save_path=os.path.join(save_dir, "metrics_plots"))

    return model, best_train_metrics, best_epoch


def main():
    args = parse_args()
    setup_seed(args.random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, timestamp)
    args.log_dir = os.path.join(args.log_dir, timestamp)

    logger = setup_logger(args.log_dir)
    logger.info("Starting training on ALL data (residue-level fusion + MIL)")
    logger.info(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading sample IDs from {args.filter_excel_path}")
    pos_df = pd.read_excel(args.filter_excel_path, sheet_name="pos600", header=None)
    neg_df = pd.read_excel(args.filter_excel_path, sheet_name="neg6000", header=None)

    pos_ids = set(pos_df.iloc[:, 0].astype(str).tolist())
    neg_ids = set(neg_df.iloc[:, 0].astype(str).tolist())
    logger.info(f"Loaded {len(pos_ids)} positive IDs and {len(neg_ids)} negative IDs from Excel")

    manifest_path = os.path.join(args.data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    samples_dir = os.path.join(args.data_dir, "samples")

    filtered_manifest = []
    for entry in manifest:
        sample_id = entry["id"]
        label = entry["label"]
        if label == 1 and sample_id in pos_ids:
            filtered_manifest.append(entry)
        elif label == 0 and sample_id in neg_ids:
            filtered_manifest.append(entry)

    original_labels = np.array([entry["label"] for entry in manifest])
    filtered_labels = np.array([entry["label"] for entry in filtered_manifest])

    logger.info(
        f"Original manifest: {len(manifest)} samples "
        f"(pos={int((original_labels == 1).sum())}, neg={int((original_labels == 0).sum())})"
    )
    logger.info(
        f"Filtered manifest: {len(filtered_manifest)} samples "
        f"(pos={int((filtered_labels == 1).sum())}, neg={int((filtered_labels == 0).sum())})"
    )

    if len(filtered_manifest) == 0:
        logger.error("No samples found! Please check whether manifest IDs match the Excel file.")
        return

    manifest = filtered_manifest
    labels_all = filtered_labels
    logger.info(
        f"Total samples for training: {len(manifest)} "
        f"(pos={int((labels_all == 1).sum())}, neg={int((labels_all == 0).sum())})"
    )

    full_dataset = ProteinDataset(samples_dir, manifest)

    generator = torch.Generator().manual_seed(args.random_seed)
    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        generator=generator,
    )
    logger.info(f"Full data shuffle (no sampling): {len(full_dataset)} samples per epoch")

    model = create_model(
        device=device,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        dropout_rate=args.dropout_rate,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        num_clusters=args.num_clusters,
    )

    n_pos = int((labels_all == 1).sum())
    n_neg = int((labels_all == 0).sum())
    if args.loss_type == "logit_adjust":
        pos_prior = float(n_pos) / float(n_pos + n_neg)
        logger.info(
            f"Using LogitAdjustedLoss(pos_prior={pos_prior:.4f}, tau={args.logit_adjust_tau}) "
            f"(pos={n_pos}, neg={n_neg})"
        )
        criterion = LogitAdjustedLoss(pos_prior=pos_prior, tau=args.logit_adjust_tau).to(device)
    else:
        logger.info(
            f"Using FocalLoss(alpha={args.focal_alpha}, gamma={args.focal_gamma}) "
            f"(pos={n_pos}, neg={n_neg})"
        )
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = min(len(train_loader) * args.warmup_epochs, num_training_steps // 10)
    scheduler = get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps)

    model, best_train_metrics, best_epoch = train_model_all_data(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        threshold=args.threshold,
        device=device,
        logger=logger,
        save_dir=args.save_dir,
        load_balancing_weight=args.load_balancing_weight,
        label_smoothing=args.label_smoothing,
        model_name=args.model_name,
    )

    summary = {
        "best_epoch": int(best_epoch + 1),
        "train_size": int(len(manifest)),
        "pos_count": n_pos,
        "neg_count": n_neg,
        "loss_type": args.loss_type,
        "threshold": args.threshold,
    }
    if best_train_metrics is not None:
        summary["best_train_metrics"] = {
            "accuracy": float(best_train_metrics["accuracy"]),
            "precision": float(best_train_metrics["precision"]),
            "recall": float(best_train_metrics["recall"]),
            "f1": float(best_train_metrics["f1"]),
            "mcc": float(best_train_metrics["mcc"]),
            "roc_auc": float(best_train_metrics["roc_auc"]),
            "pr_auc": float(best_train_metrics["pr_auc"]),
            "ap": float(best_train_metrics["ap"]),
            "optimal_metrics": best_train_metrics["optimal_metrics"],
        }

    with open(os.path.join(args.save_dir, "train_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if best_train_metrics is not None:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Training Results (best epoch {best_epoch + 1}, fixed threshold={args.threshold})")
        logger.info(f"{'=' * 60}")
        logger.info(
            f"  Acc={best_train_metrics['accuracy']:.4f}, Prec={best_train_metrics['precision']:.4f}, "
            f"Rec={best_train_metrics['recall']:.4f}, F1={best_train_metrics['f1']:.4f}"
        )
        logger.info(
            f"  MCC={best_train_metrics['mcc']:.4f}, ROC-AUC={best_train_metrics['roc_auc']:.4f}, "
            f"PR-AUC={best_train_metrics['pr_auc']:.4f}, AP={best_train_metrics['ap']:.4f}"
        )
        train_opt = best_train_metrics["optimal_metrics"]
        logger.info(f"\nOptimal threshold metrics (threshold={train_opt['threshold']}):")
        logger.info(
            f"  Acc={train_opt['accuracy']}, Prec={train_opt['precision']}, "
            f"Rec={train_opt['recall']}, F1={train_opt['f1']}, MCC={train_opt['mcc']}"
        )


if __name__ == "__main__":
    main()
