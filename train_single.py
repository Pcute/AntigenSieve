"""
train_single.py: 残基级双编码器融合 + MIL 表位定位
五折交叉验证训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Data, Batch
import numpy as np
from model import create_model, calculate_metrics, plot_metrics, FocalLoss, LogitAdjustedLoss
import logging
from datetime import datetime
import os
from tqdm import tqdm
import argparse
import random
import transformers
import torch_geometric
import json
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import pandas as pd


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train protein classifier with 5-fold CV')

    # 数据参数
    parser.add_argument('--filter_excel_path', type=str, default='../数据处理/plgdl.xlsx',
                      help='Path to filter Excel file containing pos600 and neg6000 sheets for sample filtering')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--n_folds', type=int, default=5,
                      help='Number of folds for cross-validation')
    parser.add_argument('--max_len', type=int, default=1024,
                      help='Maximum residue count for truncation (must match process_data.py)')

    # 模型参数
    parser.add_argument('--input_size', type=int, default=1280,
                      help='Input feature dimension')
    parser.add_argument('--hidden_size', type=int, default=256,
                      help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.50,
                      help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--num_experts', type=int, default=4,
                      help='Number of MoE experts for residue-level fusion routing')
    parser.add_argument('--top_k', type=int, default=2,
                      help='Top-K experts to select per residue')
    parser.add_argument('--num_clusters', type=int, default=4,
                      help='Number of clusters for clustered MIL pooling (linear/conformational/hybrid epitopes)')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10,
                      help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.025,
                      help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='Number of warmup epochs')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for binary classification')

    # 损失函数选择: 'focal' 或 'logit_adjust'
    parser.add_argument('--loss_type', type=str, default='focal',
                      choices=['focal', 'logit_adjust'],
                      help="Loss function: 'focal'=Focal Loss, "
                           "'logit_adjust'=Logit Adjustment (Menon et al. 2021, 无需采样)")
    parser.add_argument('--focal_alpha', type=float, default=0.45,
                      help='Focal Loss: alpha parameter (positive class weight)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                      help='Focal Loss: gamma parameter (focusing parameter)')
    parser.add_argument('--logit_adjust_tau', type=float, default=1.0,
                      help='Logit adjustment strength (tau). 1.0=standard prior correction')

    # 负载均衡
    parser.add_argument('--load_balancing_weight', type=float, default=0.0005,
                      help='Weight for MoE expert load balancing loss')

    # 标签平滑
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    # 其他参数
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory for logs')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                      help='Directory containing processed data (samples/ and manifest.json)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_single',
                      help='Directory to save models')
    parser.add_argument('--random_seed', type=int, default=42)

    return parser.parse_args()


def setup_logger(log_dir):
    """设置日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f'{log_dir}/train_single_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def setup_seed(seed):
    """设置所有随机种子"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
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


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """创建学习率调度器"""
    from transformers import get_cosine_schedule_with_warmup
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32 + worker_id)
    random.seed(torch.initial_seed() % 2**32 + worker_id)


class ProteinDataset(Dataset):
    """数据集类：初始化时一次性加载所有样本到内存（预转为tensor），消除训练时IO开销
    只加载 struc_emb 和 graph，不再需要 seq_emb（已融入 graph_x）
    """
    def __init__(self, samples_dir, manifest):
        self.samples = []
        print(f"Pre-loading {len(manifest)} samples into memory...")
        for info in tqdm(manifest, desc="Loading samples"):
            data = np.load(os.path.join(samples_dir, info['file']))
            self.samples.append({
                'struc_emb': torch.FloatTensor(data['struc_emb']),
                'graph': Data(
                    x=torch.FloatTensor(data['graph_x']),
                    edge_index=torch.LongTensor(data['edge_index']),
                    coords=torch.FloatTensor(data['graph_coords'])
                ),
                'label': int(data['label'])
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['struc_emb'], s['graph'], s['label']


def collate_fn(batch):
    """动态padding：按batch内最大长度padding"""
    from torch.nn.utils.rnn import pad_sequence

    strucs, graphs, labels = zip(*batch)

    padded_strucs = pad_sequence(strucs, batch_first=True)

    struc_masks = torch.zeros_like(padded_strucs[:, :, 0])
    for i, struc in enumerate(strucs):
        struc_masks[i, :struc.size(0)] = 1.0

    labels = torch.tensor(labels, dtype=torch.float32)
    graphs = Batch.from_data_list(graphs)

    return padded_strucs, struc_masks, graphs, labels


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, threshold, device, logger, save_dir,
                load_balancing_weight, label_smoothing, patience=10):
    """训练模型"""
    best_val = 0.0
    best_epoch = 0
    best_val_metrics = None  # 保存最佳epoch的验证指标
    counter = 0

    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'train_mcc': [], 'val_mcc': [],
        'train_roc_auc': [], 'val_roc_auc': [],
        'train_pr_auc': [], 'val_pr_auc': [],
        'train_ap': [], 'val_ap': [],
        'train_optimal_metrics': [], 'val_optimal_metrics': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cls_loss_sum = 0.0
        train_lb_loss_sum = 0.0
        train_probs = []
        train_labels = []

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
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

        train_loss = train_loss / len(train_loader)
        train_cls_loss_avg = train_cls_loss_sum / len(train_loader)
        train_lb_loss_avg = train_lb_loss_sum / len(train_loader)
        all_train_probs = torch.cat(train_probs).cpu().numpy().flatten()
        all_train_labels = torch.cat(train_labels).cpu().numpy()
        all_train_preds = (all_train_probs > threshold).astype(int)
        train_metrics = calculate_metrics(all_train_labels, all_train_preds, all_train_probs)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_cls_loss_sum = 0.0
        val_lb_loss_sum = 0.0
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                struc_batch, struc_mask_batch, graph_batch, labels = batch
                struc_batch = struc_batch.to(device, non_blocking=True)
                struc_mask_batch = struc_mask_batch.to(device, non_blocking=True)
                graph_batch = graph_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits, _, val_lb_loss = model(struc_batch, graph_batch, struc_mask=struc_mask_batch)
                cls_loss = criterion(logits, labels.unsqueeze(1))
                val_cls_loss_sum += cls_loss.item()
                val_lb_loss_sum += val_lb_loss.item()

                val_loss += (cls_loss + load_balancing_weight * val_lb_loss).item()

                val_probs.append(torch.sigmoid(logits).detach())
                val_labels.append(labels.detach())

        val_loss = val_loss / len(val_loader)
        val_cls_loss_avg = val_cls_loss_sum / len(val_loader)
        val_lb_loss_avg = val_lb_loss_sum / len(val_loader)
        all_val_probs = torch.cat(val_probs).cpu().numpy().flatten()
        all_val_labels = torch.cat(val_labels).cpu().numpy()
        all_val_preds = (all_val_probs > threshold).astype(int)
        val_metrics = calculate_metrics(all_val_labels, all_val_preds, all_val_probs)

        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc', 'pr_auc', 'ap', 'optimal_metrics']:
            metrics_history[f'train_{metric}'].append(train_metrics[metric])
            metrics_history[f'val_{metric}'].append(val_metrics[metric])

        logger.info(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.info(f'Train Loss: {train_loss:.4f} (cls={train_cls_loss_avg:.4f}, lb={train_lb_loss_avg:.4f}) | '
                    f'Acc={train_metrics["accuracy"]:.4f}, Prec={train_metrics["precision"]:.4f}, '
                    f'Rec={train_metrics["recall"]:.4f}, F1={train_metrics["f1"]:.4f}, '
                    f'MCC={train_metrics["mcc"]:.4f}, ROC-AUC={train_metrics["roc_auc"]:.4f}, '
                    f'PR-AUC={train_metrics["pr_auc"]:.4f}')
        logger.info(f'Val   Loss: {val_loss:.4f} (cls={val_cls_loss_avg:.4f}, lb={val_lb_loss_avg:.4f}) | '
                    f'Acc={val_metrics["accuracy"]:.4f}, Prec={val_metrics["precision"]:.4f}, '
                    f'Rec={val_metrics["recall"]:.4f}, F1={val_metrics["f1"]:.4f}, '
                    f'MCC={val_metrics["mcc"]:.4f}, ROC-AUC={val_metrics["roc_auc"]:.4f}, '
                    f'PR-AUC={val_metrics["pr_auc"]:.4f}')

        # 最优阈值指标
        train_opt = train_metrics['optimal_metrics']
        val_opt = val_metrics['optimal_metrics']
        logger.info(f'Train Optimal: threshold={train_opt["threshold"]}, '
                    f'Acc={train_opt["accuracy"]}, Prec={train_opt["precision"]}, '
                    f'Rec={train_opt["recall"]}, F1={train_opt["f1"]}, MCC={train_opt["mcc"]}')
        logger.info(f'Val   Optimal: threshold={val_opt["threshold"]}, '
                    f'Acc={val_opt["accuracy"]}, Prec={val_opt["precision"]}, '
                    f'Rec={val_opt["recall"]}, F1={val_opt["f1"]}, MCC={val_opt["mcc"]}')

        # 使用PR-AUC作为保存最佳模型的指标
        if val_metrics['pr_auc'] > best_val:
            best_val = val_metrics['pr_auc']
            best_epoch = epoch
            best_val_metrics = val_metrics
            counter = 0

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            logger.info(f'New best model saved with validation PR-AUC: {best_val:.4f}')
        else:
            counter += 1
            logger.info(f'EarlyStopping counter: {counter} out of {patience}')
            if counter >= patience:
                logger.info(f'Early stopping triggered. Best epoch: {best_epoch + 1}')
                logger.info(f'Best validation PR-AUC: {best_val:.4f}')
                break

    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    logger.info(f'Loaded best model from epoch {best_epoch + 1}')

    metrics_history['final_confusion_matrix'] = best_val_metrics['confusion_matrix']
    plot_metrics(metrics_history, save_path=os.path.join(save_dir, 'metrics_plots'))

    return model, best_val_metrics


def main():
    """主函数：五折交叉验证训练"""
    args = parse_args()
    setup_seed(args.random_seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir, timestamp)
    args.log_dir = os.path.join(args.log_dir, timestamp)

    logger = setup_logger(args.log_dir)
    logger.info('Starting 5-fold cross-validation training (residue-level fusion + MIL)')
    logger.info(f'Arguments: {args}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 加载 plgdl_filtered_for_ibpa.xlsx 中的正负样本ID
    logger.info(f"Loading sample IDs from {args.filter_excel_path}")
    pos_df = pd.read_excel(args.filter_excel_path, sheet_name='pos600', header=None)
    neg_df = pd.read_excel(args.filter_excel_path, sheet_name='neg6000', header=None)

    pos_ids = set(pos_df.iloc[:, 0].astype(str).tolist())
    neg_ids = set(neg_df.iloc[:, 0].astype(str).tolist())
    logger.info(f"Loaded {len(pos_ids)} positive IDs and {len(neg_ids)} negative IDs from Excel")

    # 加载manifest
    manifest_path = os.path.join(args.data_dir, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    samples_dir = os.path.join(args.data_dir, 'samples')

    # 过滤：只保留在 plgdl_filtered_for_ibpa.xlsx 中的样本
    filtered_manifest = []
    for entry in manifest:
        sample_id = entry['id']
        label = entry['label']
        if label == 1 and sample_id in pos_ids:
            filtered_manifest.append(entry)
        elif label == 0 and sample_id in neg_ids:
            filtered_manifest.append(entry)

    original_labels = np.array([m['label'] for m in manifest])
    filtered_labels = np.array([m['label'] for m in filtered_manifest])

    logger.info(f"Original manifest: {len(manifest)} samples "
                f"(pos={sum(original_labels==1)}, neg={sum(original_labels==0)})")
    logger.info(f"Filtered manifest: {len(filtered_manifest)} samples "
                f"(pos={sum(filtered_labels==1)}, neg={sum(filtered_labels==0)})")

    if len(filtered_manifest) == 0:
        logger.error("No samples found! Please check if the IDs in manifest match those in Excel file.")
        logger.error(f"Sample manifest IDs (first 10): {[m['id'] for m in manifest[:10]]}")
        logger.error(f"Sample Excel pos IDs (first 10): {list(pos_ids)[:10]}")
        logger.error(f"Sample Excel neg IDs (first 10): {list(neg_ids)[:10]}")
        return

    manifest = filtered_manifest
    labels_all = filtered_labels

    logger.info(f"Total samples for training: {len(manifest)} "
                f"(pos={sum(labels_all==1)}, neg={sum(labels_all==0)})")

    # 记录每折验证指标
    fold_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'mcc': [], 'roc_auc': [], 'pr_auc': [], 'ap': []
    }
    best_overall_pr_auc = 0.0
    best_fold = -1

    # 数据划分策略
    if args.n_folds > 0:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_seed)
        splits = list(skf.split(np.zeros(len(labels_all)), labels_all))
        logger.info(f"Starting {args.n_folds}-fold cross-validation training")
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_seed)
        splits = list(sss.split(np.zeros(len(labels_all)), labels_all))
        logger.info("Starting single train/val split (80:20) training")

    n_splits = len(splits)

    logger.info("Pre-loading full filtered dataset once for all folds")
    full_dataset = ProteinDataset(samples_dir, manifest)

    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'='*60}")
        if n_splits > 1:
            logger.info(f"Fold {fold+1}/{n_splits}")
        else:
            logger.info(f"Single Split (train 80% / val 20%)")
        logger.info(f"{'='*60}")

        train_labels = labels_all[train_idx]
        val_labels = labels_all[val_idx]

        logger.info(f"Train: {len(train_idx)} (pos={sum(train_labels==1)}, neg={sum(train_labels==0)})")
        logger.info(f"Val:   {len(val_idx)} (pos={sum(val_labels==1)}, neg={sum(val_labels==0)})")

        # 复用预加载的全量数据集，避免每折重复读取 .npz。
        train_dataset = Subset(full_dataset, train_idx.tolist())
        val_dataset = Subset(full_dataset, val_idx.tolist())

        # 数据加载（全量数据，无采样）
        generator = torch.Generator().manual_seed(args.random_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            generator=generator
        )
        logger.info(f"Full data shuffle (no sampling): {len(train_dataset)} samples per epoch")

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # 每折初始化新模型
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

        # 损失函数（处理不平衡）
        n_pos = (train_labels == 1).sum()
        n_neg = (train_labels == 0).sum()
        if args.loss_type == 'logit_adjust':
            pos_prior = float(n_pos) / float(n_pos + n_neg)
            logger.info(f"Using LogitAdjustedLoss(pos_prior={pos_prior:.4f}, tau={args.logit_adjust_tau}) "
                        f"(pos={int(n_pos)}, neg={int(n_neg)})")
            criterion = LogitAdjustedLoss(pos_prior=pos_prior, tau=args.logit_adjust_tau).to(device)
        else:
            logger.info(f"Using FocalLoss(alpha={args.focal_alpha}, gamma={args.focal_gamma}) "
                        f"(pos={int(n_pos)}, neg={int(n_neg)})")
            criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        num_training_steps = len(train_loader) * args.num_epochs
        num_warmup_steps = min(len(train_loader) * args.warmup_epochs, num_training_steps // 10)
        scheduler = get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps)

        fold_save_dir = os.path.join(args.save_dir, f'fold_{fold}')
        if not os.path.exists(fold_save_dir):
            os.makedirs(fold_save_dir)

        model, best_val_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
            threshold=args.threshold,
            device=device,
            logger=logger,
            save_dir=fold_save_dir,
            load_balancing_weight=args.load_balancing_weight,
            label_smoothing=args.label_smoothing,
            patience=args.patience
        )

        # 直接使用 train_model 返回的最佳 epoch 指标
        final_metrics = best_val_metrics
        val_opt = final_metrics['optimal_metrics']
        optimal_threshold = float(val_opt['threshold'])
        optimal_fold_metrics = {
            'accuracy': float(val_opt['accuracy']),
            'precision': float(val_opt['precision']),
            'recall': float(val_opt['recall']),
            'f1': float(val_opt['f1']),
            'mcc': float(val_opt['mcc'])
        }

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc', 'pr_auc', 'ap']:
            fold_metrics[metric].append(final_metrics[metric])

        # 最优阈值指标单独记录
        if 'optimal_accuracy' not in fold_metrics:
            fold_metrics['optimal_threshold'] = []
            fold_metrics['optimal_accuracy'] = []
            fold_metrics['optimal_precision'] = []
            fold_metrics['optimal_recall'] = []
            fold_metrics['optimal_f1'] = []
            fold_metrics['optimal_mcc'] = []
            fold_metrics['optimal_roc_auc'] = []
            fold_metrics['optimal_pr_auc'] = []
        fold_metrics['optimal_threshold'].append(optimal_threshold)
        fold_metrics['optimal_accuracy'].append(optimal_fold_metrics['accuracy'])
        fold_metrics['optimal_precision'].append(optimal_fold_metrics['precision'])
        fold_metrics['optimal_recall'].append(optimal_fold_metrics['recall'])
        fold_metrics['optimal_f1'].append(optimal_fold_metrics['f1'])
        fold_metrics['optimal_mcc'].append(optimal_fold_metrics['mcc'])
        fold_metrics['optimal_roc_auc'].append(float(val_opt['roc_auc']))
        fold_metrics['optimal_pr_auc'].append(float(val_opt['pr_auc']))

        logger.info(f"\nFold {fold+1} Results (fixed threshold={args.threshold}):")
        logger.info(f"  Acc={final_metrics['accuracy']:.4f}, Prec={final_metrics['precision']:.4f}, "
                     f"Rec={final_metrics['recall']:.4f}, F1={final_metrics['f1']:.4f}")
        logger.info(f"  MCC={final_metrics['mcc']:.4f}, ROC-AUC={final_metrics['roc_auc']:.4f}, "
                     f"PR-AUC={final_metrics['pr_auc']:.4f}, AP={final_metrics['ap']:.4f}")
        logger.info(f"Fold {fold+1} Results (optimal threshold={optimal_threshold:.4f}):")
        logger.info(f"  Acc={optimal_fold_metrics['accuracy']:.4f}, Prec={optimal_fold_metrics['precision']:.4f}, "
                     f"Rec={optimal_fold_metrics['recall']:.4f}, F1={optimal_fold_metrics['f1']:.4f}, "
                     f"MCC={optimal_fold_metrics['mcc']:.4f}")
        logger.info(f"  ROC-AUC={float(val_opt['roc_auc']):.4f}, PR-AUC={float(val_opt['pr_auc']):.4f}")

        if final_metrics['pr_auc'] > best_overall_pr_auc:
            best_overall_pr_auc = final_metrics['pr_auc']
            best_fold = fold

        # 释放当前fold的模型和数据
        del model, train_dataset, val_dataset, train_loader, val_loader
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # 五折汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"{args.n_folds}-Fold Cross-Validation Results (fixed threshold={args.threshold})")
    logger.info(f"{'='*60}")

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'roc_auc', 'pr_auc', 'ap']:
        values = fold_metrics[metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"  {metric:>10s}: {mean_val:.4f} +/- {std_val:.4f}  "
                     f"(per fold: {[f'{v:.4f}' for v in values]})")

    logger.info(f"\n{'='*60}")
    logger.info(f"{args.n_folds}-Fold Results (optimal threshold per fold)")
    logger.info(f"{'='*60}")

    for metric in ['optimal_threshold', 'optimal_accuracy', 'optimal_precision',
                   'optimal_recall', 'optimal_f1', 'optimal_mcc',
                   'optimal_roc_auc', 'optimal_pr_auc']:
        values = fold_metrics[metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"  {metric:>20s}: {mean_val:.4f} +/- {std_val:.4f}  "
                     f"(per fold: {[f'{v:.4f}' for v in values]})")

    # 复制最佳折的模型
    import shutil
    best_model_src = os.path.join(args.save_dir, f'fold_{best_fold}', 'best_model.pth')
    best_model_dst = os.path.join(args.save_dir, 'best_model.pth')
    shutil.copy2(best_model_src, best_model_dst)
    logger.info(f"\nBest model from fold {best_fold+1} "
                f"(PR-AUC={best_overall_pr_auc:.4f}) copied to {best_model_dst}")

    logger.info('Training completed')


if __name__ == "__main__":
    main()
