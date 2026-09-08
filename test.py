import argparse
import json
import logging
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch_geometric
import transformers
import umap
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from model import calculate_metrics, create_model
from utils.foldseek_util import get_struc_seq


def parse_args():
    parser = argparse.ArgumentParser(description="Validate protein classifier model")

    parser.add_argument("--excel_path", type=str, default="../val/thirdparty.xlsx",
                        help="Path to validation Excel file")
    parser.add_argument("--pdb_dir", type=str, default="../val/thirdparty",
                        help="Directory containing PDB files")
    parser.add_argument("--processed_data_dir", type=str, default="",
                        help="Optional processed test data directory containing manifest.json and samples/*.npz")

    parser.add_argument("--model_path", type=str, default="./checkpoints_all",
                        help="Path to train_all.py checkpoint file or directory containing model_all_data.pth")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=1024,
                        help="Maximum residue count for truncation")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--input_size", type=int, default=1280)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.50)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_clusters", type=int, default=4)

    parser.add_argument("--esm_path", type=str, default="../pre/esm2_t33_650M_UR50D")
    parser.add_argument("--saport_path", type=str, default="../pre/saport_650m_af2")

    parser.add_argument("--save_dir", type=str, default="./test_results")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--foldseek_path", type=str, default="../bin/foldseek")

    return parser.parse_args()


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{log_dir}/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def resolve_checkpoint_path(model_path):
    if os.path.isfile(model_path):
        return model_path

    candidates = [
        os.path.join(model_path, "model_all_data.pth"),
        os.path.join(model_path, "best_model.pth"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if os.path.isdir(model_path):
        nested_candidates = []
        for root, _, files in os.walk(model_path):
            if "model_all_data.pth" in files:
                nested_candidates.append(os.path.join(root, "model_all_data.pth"))
            elif "best_model.pth" in files:
                nested_candidates.append(os.path.join(root, "best_model.pth"))
        if nested_candidates:
            nested_candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
            return nested_candidates[0]

    raise FileNotFoundError(f"No checkpoint found under: {model_path}")


def read_pdb_structure(pdb_file):
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)

        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        ca_coords.append(residue["CA"].get_coord())

        return np.array(ca_coords)
    except Exception as exc:
        logging.getLogger(__name__).error(f"Error reading PDB file {pdb_file}: {exc}")
        return None


def create_protein_graph(coords, distance_threshold=8.0):
    if coords is None or len(coords) == 0:
        return None

    n_residues = len(coords)
    distances = squareform(pdist(coords))
    adj_matrix = (distances <= distance_threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)

    for i in range(n_residues - 1):
        adj_matrix[i, i + 1] = 1
        adj_matrix[i + 1, i] = 1

    edge_index = np.where(adj_matrix == 1)
    edge_index = np.stack([edge_index[0], edge_index[1]], axis=0)
    return edge_index, coords


def process_sequence(sequence, tokenizer, model, device, max_len):
    if len(sequence) > max_len - 2:
        sequence = sequence[: max_len - 2]

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

    embeddings = embeddings[0, 1:-1, :]
    actual_len = embeddings.size(0)
    return embeddings.cpu().numpy(), actual_len


def saprot_embed(sequence, tokenizer, model, device, max_len):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

    embeddings = embeddings[0, 1:-1, :]
    actual_len = embeddings.size(0)
    return embeddings.cpu().numpy(), actual_len


def process_sequences_batch(sequences, tokenizer, model, device, max_len, batch_size=16, progress_fn=None):
    truncated = [seq[: max_len - 2] if len(seq) > max_len - 2 else seq for seq in sequences]
    all_embeddings = []
    all_actual_lens = []
    total_batches = (len(truncated) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(truncated), batch_size)):
        batch_seqs = truncated[start:start + batch_size]
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]

        attention_mask = inputs["attention_mask"]
        for idx in range(len(batch_seqs)):
            seq_len = int(attention_mask[idx].sum().item()) - 2
            emb = hidden[idx, 1:1 + seq_len, :]
            all_embeddings.append(emb.cpu().numpy())
            all_actual_lens.append(seq_len)

        if progress_fn:
            progress_fn(batch_idx + 1, total_batches)

    return all_embeddings, all_actual_lens


def saprot_embed_batch(sequences, tokenizer, model, device, max_len, batch_size=16, progress_fn=None):
    all_embeddings = []
    all_actual_lens = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    for batch_idx, start in enumerate(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[start:start + batch_size]
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]

        attention_mask = inputs["attention_mask"]
        for idx in range(len(batch_seqs)):
            seq_len = int(attention_mask[idx].sum().item()) - 2
            emb = hidden[idx, 1:1 + seq_len, :]
            all_embeddings.append(emb.cpu().numpy())
            all_actual_lens.append(seq_len)

        if progress_fn:
            progress_fn(batch_idx + 1, total_batches)

    return all_embeddings, all_actual_lens


class ValidationDataset(Dataset):
    def __init__(self, samples):
        self.samples = []
        for sample in tqdm(samples, desc="Loading validation samples"):
            self.samples.append({
                "struc_emb": torch.FloatTensor(sample["struc_emb"]),
                "graph": Data(
                    x=torch.FloatTensor(sample["graph_x"]),
                    edge_index=torch.LongTensor(sample["edge_index"]),
                    coords=torch.FloatTensor(sample["graph_coords"]),
                ),
                "label": int(sample["label"]),
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


def _label_from_row(row):
    label_value = row.get("Class", row.get("label", row.get("Label", "")))
    if isinstance(label_value, str):
        return 1 if label_value.lower() in {"pos", "positive", "1"} else 0
    return int(label_value)


def _read_processed_validation_data(processed_data_dir, logger):
    manifest_path = os.path.join(processed_data_dir, "manifest.json")
    samples_dir = os.path.join(processed_data_dir, "samples")
    if not os.path.exists(manifest_path) or not os.path.isdir(samples_dir):
        return None, None

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    samples = []
    ids = []
    for entry in tqdm(manifest, desc="Loading processed validation data"):
        data = np.load(os.path.join(samples_dir, entry["file"]))
        label = int(entry["label"]) if "label" in entry else int(data["label"])
        samples.append({
            "struc_emb": data["struc_emb"],
            "graph_x": data["graph_x"],
            "edge_index": data["edge_index"],
            "graph_coords": data["graph_coords"],
            "label": label,
            "aligned_len": int(entry.get("aligned_len", len(data["graph_x"]))),
            "struc_len": int(entry.get("struc_len", len(data["struc_emb"]))),
        })
        ids.append(str(entry.get("id", os.path.splitext(entry["file"])[0])))

    labels = np.array([sample["label"] for sample in samples])
    logger.info(f"Loaded processed validation data from {processed_data_dir}")
    logger.info(
        f"Processed dataset size: {len(samples)} "
        f"(pos={int((labels == 1).sum())}, neg={int((labels == 0).sum())})"
    )
    return samples, ids


def _save_processed_validation_sample(sample, sample_id, protein_id, samples_dir):
    filename = f"{sample_id:05d}.npz"
    np.savez_compressed(
        os.path.join(samples_dir, filename),
        struc_emb=sample["struc_emb"],
        graph_x=sample["graph_x"],
        edge_index=sample["edge_index"],
        graph_coords=sample["graph_coords"],
        label=np.array(sample["label"]),
    )
    return {
        "id": str(protein_id),
        "label": int(sample["label"]),
        "struc_len": int(sample["struc_len"]),
        "aligned_len": int(sample["aligned_len"]),
        "file": filename,
    }


def _as_object_array(items):
    array = np.empty(len(items), dtype=object)
    for idx, item in enumerate(items):
        array[idx] = item
    return array


def prepare_validation_data(excel_path, pdb_dir, device, esm_model, esm_tokenizer,
                            saprot_model, saprot_tokenizer, max_len, foldseek_path, logger,
                            processed_data_dir=""):
    df = pd.read_excel(excel_path)
    logger.info(f"Total validation samples: {len(df)}")

    samples = []
    ids = []
    manifest = []
    samples_dir = ""
    if processed_data_dir:
        samples_dir = os.path.join(processed_data_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing validation samples"):
        protein_id = str(row["Protein ID (Uniprot/NCBI)"])
        pdb_file = os.path.join(pdb_dir, f"AF-{protein_id}-F1-model_v4.pdb")

        if not os.path.exists(pdb_file):
            logger.warning(f"PDB file not found: {protein_id}")
            continue

        try:
            seq_emb, seq_actual_len = process_sequence(
                row["Sequence"], esm_tokenizer, esm_model, device, max_len
            )

            parsed_seqs = get_struc_seq(foldseek_path, pdb_file, ["A"], plddt_mask=True)["A"]
            _, _, combined_seq = parsed_seqs
            struc_emb, struc_actual_len = saprot_embed(
                combined_seq, saprot_tokenizer, saprot_model, device, max_len
            )

            label = _label_from_row(row)
            coords = read_pdb_structure(pdb_file)

            if coords is not None and len(coords) > 0:
                aligned_len = min(seq_actual_len, len(coords))
                coords_truncated = coords[:aligned_len]
                edge_index, graph_coords = create_protein_graph(coords_truncated)
                graph_x = seq_emb[:aligned_len]
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                graph_coords = np.zeros((1, 3), dtype=np.float32)
                graph_x = seq_emb[:1]
                aligned_len = len(graph_x)

            sample = {
                "struc_emb": struc_emb,
                "graph_x": graph_x,
                "edge_index": edge_index,
                "graph_coords": graph_coords,
                "label": label,
                "aligned_len": int(aligned_len),
                "struc_len": int(struc_actual_len),
            }
            samples.append(sample)
            ids.append(protein_id)
            if processed_data_dir:
                manifest.append(
                    _save_processed_validation_sample(
                        sample,
                        len(manifest),
                        protein_id,
                        samples_dir,
                    )
                )

        except Exception as exc:
            logger.error(f"Structure processing failed: {protein_id}, error: {exc}")
            continue

    if not samples:
        logger.error("No valid validation samples found!")
        return None, None

    labels = np.array([sample["label"] for sample in samples])
    logger.info(f"Final dataset size: {len(samples)}")
    logger.info(f"Positive samples: {int((labels == 1).sum())}")
    logger.info(f"Negative samples: {int((labels == 0).sum())}")
    if processed_data_dir:
        manifest_path = os.path.join(processed_data_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed validation data to: {processed_data_dir}")
    return samples, ids


def _extract_sample_interpretability(mil_weights, aux, sample_len):
    mil = mil_weights[:sample_len].astype(np.float32)
    raw_scores = aux["raw_scores"][:sample_len].astype(np.float32)
    gate = aux["gate"][:sample_len, 0].astype(np.float32)
    router = aux["router_weights"][:sample_len].astype(np.float32)
    topk_indices = aux["topk_indices"][:sample_len].astype(np.int64)
    rep_attention = aux["rep_attention"][:, :sample_len].astype(np.float32)
    rep_weights = aux["rep_weights"].astype(np.float32)          # [K]
    rep_indices = aux["rep_indices"].astype(np.int64)            # [K]
    dominant_expert = router.argmax(axis=-1).astype(np.int64)

    # 最终残基权重：把 K 个代表点的 attention 按代表点权重加权求和
    #   rep_saliency[j] = Σ_k rep_weights[k] · rep_attention[k, j]
    rep_saliency = (rep_weights[:, None] * rep_attention).sum(axis=0).astype(np.float32)  # [L]
    pooling_attention = aux.get("pooling_attention", rep_saliency)[:sample_len].astype(np.float32)
    residue_logits = aux["residue_logits"][:sample_len].astype(np.float32)
    residue_contributions = aux.get(
        "residue_contributions", np.zeros(sample_len, dtype=np.float32)
    )[:sample_len].astype(np.float32)

    return {
        "mil_weights": mil,
        "raw_scores": raw_scores,
        "rep_saliency": rep_saliency,
        "pooling_attention": pooling_attention,
        "residue_logits": residue_logits,
        "residue_contributions": residue_contributions,
        "gate": gate,
        "router_weights": router,
        "topk_indices": topk_indices,
        "rep_attention": rep_attention,
        "rep_weights": rep_weights,
        "rep_indices": rep_indices,
        "dominant_expert": dominant_expert,
        "sample_len": int(sample_len),
        "expert_usage": router.mean(axis=0).astype(np.float32),
    }


def run_inference(model, val_loader, device, threshold, logger, collect_aux=False, collect_latent=False):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_mil_weights = []
    all_aux = []
    latent_chunks = []
    captured = []

    hook = None
    if collect_latent:
        def hook_fn(_, inputs, __):
            captured.append(inputs[0].detach())

        hook = model.classifier.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            struc_batch, struc_mask_batch, graph_batch, labels = batch
            struc_batch = struc_batch.to(device)
            struc_mask_batch = struc_mask_batch.to(device)
            graph_batch = graph_batch.to(device)
            labels = labels.to(device)

            if collect_aux:
                logits, mil_weights, _, aux = model(
                    struc_batch,
                    graph_batch,
                    struc_mask=struc_mask_batch,
                    return_aux=True,
                )
            else:
                logits, mil_weights, _ = model(
                    struc_batch,
                    graph_batch,
                    struc_mask=struc_mask_batch,
                )
                aux = None

            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())

            graph_lengths = (graph_batch.ptr[1:] - graph_batch.ptr[:-1]).tolist()
            for idx, graph_len in enumerate(graph_lengths):
                sample_len = min(int(graph_len), int(mil_weights.size(1)))
                sample_mil = mil_weights[idx, :sample_len].cpu().numpy()
                all_mil_weights.append(sample_mil)

                if collect_aux:
                    sample_aux = {
                        "gate": aux["gate"][idx].cpu().numpy(),
                        "router_weights": aux["router_weights"][idx].cpu().numpy(),
                        "topk_indices": aux["topk_indices"][idx].cpu().numpy(),
                        "rep_attention": aux["rep_attention"][idx].cpu().numpy(),
                        "rep_weights": aux["rep_weights"][idx].cpu().numpy(),
                        "rep_indices": aux["rep_indices"][idx].cpu().numpy(),
                        "raw_scores": aux["raw_scores"][idx].cpu().numpy(),
                        "pooling_attention": aux["pooling_attention"][idx].cpu().numpy(),
                        "residue_logits": aux["residue_logits"][idx].cpu().numpy(),
                        "residue_contributions": aux["residue_contributions"][idx].cpu().numpy(),
                    }
                    all_aux.append(_extract_sample_interpretability(sample_mil, sample_aux, sample_len))

            if collect_latent and captured:
                latent_chunks.append(captured[-1].cpu().numpy())
                captured.clear()

    if hook is not None:
        hook.remove()

    result = {
        "predictions": np.array(all_preds),
        "probabilities": np.array(all_probs),
        "labels": np.array(all_labels),
        "mil_weights": all_mil_weights,
    }
    if collect_aux:
        result["interpretability"] = all_aux
    if collect_latent and latent_chunks:
        result["latent_features"] = np.concatenate(latent_chunks, axis=0)
    return result


def evaluate_model(model, val_loader, device, threshold, logger):
    result = run_inference(
        model,
        val_loader,
        device,
        threshold,
        logger,
        collect_aux=False,
        collect_latent=False,
    )
    return result["predictions"], result["probabilities"], result["labels"], result["mil_weights"]


def evaluate_model_with_latent(model, val_loader, device, threshold, logger):
    result = run_inference(
        model,
        val_loader,
        device,
        threshold,
        logger,
        collect_aux=False,
        collect_latent=True,
    )
    return (
        result["predictions"],
        result["probabilities"],
        result["labels"],
        result["latent_features"],
    )


def evaluate_model_with_interpretability(model, val_loader, device, threshold, logger, collect_latent=True):
    return run_inference(
        model,
        val_loader,
        device,
        threshold,
        logger,
        collect_aux=True,
        collect_latent=collect_latent,
    )


def plot_confusion_matrix(cm, save_path):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    save_figure_all_formats(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()


def save_figure_all_formats(png_path, dpi=150):
    base, _ = os.path.splitext(png_path)
    plt.savefig(png_path, dpi=dpi)
    plt.savefig(f"{base}.pdf", bbox_inches="tight")


def plot_roc_curve(fpr, tpr, auc_val, save_path):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_figure_all_formats(os.path.join(save_path, "roc_curve.png"))
    plt.close()


def plot_pr_curve(precision, recall, pr_auc, save_path):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    save_figure_all_formats(os.path.join(save_path, "pr_curve.png"))
    plt.close()


def _project_latent(latent_features, method):
    n_samples = latent_features.shape[0]
    if n_samples < 3:
        raise ValueError(f"{method.upper()} requires at least 3 samples, got {n_samples}")

    if method == "umap":
        reducer = umap.UMAP(n_neighbors=min(15, max(2, n_samples - 1)), min_dist=0.1, random_state=42)
        return reducer.fit_transform(latent_features)

    perplexity = min(30, max(2, n_samples // 3))
    perplexity = min(perplexity, n_samples - 1)
    reducer = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42)
    return reducer.fit_transform(latent_features)


def plot_latent_projection(latent_features, labels, probs, save_path, method):
    os.makedirs(save_path, exist_ok=True)
    embedding = _project_latent(latent_features, method)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    pos_mask = labels == 1
    neg_mask = labels == 0

    axes[0].scatter(
        embedding[neg_mask, 0],
        embedding[neg_mask, 1],
        c="#2166ac",
        label="Negative",
        alpha=0.55,
        s=26,
        edgecolors="none",
    )
    axes[0].scatter(
        embedding[pos_mask, 0],
        embedding[pos_mask, 1],
        c="#b2182b",
        label="Positive",
        alpha=0.8,
        s=30,
        edgecolors="none",
    )
    axes[0].set_title(f"{method.upper()} by True Label")
    axes[0].set_xlabel(f"{method.upper()}-1")
    axes[0].set_ylabel(f"{method.upper()}-2")
    axes[0].legend(markerscale=1.5)

    sc = axes[1].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=probs,
        cmap="RdYlBu_r",
        alpha=0.75,
        s=26,
        edgecolors="none",
    )
    axes[1].set_title(f"{method.upper()} by Predicted Probability")
    axes[1].set_xlabel(f"{method.upper()}-1")
    axes[1].set_ylabel(f"{method.upper()}-2")
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label("P(Protective)")

    plt.tight_layout()
    out_path = os.path.join(save_path, f"{method}_latent.png")
    save_figure_all_formats(out_path)
    plt.close()
    logging.getLogger(__name__).info(f"{method.upper()} plot saved to {out_path}")


def plot_gate_distribution(interpretability, save_path):
    all_gate = [sample["gate"] for sample in interpretability if len(sample["gate"]) > 0]
    if not all_gate:
        return

    gate_values = np.concatenate(all_gate, axis=0)
    plt.figure(figsize=(8, 5))
    plt.hist(gate_values, bins=40, color="#2a9d8f", alpha=0.85, edgecolor="white")
    plt.axvline(0.5, color="#e76f51", linestyle="--", linewidth=1.5, label="Gate = 0.5")
    plt.xlabel("Gate Value")
    plt.ylabel("Residue Count")
    plt.title("Global Gate Distribution")
    plt.legend()
    plt.tight_layout()
    save_figure_all_formats(os.path.join(save_path, "gate_distribution.png"))
    plt.close()


def plot_expert_usage(interpretability, save_path):
    routers = [sample["router_weights"] for sample in interpretability if len(sample["router_weights"]) > 0]
    if not routers:
        return

    router_values = np.concatenate(routers, axis=0)
    expert_usage = router_values.mean(axis=0)
    expert_ids = np.arange(len(expert_usage))

    plt.figure(figsize=(8, 5))
    plt.bar(expert_ids, expert_usage, color="#577590")
    plt.xticks(expert_ids, [f"Expert {idx}" for idx in expert_ids])
    plt.ylabel("Average Router Weight")
    plt.title("Global Expert Usage")
    plt.tight_layout()
    save_figure_all_formats(os.path.join(save_path, "expert_usage.png"))
    plt.close()


def build_interpretability_summary(ids, true_labels, predictions, probabilities, interpretability, num_clusters):
    summaries = []
    for idx, protein_id in enumerate(ids):
        sample = interpretability[idx]
        mil = sample["mil_weights"]
        gate = sample["gate"]
        dominant_expert = sample["dominant_expert"]
        expert_usage = sample["expert_usage"]
        rep_attention = sample["rep_attention"]
        saliency = sample.get("residue_contributions", sample.get("rep_saliency", mil))
        rep_indices = sample.get("rep_indices", None)

        rep_rows = []
        if rep_indices is not None:
            anchor_positions = np.asarray(rep_indices).reshape(-1).astype(np.int64)
        else:
            anchor_positions = np.argsort(mil)[::-1][: min(num_clusters, len(mil))]
        for rep_rank in range(min(rep_attention.shape[0], len(anchor_positions))):
            peak_idx = int(np.argmax(rep_attention[rep_rank]))
            rep_rows.append({
                "rank": rep_rank + 1,
                "anchor_position": int(anchor_positions[rep_rank]) + 1,
                "peak_position": peak_idx + 1,
                "peak_weight": float(rep_attention[rep_rank, peak_idx]),
            })

        positive_positions = np.flatnonzero(np.isfinite(saliency) & (saliency > 0))
        ranked_positive = positive_positions[np.argsort(saliency[positive_positions])[::-1]]
        top_residue_count = min(10, len(ranked_positive))
        top_residues = []
        for pos in ranked_positive[:top_residue_count]:
            top_residues.append({
                "position": int(pos) + 1,
                "saliency": float(saliency[pos]),  # backward-compatible key; value is signed logit contribution
                "positive_logit_contribution": float(saliency[pos]),
                "pooling_attention": float(sample.get("pooling_attention", sample.get("rep_saliency", mil))[pos]),
                "gate": float(gate[pos]),
                "dominant_expert": int(dominant_expert[pos]),
            })

        summaries.append({
            "id": protein_id,
            "true_label": int(true_labels[idx]),
            "prediction": int(predictions[idx]),
            "probability": float(probabilities[idx]),
            "length": int(sample["sample_len"]),
            "mean_gate": float(gate.mean()) if len(gate) else 0.0,
            "sequence_bias_ratio": float((gate >= 0.5).mean()) if len(gate) else 0.0,
            "structure_bias_ratio": float((gate < 0.5).mean()) if len(gate) else 0.0,
            "top_expert": int(np.bincount(dominant_expert).argmax()) if len(dominant_expert) else 0,
            "expert_usage": [float(val) for val in expert_usage.tolist()],
            "representatives": rep_rows,
            "residue_score_type": "positive_class_logit_contribution",
            "top_residues": top_residues,
        })
    return summaries


def main():
    args = parse_args()
    setup_seed(args.random_seed)

    logger = setup_logger(args.log_dir)
    logger.info("Starting validation (residue-level fusion + interpretability)")
    logger.info(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    processed_data_dir = args.processed_data_dir.strip()
    if not processed_data_dir:
        dataset_name = os.path.splitext(os.path.basename(args.excel_path))[0]
        processed_data_dir = os.path.join(args.save_dir, f"{dataset_name}_processed")

    samples, ids = _read_processed_validation_data(processed_data_dir, logger)
    if samples is None:
        logger.info(
            "Processed validation cache not found. "
            "Generating data with the same fields as process_data.py."
        )

        esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
        esm_model = AutoModel.from_pretrained(args.esm_path).to(device)
        esm_model.eval()

        saprot_tokenizer = AutoTokenizer.from_pretrained(args.saport_path)
        saprot_model = AutoModel.from_pretrained(args.saport_path).to(device)
        saprot_model.eval()

        samples, ids = prepare_validation_data(
            args.excel_path,
            args.pdb_dir,
            device,
            esm_model,
            esm_tokenizer,
            saprot_model,
            saprot_tokenizer,
            args.max_len,
            args.foldseek_path,
            logger,
            processed_data_dir=processed_data_dir,
        )
    if samples is None:
        logger.error("No validation data available. Exiting.")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    val_dataset = ValidationDataset(samples)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

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
    checkpoint_path = resolve_checkpoint_path(args.model_path)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    inference = evaluate_model_with_interpretability(
        model,
        val_loader,
        device,
        args.threshold,
        logger,
        collect_latent=True,
    )

    predictions = inference["predictions"]
    probabilities = inference["probabilities"]
    true_labels = inference["labels"]
    interpretability = inference["interpretability"]
    latent_features = inference.get("latent_features")

    np.save(os.path.join(args.save_dir, "residue_contributions.npy"),
            _as_object_array([sample["residue_contributions"] for sample in interpretability]))
    np.save(os.path.join(args.save_dir, "val_ids.npy"), np.array(ids))

    metrics = calculate_metrics(true_labels, predictions.flatten(), probabilities.flatten())
    fpr, tpr, _ = roc_curve(true_labels, probabilities.flatten())
    precisions, recalls, thresholds = precision_recall_curve(true_labels, probabilities.flatten())

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    threshold_idx = min(best_idx, len(thresholds) - 1)
    optimal_threshold = float(thresholds[threshold_idx]) if len(thresholds) > 0 else args.threshold
    optimal_preds = (probabilities.flatten() > optimal_threshold).astype(int)
    optimal_metrics = calculate_metrics(true_labels, optimal_preds, probabilities.flatten())

    summary_rows = build_interpretability_summary(
        ids,
        true_labels,
        predictions.flatten(),
        probabilities.flatten(),
        interpretability,
        args.num_clusters,
    )

    results_df = pd.DataFrame({
        "ID": ids,
        "True_Label": true_labels,
        "Prediction_fixed": predictions.flatten(),
        "Prediction_optimal": optimal_preds,
        "Probability": probabilities.flatten(),
        "Length": [row["length"] for row in summary_rows],
        "Mean_Gate": [row["mean_gate"] for row in summary_rows],
        "Sequence_Bias_Ratio": [row["sequence_bias_ratio"] for row in summary_rows],
        "Top_Expert": [row["top_expert"] for row in summary_rows],
    })
    results_df.to_csv(os.path.join(args.save_dir, "predictions.csv"), index=False)

    with open(os.path.join(args.save_dir, "interpretability_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=False, indent=2)

    plot_roc_curve(fpr, tpr, metrics["roc_auc"], save_path=args.save_dir)
    plot_pr_curve(precisions, recalls, metrics["pr_auc"], save_path=args.save_dir)
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=args.save_dir)
    plot_confusion_matrix(optimal_metrics["confusion_matrix"], save_path=os.path.join(args.save_dir, "optimal"))
    plot_gate_distribution(interpretability, args.save_dir)
    plot_expert_usage(interpretability, args.save_dir)

    if latent_features is not None:
        for method in ("umap", "tsne"):
            try:
                plot_latent_projection(
                    latent_features,
                    true_labels,
                    probabilities.flatten(),
                    save_path=args.save_dir,
                    method=method,
                )
            except Exception as exc:
                logger.warning(f"{method.upper()} visualization skipped: {exc}")

    logger.info(f"\nEvaluation Metrics (fixed threshold={args.threshold}):")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"AUC-ROC: {metrics['roc_auc']:.4f}")
    logger.info(f"AUC-PR: {metrics['pr_auc']:.4f}")
    logger.info(f"MCC: {metrics['mcc']:.4f}")

    logger.info(f"\nEvaluation Metrics (optimal threshold={optimal_threshold:.4f}):")
    logger.info(f"Accuracy: {optimal_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {optimal_metrics['precision']:.4f}")
    logger.info(f"Recall: {optimal_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {optimal_metrics['f1']:.4f}")
    logger.info(f"AUC-ROC: {optimal_metrics['roc_auc']:.4f}")
    logger.info(f"AUC-PR: {optimal_metrics['pr_auc']:.4f}")
    logger.info(f"MCC: {optimal_metrics['mcc']:.4f}")

    logger.info("\nPrediction Statistics:")
    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(f"Positive predictions: {int((predictions == 1).sum())}")
    logger.info(f"Negative predictions: {int((predictions == 0).sum())}")
    logger.info("Validation completed")


if __name__ == "__main__":
    main()
