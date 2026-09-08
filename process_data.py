import pandas as pd
import numpy as np
import os
import torch
from Bio.PDB import *
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer, AutoModel
import time
import logging
from datetime import datetime
from tqdm import tqdm
from utils.foldseek_util import get_struc_seq
from scipy.spatial.distance import pdist, squareform
import argparse
import json

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Process protein sequence and structure data')

    # 数据路径参数
    parser.add_argument('--excel_path', type=str, default='../PLGDL-master/strc/data/pos_neg.xlsx',
                      help='Path to Excel file containing positive and negative sequences')
    parser.add_argument('--pos_dir', type=str, default='../pos_neg_structure_data/pos',
                      help='Directory containing positive sample PDB files')
    parser.add_argument('--neg_dir', type=str, default='../pos_neg_structure_data/neg',
                      help='Directory containing negative sample PDB files')
    parser.add_argument('--val_excel_path', type=str, default='',
                      help='Path to validation Excel file for removing duplicates')

    # 预训练模型路径
    parser.add_argument('--saport_path', type=str, default='../pre/saport_650m_af2',
                      help='Path to SaProt model')
    parser.add_argument('--esm2_path', type=str, default='../pre/esm2_t33_650M_UR50D',
                      help='Path to ESM2 model')

    # 数据处理参数
    parser.add_argument('--max_len', type=int, default=1024,
                      help='Maximum residue count for truncation (default: 1024)')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')

    # 其他参数
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='Directory to save logs')
    parser.add_argument('--save_dir', type=str, default='./processed_data',
                      help='Directory to save processed data')
    parser.add_argument('--foldseek_path', type=str, default='../bin/foldseek',
                      help='Path to foldseek executable')

    return parser.parse_args()

def setup_logger(log_dir):
    """设置日志记录器"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f'{log_dir}/data_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def read_excel_data(file_path, val_excel_paths, logger):
    """读取Excel文件中的序列和标签数据，并剔除重复数据"""
    logger.info("正在读取数据...")
    pos_df = pd.read_excel(file_path, sheet_name='pos600')
    neg_df = pd.read_excel(file_path, sheet_name='neg6000')

    # 读取前20条数据
    # pos_df = pos_df.head(20)
    # neg_df = neg_df.head(80)

    # 筛选type列为"Bacteria、Viruses、Eukaryota"的数据,其中Bacteria，pos中多一个空格
    # pos_df = pos_df[pos_df['type'] == 'Viruses']
    # neg_df = neg_df[neg_df['type'] == 'Viruses']

    removed_rows = []

    for val_path in val_excel_paths:
        val_df = pd.read_excel(val_path)
        val_names = set(val_df['Protein ID (Uniprot/NCBI)'].astype(str).tolist())
        val_label = val_path.split('/')[-1]

        def is_subseq_in_val(id_str, val_names):
            id_str = str(id_str)
            for name in val_names:
                if id_str in name:
                    return True
            return False

        pos_mask = pos_df['ID'].astype(str).apply(lambda x: is_subseq_in_val(x, val_names))
        neg_mask = neg_df['ID'].astype(str).apply(lambda x: is_subseq_in_val(x, val_names))

        if pos_mask.any():
            for idx, row in pos_df[pos_mask].iterrows():
                id_str = str(row['ID'])
                for name in val_names:
                    if id_str in name:
                        logger.info(f"[pos_df] 被 {val_label} 表的 Name='{name}' 剔除: ID={id_str}")
                        removed_rows.append({'type': 'pos', 'ID': id_str, 'val_file': val_label, 'val_name': name})
                        break
        if neg_mask.any():
            for idx, row in neg_df[neg_mask].iterrows():
                id_str = str(row['ID'])
                for name in val_names:
                    if id_str in name:
                        logger.info(f"[neg_df] 被 {val_label} 表的 Name='{name}' 剔除: ID={id_str}")
                        removed_rows.append({'type': 'neg', 'ID': id_str, 'val_file': val_label, 'val_name': name})
                        break

        pos_df = pos_df[~pos_mask]
        neg_df = neg_df[~neg_mask]
        logger.info(f"{val_label}: 剩余正样本: {len(pos_df)}，剩余负样本: {len(neg_df)}")

    logger.info(f"总共剔除 {len(removed_rows)} 行")

    pos_df['label'] = 1
    neg_df['label'] = 0
    return pos_df, neg_df

def read_pdb_structure(pdb_file):
    """读取PDB文件并提取结构特征"""
    try:
        parser = PDBParser()
        structure = parser.get_structure('protein', pdb_file)

        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_coords.append(residue['CA'].get_coord())

        return np.array(ca_coords)
    except Exception as e:
        logger.error(f"Error reading PDB file {pdb_file}: {str(e)}")
        return None

def apply_rotary_pos_emb(x, freqs_cos, freqs_sin):
    """应用旋转位置编码"""
    assert x.size(-1) % 2 == 0, "x.size(-1) must be even"
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat(
        (x1 * freqs_cos - x2 * freqs_sin,
         x1 * freqs_sin + x2 * freqs_cos),
        dim=-1
    )
    return rotated

def create_protein_graph(coords, distance_threshold=8.0):
    """基于空间距离 + 序列邻接创建蛋白质图结构

    边类型：
    1. 空间边：CA距离 <= distance_threshold 的残基对
    2. 序列边：相邻残基 (i, i+1)，保证骨架连通性
    """
    if coords is None or len(coords) == 0:
        return None

    n_residues = len(coords)

    # 空间距离边
    distances = squareform(pdist(coords))
    adj_matrix = (distances <= distance_threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)

    # 序列边：相邻残基 (i, i+1) 双向
    for i in range(n_residues - 1):
        adj_matrix[i, i + 1] = 1
        adj_matrix[i + 1, i] = 1

    edge_index = np.where(adj_matrix == 1)
    edge_index = np.stack([edge_index[0], edge_index[1]], axis=0)

    return edge_index, coords

def process_sequence(sequence, tokenizer, model, device, max_len):
    """使用ESM2提取嵌入，只截断不padding
    注意：ESM2预训练已包含位置编码，不再叠加额外RoPE
    Returns:
        emb: [actual_len, dim] 变长嵌入
        actual_len: 实际长度
    """
    if len(sequence) > max_len - 2:
        sequence = sequence[:max_len - 2]

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

    embeddings = embeddings[0, 1:-1, :]  # [actual_len, dim]
    actual_len = embeddings.size(0)

    return embeddings.cpu().numpy(), actual_len  # [actual_len, dim], 不padding

def saprot_embed(sequence, tokenizer, model, device, max_len):
    """使用SaProt提取嵌入，只截断不padding
    注意：SaProt预训练已包含位置编码，不再叠加额外RoPE；
    截断交给tokenizer处理，避免在token中间切断
    Returns:
        emb: [actual_len, dim] 变长嵌入
        actual_len: 实际长度
    """
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_len
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

    embeddings = embeddings[0, 1:-1, :]  # [actual_len, dim]
    actual_len = embeddings.size(0)

    return embeddings.cpu().numpy(), actual_len  # [actual_len, dim], 不padding

def process_and_save_sample(row, struct_dir, tokenizer, model, device,
                             saprot_tokenizer, saprot_model, foldseek_path, max_len,
                             sample_id, samples_dir):
    """处理单个样本并立即保存到磁盘，不在内存中累积
    Returns:
        manifest_entry: 样本元信息字典，如果处理失败返回None
    """
    pdb_file = os.path.join(struct_dir, f"AF-{row['ID']}-F1-model_v4.pdb")
    if not os.path.exists(pdb_file):
        logger.warning(f"PDB file not found, skipping: {row['ID']}")
        return None

    try:
        # 序列嵌入（变长，不padding）
        seq_emb, seq_actual_len = process_sequence(row['sequence'], tokenizer, model, device, max_len)

        # 结构感知序列嵌入
        parsed_seqs = get_struc_seq(foldseek_path, pdb_file, ["A"], plddt_mask=True)["A"]
        seq, struct_seq, combined_seq = parsed_seqs
        struc_emb, struc_actual_len = saprot_embed(combined_seq, saprot_tokenizer, saprot_model, device, max_len)

        # 图结构
        coords = read_pdb_structure(pdb_file)
        if coords is None:
            logger.warning(f"Failed to read PDB structure, skipping: {row['ID']}")
            return None
        aligned_len = min(seq_actual_len, len(coords))
        coords_truncated = coords[:aligned_len]
        edge_index, graph_coords = create_protein_graph(coords_truncated)

        # 图节点特征 = seq_emb前aligned_len个残基
        graph_x = seq_emb[:aligned_len]

        # 逐样本保存（不padding，极大节省磁盘和后续内存）
        sample_path = os.path.join(samples_dir, f'{sample_id:05d}.npz')
        np.savez_compressed(sample_path,
            seq_emb=seq_emb,           # [seq_actual_len, dim]
            struc_emb=struc_emb,       # [struc_actual_len, dim]
            graph_x=graph_x,           # [aligned_len, dim]
            edge_index=edge_index,     # [2, n_edges]
            graph_coords=graph_coords, # [aligned_len, 3]
            label=np.array(row['label']))

        return {
            'id': str(row['ID']),
            'label': int(row['label']),
            'seq_len': int(seq_actual_len),
            'struc_len': int(struc_actual_len),
            'aligned_len': int(aligned_len),
            'file': f'{sample_id:05d}.npz'
        }

    except Exception as e:
        logger.error(f"Structure processing failed, skipping: {row['ID']}, error: {str(e)}")
        return None

def main():
    """主函数：提取嵌入并逐样本保存，不做padding/SMOTE/交叉验证"""
    global args
    args = parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    start_time = time.time()
    logger = setup_logger(args.log_dir)
    logger.info('Starting data processing (variable-length, no padding)')
    logger.info(f'Arguments: {args}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # 加载ESM2模型
    esm2_path_abs = os.path.abspath(args.esm2_path)
    tokenizer = AutoTokenizer.from_pretrained(esm2_path_abs)
    model = AutoModel.from_pretrained(esm2_path_abs)
    model = model.to(device)
    model.eval()

    # 加载SaProt模型
    saport_path_abs = os.path.abspath(args.saport_path)
    saprot_tokenizer = AutoTokenizer.from_pretrained(saport_path_abs)
    saprot_model = AutoModel.from_pretrained(saport_path_abs)
    saprot_model = saprot_model.to(device)
    saprot_model.eval()

    # 读取Excel数据
    val_excel_paths = [
        # '../val/thirdparty_homology.xlsx',
        # '../val/thirdparty.xlsx',
        # '../new_datasets/布鲁氏菌已知靶标抗原.xlsx',
        # '../new_datasets/猴痘已知靶标抗原.xlsx',
        # '../new_datasets/疟原虫已知靶标抗原.xlsx'
    ]
    pos_df, neg_df = read_excel_data(args.excel_path, val_excel_paths, logger)

    # 创建样本保存目录
    samples_dir = os.path.join(args.save_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    manifest = []
    sample_id = 0

    seq_lens = []
    struc_lens = []

    # 处理正样本
    logger.info("Processing positive samples...")
    for idx, row in tqdm(pos_df.iterrows(), total=len(pos_df), desc="Processing positive samples"):
        entry = process_and_save_sample(
            row, args.pos_dir, tokenizer, model, device,
            saprot_tokenizer, saprot_model, args.foldseek_path, args.max_len,
            sample_id, samples_dir
        )
        if entry is not None:
            manifest.append(entry)
            seq_lens.append(entry['seq_len'])
            struc_lens.append(entry['struc_len'])
            sample_id += 1

    # 处理负样本
    logger.info("Processing negative samples...")
    for idx, row in tqdm(neg_df.iterrows(), total=len(neg_df), desc="Processing negative samples"):
        entry = process_and_save_sample(
            row, args.neg_dir, tokenizer, model, device,
            saprot_tokenizer, saprot_model, args.foldseek_path, args.max_len,
            sample_id, samples_dir
        )
        if entry is not None:
            manifest.append(entry)
            seq_lens.append(entry['seq_len'])
            struc_lens.append(entry['struc_len'])
            sample_id += 1

    # 保存manifest（样本元信息索引，训练时按需读取）
    manifest_path = os.path.join(args.save_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # 统计信息
    labels = np.array([m['label'] for m in manifest])
    seq_lens = np.array(seq_lens)
    struc_lens = np.array(struc_lens)

    logger.info(f"Total samples: {len(manifest)}")
    logger.info(f"Positive samples: {sum(labels == 1)}")
    logger.info(f"Negative samples: {sum(labels == 0)}")
    logger.info(f"Seq length stats: min={seq_lens.min()}, max={seq_lens.max()}, mean={seq_lens.mean():.1f}, median={np.median(seq_lens):.1f}")
    logger.info(f"Struc length stats: min={struc_lens.min()}, max={struc_lens.max()}, mean={struc_lens.mean():.1f}, median={np.median(struc_lens):.1f}")
    logger.info(f"Saved to: {samples_dir}")

    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    logger = setup_logger('logs')
    main()
