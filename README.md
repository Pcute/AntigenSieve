# AntigenSieve

**Residue-Level Evidence Mining for Proteome-Scale Protective Antigen Discovery**

AntigenSieve is an interpretable sequence–structure learning framework for protective antigen prediction. Unlike conventional whole-protein predictors that apply uniform multimodal fusion and global aggregation, AntigenSieve models predictive evidence at residue resolution while requiring only protein-level labels for training.

The framework is built around two complementary forms of residue-level heterogeneity. First, the relative contribution and interaction of semantic and geometric information can vary across residues according to their local structural context. Second, predictive evidence for a protective antigen may be concentrated in a limited number of informative regions rather than distributed uniformly across the entire protein.

To address these properties, AntigenSieve combines **Exogenous–Endogenous Synergistic Residue Feature Fusion** with **Adaptive Anchor-Modulated Residual Saliency Learning**. The first module explicitly allocates semantic and geometric contributions through residue-wise gating while implicitly modeling nonlinear cross-modal interactions through sparse expert routing. The second module uses protein-level weak supervision to identify salient residue anchors and recover complementary contextual evidence around them before producing a protein-level protective antigen score.

The framework is designed for proteome-scale reverse-vaccinology workflows. For each protein, AntigenSieve outputs a protective antigen probability together with residue-level predictive-evidence signals that can support candidate prioritization, mechanistic interpretation, hypothesis generation, and downstream experimental validation.

*Overview of the AntigenSieve residue-level evidence-mining architecture and downstream experimental validation workflow.*

## Highlights

- **Residue-level semantic–geometric encoding:** combines SaProt-derived structure-aware semantic representations with ESM-2-informed geometric graph features at single-residue resolution.
- **Exogenous–Endogenous Synergistic Residue Feature Fusion:** explicitly models residue-wise modality dependence through scalar gating while implicitly capturing residue-conditioned nonlinear semantic–geometric interactions through sparse MoE routing.
- **Adaptive Anchor-Modulated Residual Saliency Learning:** identifies high-saliency residue anchors under protein-level supervision and uses anchor-conditioned attention to recover complementary local or nonlocal predictive evidence.
- **Localized evidence aggregation:** allows multiple informative residue regions to jointly contribute to protein-level prediction rather than compressing all residue information through a uniform global readout.
- **Weak supervision:** learns residue-level predictive-evidence allocation using only protein-level protective/non-protective labels, without residue-level epitope annotations.
- **Built-in interpretability:** exports residue saliency scores, modality-gate values, expert-routing statistics, selected expert identities, and anchor-specific attention profiles.

In the accompanying manuscript, AntigenSieve achieved a PR-AUC of **0.7134** on the PLGDL benchmark and **0.90** on the independent iBPA benchmark. All retrospectively validated protective antigens from *Brucella*, *Plasmodium*, and mpox virus ranked within the top **7.9%** of their respective proteomes. In a prospective *Mycoplasma pneumoniae* screen, 3 of 16 experimentally tested candidates significantly reduced pulmonary bacterial burden in mice.

## Model overview

AntigenSieve progressively transforms complementary residue representations into localized predictive evidence and ultimately into a protein-level protective antigen probability.

1. **Structure-aware semantic encoding.**  
   Each amino acid is paired with its Foldseek structural-alphabet token and encoded using frozen SaProt, producing residue-level semantic representations that jointly reflect sequence identity and local structural state.

2. **Geometric residue encoding.**  
   Frozen ESM-2 residue embeddings are used as graph nodes. Residues are connected according to three-dimensional contacts and backbone adjacency, and geometric information is propagated using GATv2 message passing with distance- and direction-aware edge attributes.

3. **Residue alignment.**  
   Semantic and geometric representations are aligned at residue resolution to preserve one-to-one correspondence before multimodal integration.

4. **Exogenous–Endogenous Synergistic Residue Feature Fusion.**  
   Two complementary pathways operate in parallel:
   - the **exogenous pathway** uses a scalar gate to explicitly allocate the relative contribution of semantic and geometric information for each residue;
   - the **endogenous pathway** uses sparse MoE routing to learn residue-conditioned nonlinear interaction patterns in the joint semantic–geometric feature space.

   The two representations are combined to produce a fused residue-level representation.

5. **Adaptive Anchor-Modulated Residual Saliency Learning.**  
   A learnable MIL query first assigns saliency scores to valid fused residues. The highest-saliency residues are adaptively selected as representative anchors. Each anchor then modulates a second attention operation over the valid residues to recover complementary contextual evidence that may not be represented by the anchor alone.

6. **Protein-level prediction.**  
   Anchor-specific representations are weighted according to their initial saliency and aggregated into a protein-level representation, which is passed to a binary classifier to produce the final protective antigen probability.

Proteins longer than 1,022 residues are truncated to fit the 1,024-token limit of the pretrained encoders, including special tokens.

## Repository structure

```text
.
├── model.py           # AntigenSieve architecture, losses, metrics, and visualization utilities
├── train_single.py    # Five-fold stratified cross-validation training
├── test.py            # Evaluation, prediction, and interpretability export
├── process_data.py    # Sequence/structure preprocessing and feature caching
├── ablation.py        # Ablation experiments
└── util_*.py          # Additional analysis and visualization utilities
```

## Requirements

A CUDA-capable GPU is recommended for feature generation and model training.

The main software dependencies are:

- Python 3
- PyTorch
- PyTorch Geometric
- Transformers
- NumPy
- pandas
- SciPy
- scikit-learn
- Biopython
- Matplotlib
- seaborn
- tqdm
- umap-learn
- openpyxl
- Foldseek
- Local ESM-2 (`esm2_t33_650M_UR50D`) model directory
- Local SaProt (`saport_650m_af2`) model directory

`test.py` uses `utils.foldseek_util.get_struc_seq` to generate Foldseek structural-alphabet tokens. Ensure that this utility is available in the repository and that the Foldseek executable has permission to run.

## Data preparation

### Training data

`train_single.py` expects a preprocessed dataset with the following layout:

```text
processed_data/
├── manifest.json
└── samples/
    ├── 00000.npz
    ├── 00001.npz
    └── ...
```

Each entry in `manifest.json` must contain:

- `id`
- `label`
- `file`

Each `.npz` sample must contain:

- `struc_emb`
- `graph_x`
- `edge_index`
- `graph_coords`
- `label`

The filtering workbook supplied through `--filter_excel_path` must contain two sheets named `pos600` and `neg6000`. Protein identifiers are read from the first column of each sheet.

### Test data

For raw-data inference, the test workbook must contain the following columns:

| Column | Description |
| --- | --- |
| `Protein ID (Uniprot/NCBI)` | Protein identifier used to locate the corresponding structure |
| `Sequence` | Amino-acid sequence |
| `Class` | Ground-truth class, such as `positive`/`negative` or `1`/`0` |

PDB files must be stored in one directory and named as follows:

```text
AF-<Protein ID (Uniprot/NCBI)>-F1-model_v4.pdb
```

On the first run, `test.py` generates and caches processed sequence–structure features under the result directory. Subsequent runs can reuse the cached features through `--processed_data_dir`.

## Usage

Run all commands from the repository root.

### Train with stratified cross-validation

```bash
python train_single.py \
  --filter_excel_path /path/to/plgdl.xlsx \
  --data_dir /path/to/processed_data \
  --save_dir ./checkpoints_single
```

By default, the script performs five-fold stratified cross-validation for up to 50 epochs, uses focal loss, and applies early stopping based on validation PR-AUC.

Results are written to:

```text
checkpoints_single/<RUN_TIMESTAMP>/
```

The best-performing fold checkpoint is also copied to:

```text
best_model.pth
```

within the corresponding run directory.

### Test and export residue-level predictive evidence

```bash
python test.py \
  --excel_path /path/to/test.xlsx \
  --pdb_dir /path/to/pdb_files \
  --model_path ./checkpoints_single/<RUN_TIMESTAMP>/best_model.pth \
  --esm_path /path/to/esm2_t33_650M_UR50D \
  --saport_path /path/to/saport_650m_af2 \
  --foldseek_path /path/to/foldseek \
  --save_dir ./test_results
```

The architecture arguments used for testing must match those used to train the checkpoint, including:

- `--input_size`
- `--hidden_size`
- `--num_heads`
- `--num_experts`
- `--top_k`
- `--num_clusters`

In the current implementation, parameters controlling the number of representative regions or clusters should remain consistent with the number of adaptive residue anchors used during training.

## Outputs

Training produces timestamped logs, per-fold checkpoints, metric summaries, and learning curves.

Testing writes the principal prediction and interpretability artifacts to `--save_dir`.

### Protein-level predictions

- `predictions.csv`  
  Protein-level protective antigen probabilities and binary predictions.

### Residue-level predictive evidence

- `interpretability_summary.json`  
  Residue saliency scores, representative anchors or regions, modality-gate values, expert-routing information, and anchor-specific attention summaries.

- `residue_contributions.npy`  
  Per-residue contributions to the positive-class prediction.

- `val_ids.npy`  
  Protein identifiers aligned with the exported residue-level arrays.

### Visualization outputs

The evaluation pipeline can additionally generate:

- ROC curves
- precision–recall curves
- confusion matrices
- residue-saliency profiles
- gate-distribution plots
- expert-usage plots
- anchor-specific attention visualizations
- UMAP projections
- t-SNE projections

Residue saliency, gate values, expert-routing weights, anchor-specific attention, and residue-contribution scores should be interpreted as **model-derived predictive-evidence allocation signals**. Because AntigenSieve is trained only with protein-level labels, these quantities should not be interpreted as experimentally validated B-cell or T-cell epitopes without independent validation.

## Interpretation of model outputs

AntigenSieve separates residue-level evidence analysis into several complementary signals:

- **Residue saliency (`a_i`)** reflects the initial allocation of protein-level predictive evidence across valid residues.
- **Modality gate (`g_i`)** reflects the explicit relative reliance on semantic versus geometric information at each residue.
- **Expert-routing weights (`π_i,e`)** indicate which nonlinear interaction experts are activated for each residue.
- **Adaptive anchors** correspond to high-saliency residues selected to initiate contextual evidence refinement.
- **Anchor-specific attention (`β_k,i`)** describes how each selected anchor retrieves complementary contextual evidence from other valid residues.

These signals provide interpretable hypotheses about how AntigenSieve reaches a protein-level prediction, but they are not directly supervised residue annotations.

## Ablation analysis

The accompanying experiments evaluate the contributions of the major architectural components through eight simplified variants of the full model.

The principal ablations include:

- **No geometry:** replaces edge-aware geometric encoding with topology-only graph convolution.
- **No exogenous gate:** removes explicit residue-wise scalar modality allocation.
- **No endogenous MoE:** removes residue-conditioned sparse expert interaction modeling.
- **Mean pooling:** replaces adaptive saliency-based evidence aggregation with masked mean pooling.
- **Single anchor:** restricts adaptive anchor-modulated saliency learning to one representative anchor.
- **Equal-weight fusion:** removes both the exogenous gate and endogenous MoE pathways and uses equal-weight semantic–geometric summation.
- **Late fusion:** performs semantic and geometric fusion only after protein-level pooling.
- **Protein-level MoE:** moves expert routing from residue resolution to the protein level.

These experiments separately assess the importance of explicit modality allocation, nonlinear cross-modal interaction modeling, localized evidence aggregation, geometric encoding, and residue-level versus protein-level fusion.

## Citation

If AntigenSieve is useful in your research, please cite the accompanying manuscript:

```bibtex
@article{zhao2026antigensieve,
  title   = {AntigenSieve: Residue-Level Evidence Mining for Prospective In Vivo Discovery of Protective Antigens},
  author  = {Zhao, Yunxiang and Pan, Yunhui and Jiang, Shuyang and others},
  year    = {2026},
  note    = {Manuscript}
}
```

The citation will be updated when the final publication information becomes available.

## Web server

An interactive AntigenSieve web server is available at:

[https://ai4biosciences.com/AntigenSieve](https://ai4biosciences.com/AntigenSieve)

## License

AntigenSieve is released under the MIT License.
