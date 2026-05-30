# SILC_G

The code for paper "SILC_G: Semantic-aware Integration with Learning and Contrastive Graph Neural Networks".

## 1. Overview

This repository implements drug-target interaction prediction with multi-modal feature fusion and contrastive graph learning. The main pipeline combines:

- **Drug representation**: ChemBERTa-based molecular encoding
- **Protein representation**: ESM-based protein sequence encoding
- **Graph learning**: DGL-based graph neural networks
- **Contrastive learning**: Sample bank mechanism for representation refinement

The repository is organized as follows:

- `data_process/` contains preprocessing scripts for drug and protein inputs;
- `model/` contains the core model implementation;
- `util/` contains dataset loading, seeding, and sequence utilities;
- `predata/` contains the CSV split files used for training, validation, and testing;
- `BioEncoder/` contains the ChemBERTa tokenizer and model assets;
- `train.py` contains the training and testing code.

## 2. Dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
```

Main dependencies used in this project include:

- `torch`
- `dgl`
- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`
- `biopython`
- `graphein`
- `transformers`

## 3. Data Preparation

The code expects each dataset to be arranged in the following style:

```text
data/<dataset_name>/full_data/
predata/<dataset_name>/full_data/
```

`train.py` loads the split CSV files from `predata/<dataset_name>/full_data/` and loads the graph pickle files from `data/<dataset_name>/full_data/`.

Supported datasets in this repository:

- `biosnap`
- `celegans`
- `DAVIS`
- `human`

Make sure the following files exist for each dataset split:

- `train.csv`
- `val.csv`
- `test.csv`

The preprocessing and graph-generation outputs should also be placed in the corresponding `data/<dataset_name>/full_data/` directory.

The original raw data sources are:

- BioSNAP and DAVIS: https://github.com/kexinhuang12345/MolTrans
- Human and C. elegans: https://github.com/lifanchen-simm/transformerCPI

protein 3D structure files should be downloaded according to PID values and placed in `predata\<dataset_name>\pdb`.

## 4. Example

### Training

Run training with:

```bash
python train.py --data biosnap --split full_data --batch_size 64 --epochs 50
```

### Testing Only

If you only want to evaluate the saved model:

```bash
python train.py --data biosnap --split full_data --test
```
