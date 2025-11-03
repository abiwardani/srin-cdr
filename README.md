# srin-cdr

A cross-domain recommendation dataset using the Amazon Books and Amazon Movies & TV datasets. CDR is performed using Collective Matrix Factorization (CMF) as a benchmark and Neural Collaborative Filtering (NCF) as a deep learning model. CMF uses naive ID embeddings while NCF as a context-aware model uses precomputed item feature vectors from BERT's base model.

## Requirements

```
scipy>=1.16.2
torch>=2.0.1
python>=3.11.0
```

## Quick-Start

To run CDR with the provided dataset, execute the following.

```bash
python run_cdr.py --model=CMF
```

This script will run the CMF model with amazon-custom-books as source domain dataset and amazon-custom-movies-tv as target domain dataset.

## Usage

For full usage:

```bash
python run_cdr.py --model=[model] --epochs=[epochs] --lr=[lr] --data-folder-path=[path/to/dataset] --source-name=[name of source dataset] --target-name=[name of target dataset] --load-embeds[bool]
```

## Data Engineering

The notebook [`data_engineering.ipynb`](dataset/data_engineering.ipynb) documents the process of data engineering from the original Amazon Books and Amazon Movies & TV datasets into the filtered dataset used in this project. The final datasets contain approximately ~4000 users, ~5000 items in the source dataset, and ~5000 items in the target dataset. The item feature embeddings based on the item title/category/descriptions provided are reduced from 768 dimensions to 96 using PCA.

## Results

Best CMF score is f1: 0.9782, source RMSE: 0.2978 and target RMSE: 0.2807
Best NCF score is f1: 0.9776, source-oriented RMSE: 0.5092, and target-oriented RMSE: 0.5659

## Acknowledgement

The data structuring of this project is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

CMF is based on 