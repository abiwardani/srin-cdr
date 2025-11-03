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

Best CMF score is f1: 0.9654, precision: 0.9365 and target-dataset BCE: 0.5062
Best NCF score is f1: 0.9582, precision: 0.9274, and target-dataset MSE: 0.8832

## Acknowledgement

The datasets are obtained from [Amazon-Books](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon_ratings/Amazon_Books.zip) and [Amazon-Movies-TV](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon_ratings/Amazon_Movies_and_TV.zip).

The data structuring of this project is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

The implementation of CMF is based on [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR) which is based on the work of Singh and Gordon: [Relational Learning via Collective Matrix Factorization](https://dl.acm.org/doi/10.1145/1401890.1401969) (SIGKDD 2008).

The implementation of NCF is based on [DDRCDR](https://github.com/lpworld/DDTCDR), which is an implementation of the work of Li and Tuzhilin [DDTCDR: Deep Dual Transfer Cross Domain Recommendation](https://dl.acm.org/doi/abs/10.1145/3336191.3371793), a transfer learning modification of the original NCF framework proposed in [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569).