# LightGCN Recommender 

Hashem Jaber:
    - Built the genome based evolution parameter search 
    - Built a pretraining tasks emnbeddings generators
    - Trained the final recomender model that achieved a score of 0.147 NDCG@20

## Setup

### Grid Search 

Note: ~30Gb of RAM is needed 

```py
uv venv
source .venv/bin/activate
uv pip install recommenders tensorflow gradio numpy pandas tqdm torch
```

Adjust variables in `lightgcn_grid.py`:

```
...
testing = False 

# grid search terms:
for emb in [64, 128, 256]:
    for layers in [2, 3, 4]:
...
for decay in [1e-5, 1e-4, 1e-3]:
    for lr in [5e-4, 1e-3, 2e-3]]
```

then run:

```
python lightgcn_grid.py
```
or 
For genomic pretrained based lightGCN
```
python lightgcn_genomic_based_pre_train.py
```

