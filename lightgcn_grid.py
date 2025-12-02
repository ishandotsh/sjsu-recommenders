import pandas as pd
from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_PREDICTION_COL
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from copy import deepcopy
import os

def load_split(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line_arr = line.strip().split()
            u = int(line_arr[0])
            for item_str in line_arr[1:]:
                rows.append((u, int(item_str), 1.0))

    df = pd.DataFrame(rows, columns=[DEFAULT_USER_COL,
                                     DEFAULT_ITEM_COL,
                                     DEFAULT_RATING_COL])
    return df

train_df = load_split("data/train.txt")
val_df   = load_split("data/val.txt")
test_df  = load_split("data/test.txt")

train_df = pd.concat([train_df, val_df], ignore_index=True)

data_val = ImplicitCF(
    train=train_df,
    test=test_df,
    seed=1337,
)

best_ndcg = None
best_cfg = None

results = []

# testing = True
testing = False

if testing:
    # for emb in [64, 128, 256]:
    # for emb in [256]:
        # for layers in [2]:
    # for decay in [1e-5, 1e-4, 1e-3]:
    for decay in [1e-3]:
        for lr in [2e-3]:
        # for lr in [5e-4, 1e-3, 2e-3]:
            # model_dir = f"models/lightgcn_256_2_{decay}_{lr}"
            model_dir = f"models/lightgcn"
            os.mkdir(model_dir)
            print(f"Running {decay} {lr}")
            hparams = prepare_hparams(
                yaml_file=None,
                MODEL_DIR=model_dir,
                model_type="lightgcn",
                embed_size=256,
                n_layers=2,
                decay=0.0001,
                learning_rate=0.002,
                batch_size=2048,
                epochs=100,
                eval_epoch=25,
                top_k=20,
                metrics=["ndcg", "recall"],
                save_model=True,
                # save_model=True,
                save_epoch=100,
            )
            model = LightGCN(hparams, data_val, seed=1337)
            model.fit()
            ndcg_val, recall_val = model.run_eval()
            results.append((decay, lr, ndcg_val, recall_val))
            if best_ndcg is None or ndcg_val > best_ndcg:
                best_ndcg = ndcg_val
                best_cfg = (decay, lr)

    print("Best NDCG:", best_ndcg, "with decay, lr:", best_cfg)

print(results)



# 64, 2
# ndcg = 0.08441, recall = 0.16314
# 64, 3
# ndcg = 0.08181, recall = 0.15731
# 64, 4
# ndcg = 0.07828, recall = 0.15102
# 128, 2
# ndcg = 0.09196, recall = 0.17580
# 128, 3
# ndcg = 0.08783, recall = 0.16896
# 128, 4
# ndcg = 0.08439, recall = 0.16350
# 256, 2
# ndcg = 0.09858, recall = 0.18768   BEST
# 256, 3
# ndcg = 0.09462, recall = 0.18174
# 256, 4
# ndcg = 0.08968, recall = 0.17345


# Running 1e-05 0.0005
# ndcg = 0.08675, recall = 0.16744
# Running 1e-05 0.001
# ndcg = 0.09849, recall = 0.18668 SAME CONFIG AS FIRST BEST 
# Running 1e-05 0.002
# ndcg = 0.10042, recall = 0.18768
# Running 0.0001 0.0005
# ndcg = 0.08682, recall = 0.16767
# Running 0.0001 0.001
# ndcg = 0.09858, recall = 0.18768
# Running 0.0001 0.002
# ndcg = 0.10400, recall = 0.19476 OVERALL BEST ***
# Running 0.001 0.0005
# ndcg = 0.08197, recall = 0.15928
# Running 0.001 0.001
# ndcg = 0.08732, recall = 0.16890


# Submission file 

def create_sub_file(name = "lightgcn_final.txt"):

    train_full = pd.concat([train_df, test_df], ignore_index=True)

    data_test = ImplicitCF(
        train=train_full,
        test=None,
        adj_dir=None,
        seed=1337,
    )
    os.mkdir("models/lightgcn_final")
    hparams_final = prepare_hparams(
        yaml_file=None,
        MODEL_DIR='models/lightgcn_final',
        model_type="lightgcn",
        eval_epoch=1000,
        embed_size=256,
        n_layers=2,
        decay=0.0001,
        learning_rate=0.002,
        batch_size=2048,
        epochs=100,
        top_k=20,
        metrics=["ndcg", "recall"],
        save_model=True,
        save_epoch=100,
    )
    
    model_final = LightGCN(hparams_final, data_test, seed=1337)
    model_final.fit()

    user_ids = train_full[DEFAULT_USER_COL].unique()
    user_ids.sort()

    def join_items(x):
        return " ".join(str(i) for i in x.tolist())

    chunk_size = 5000

    with open(f"subs/{name}", "w") as f:
        for start in range(0, len(user_ids), chunk_size):
            end = start + chunk_size
            batch_ids = user_ids[start:end]

            batch_users = pd.DataFrame({DEFAULT_USER_COL: batch_ids})

            batch_recs = model_final.recommend_k_items(
                test=batch_users,
                top_k=20,
                sort_top_k=True,
                remove_seen=True,
                use_id=False,
            )

            batch_submission = (
                batch_recs.sort_values([DEFAULT_USER_COL, DEFAULT_PREDICTION_COL],
                                    ascending=[True, False])
                        .groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]
                        .apply(join_items)
                        .reset_index()
            )

            for _, row in batch_submission.iterrows():
                f.write(f"{int(row[DEFAULT_USER_COL])} {row[DEFAULT_ITEM_COL]}\n")

create_sub_file()
