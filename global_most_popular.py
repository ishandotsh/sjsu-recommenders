from collections import OrderedDict, defaultdict
from math import log2
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from tqdm import tqdm

K = 20


def load_data(path: str) -> Dict[int, List[int]]:
    data: Dict[int, List[int]] = {}
    with open(path, "r") as f:
        for line in f:
            line_arr = [int(x) for x in line.split()]
            if not line_arr:
                continue
            data[line_arr[0]] = line_arr[1:]
    return data


def build_item_counts(data: Dict[int, Sequence[int]]) -> OrderedDict:
    item_counts = defaultdict(int)
    for items in data.values():
        for item in items:
            item_counts[item] += 1
    return OrderedDict(sorted(item_counts.items(), key=lambda x: x[1], reverse=True))


def recommend_popular(
    item_counts_ordered: OrderedDict, seen_items: Set[int], k: int = K
) -> List[int]:
    recs: List[int] = []
    for item in item_counts_ordered.keys():
        if item in seen_items:
            continue
        recs.append(item)
        if len(recs) == k:
            break
    return recs


def dcg_at_k(hit_ranks: Iterable[int]) -> float:
    return sum(1.0 / log2(rank + 1) for rank in hit_ranks)


def evaluate_user(
    recs: Sequence[int], relevant: Set[int], k: int = K
) -> Tuple[float, float, float]:
    if not relevant:
        return 0.0, 0.0, 0.0

    hits = 0
    precisions = 0.0
    hit_ranks: List[int] = []

    for idx, item in enumerate(recs[:k], start=1):
        if item in relevant:
            hits += 1
            precisions += hits / idx
            hit_ranks.append(idx)

    recall = hits / len(relevant)
    map_k = precisions / len(relevant)

    ideal_hits = min(len(relevant), k)
    idcg = dcg_at_k(range(1, ideal_hits + 1)) if ideal_hits > 0 else 0.0
    ndcg = dcg_at_k(hit_ranks) / idcg if idcg > 0 else 0.0

    return recall, map_k, ndcg


def evaluate_on_validation(train_path: str, val_path: str, k: int = K) -> None:
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    item_counts_ordered = build_item_counts(train_data)

    total_recall = 0.0
    total_map = 0.0
    total_ndcg = 0.0
    users_evaluated = 0

    for user, val_items in val_data.items():
        relevant = set(val_items)
        seen = set(train_data.get(user, []))
        recs = recommend_popular(item_counts_ordered, seen, k)
        recall, map_k, ndcg = evaluate_user(recs, relevant, k)
        total_recall += recall
        total_map += map_k
        total_ndcg += ndcg
        users_evaluated += 1

    if users_evaluated == 0:
        print("No users found in validation set for evaluation.")
        return

    avg_recall = total_recall / users_evaluated
    avg_map = total_map / users_evaluated
    avg_ndcg = total_ndcg / users_evaluated

    print(f"Validation users evaluated: {users_evaluated}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"MAP@{k}: {avg_map:.4f}")
    print(f"NDCG@{k}: {avg_ndcg:.4f}")


def write_submission(
    data: Dict[int, List[int]], item_counts_ordered: OrderedDict, k: int = K
) -> str:
    output_lines: List[str] = []
    for user, items in tqdm(data.items()):
        seen = set(items)
        recs = recommend_popular(item_counts_ordered, seen, k)
        line = " ".join([str(user)] + [str(item) for item in recs])
        output_lines.append(line)
    return "\n".join(output_lines) + "\n"


if __name__ == "__main__":
    # Evaluate global popularity trained on train.txt against val.txt.
    evaluate_on_validation("data/train.txt", "data/val.txt", K)

    # Original behavior: train on the full dataset and write submission.
    data = load_data("data/original.txt")
    item_counts_ordered = build_item_counts(data)
    output = write_submission(data, item_counts_ordered, K)

    with open("subs/global_most_popular1.txt", "w") as f:
        f.write(output)
