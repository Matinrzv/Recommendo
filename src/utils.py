import pandas as pd
from collections import defaultdict

def print_movies(df, max_rows=10):
    if df.empty:
        print("No movies to show.")
        return
    print(df.head(max_rows).to_string(index=False))

def precision_at_k(predictions, k=10, threshold=4.0):

    user_est = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est[uid].append((iid, est, true_r))

    precisions = []
    for uid, items in user_est.items():
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        topk = items_sorted[:k]
        hits = sum((1 for (_, _, true_r) in topk if true_r >= threshold))
        precisions.append(hits / k)
    return pd.Series(precisions).mean()
def get_unseen_movies(ratings_df, movies_df, user_id):

    seen = ratings_df[ratings_df.userId == user_id]["movieId"].unique().tolist()
    all_movies = movies_df["movieId"].unique().tolist()
    unseen = [m for m in all_movies if m not in seen]
    return unseen
