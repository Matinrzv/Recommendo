import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class RecommenderSystem:
    def __init__(self, movies_path: str, ratings_path: str):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.tfidf_matrix = None
        self.nn_model = None
        self.movieId_to_idx = {}

    def get_popular_movies(self, min_ratings: int = 50, topn: int = 10):
        pop = self.ratings.groupby("movieId").agg({"rating": ["mean", "count"]})
        pop.columns = ["rating_mean", "rating_count"]
        pop = pop.reset_index().merge(self.movies, on="movieId", how="left")
        popular_recs = pop[pop["rating_count"] >= min_ratings].sort_values(
            "rating_mean", ascending=False
        )
        return popular_recs[["title", "rating_mean", "rating_count"]].head(topn)

    def build_content_model(self):
        self.movies["genres"] = self.movies["genres"].fillna("")
        self.movies["genres_clean"] = self.movies["genres"].str.replace("|", " ")
        tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tfidf.fit_transform(self.movies["genres_clean"])
        self.nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nn_model.fit(self.tfidf_matrix)
        self.movieId_to_idx = pd.Series(
            self.movies.index, index=self.movies["movieId"]
        ).to_dict()

    def recommend_content_based(self, movie_id: int, topn: int = 10):
        if self.nn_model is None:
            self.build_content_model()
        if movie_id not in self.movieId_to_idx:
            return pd.DataFrame()
        idx = self.movieId_to_idx[movie_id]
        distances, indices = self.nn_model.kneighbors(self.tfidf_matrix[idx], n_neighbors=topn+1)
        rec_indices = indices.flatten()[1:] 
        return self.movies.iloc[rec_indices][["movieId", "title", "genres"]]

    def recommend_cf_simple(self, user_id: int, topn: int = 10):
        movie_mean = self.ratings.groupby("movieId")["rating"].mean()
        seen_movies = self.ratings[self.ratings.userId == user_id]["movieId"].tolist()
        unseen_movies = [m for m in self.movies["movieId"].tolist() if m not in seen_movies]
        preds = [(mid, movie_mean.get(mid, 0)) for mid in unseen_movies]
        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:topn]
        top_df = pd.DataFrame(top, columns=["movieId", "est"])
        return top_df.merge(self.movies, on="movieId")[["movieId", "title", "est", "genres"]]
