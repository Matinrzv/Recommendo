from src.recommender import RecommenderSystem
from src.utils import print_movies

movies_path = "data/movies.csv"
ratings_path = "data/ratings.csv"

rec = RecommenderSystem(movies_path, ratings_path)

print("==== Popular Movies ====")
popular = rec.get_popular_movies(topn=5)
print_movies(popular)

print("\n==== Content-Based Recommendation ====")
example_movie_id = 1 
content_recs = rec.recommend_content_based(movie_id=example_movie_id, topn=5)
print(f"Recommendations similar to movie_id={example_movie_id}:")
print_movies(content_recs)

print("\n==== Collaborative Filtering Recommendation ====")
user_id_example = 1 
cf_recs = rec.recommend_cf_simple(user_id=user_id_example, topn=5)
print(f"CF recommendations for user_id={user_id_example}:")
print_movies(cf_recs)
