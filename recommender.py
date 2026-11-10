"""Movie recommendation system: content-based (TF-IDF), collaborative
filtering (item-item) and a hybrid of both. Uses pandas + scikit-learn."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES_FILE = "movies.csv"
RATINGS_FILE = "ratings.csv"
LIKE_THRESHOLD = 4          # rating >= 4 counts as "liked"
METHODS = ("popular", "content", "collaborative", "hybrid")


def load_data(movies_path=MOVIES_FILE, ratings_path=RATINGS_FILE):
    return pd.read_csv(movies_path), pd.read_csv(ratings_path)


def _minmax(x):
    x = np.asarray(x, dtype=float)
    span = x.max() - x.min()
    return np.zeros_like(x) if span == 0 else (x - x.min()) / span


class MovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies.reset_index(drop=True)
        self.ratings = ratings.copy()
        self.pos = {mid: i for i, mid in enumerate(self.movies.movie_id)}
        self._build_content()
        self._build_collaborative()
        self._build_popularity()

    # ---------- model building ----------
    def _build_content(self):
        # Genres are repeated so they weigh more than plot words.
        genres = self.movies.genres.str.replace("-", "", regex=False).str.replace("|", " ", regex=False)
        text = (genres + " ") * 2 + self.movies["plot"]
        self.tfidf = TfidfVectorizer(stop_words="english").fit_transform(text)
        self.content_sim = cosine_similarity(self.tfidf)

    def _build_collaborative(self):
        ids = self.movies.movie_id
        matrix = self.ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
        self.matrix = matrix.reindex(columns=ids)
        self.user_mean = self.matrix.mean(axis=1)
        centered = self.matrix.sub(self.user_mean, axis=0).fillna(0.0)
        self.item_sim = cosine_similarity(centered.T.values)   # movie x movie

    def _build_popularity(self):
        stats = self.ratings.groupby("movie_id").rating.agg(["mean", "count"])
        m = stats["count"].quantile(0.5)
        c = self.ratings.rating.mean()
        stats["score"] = (stats["count"] * stats["mean"] + m * c) / (stats["count"] + m)
        self.popularity = stats["score"].reindex(self.movies.movie_id).fillna(c).values

    # ---------- helpers ----------
    def find_movie(self, title):
        t = title.strip().lower()
        exact = self.movies[self.movies.title.str.lower() == t]
        part = self.movies[self.movies.title.str.lower().str.contains(t, regex=False)]
        found = exact if len(exact) else part
        if found.empty:
            raise ValueError(f"Movie '{title}' not found.")
        return int(found.movie_id.iloc[0])

    def _user_history(self, user_id):
        return self.ratings[self.ratings.user_id == user_id]

    def _format(self, scores, exclude_ids, n):
        df = self.movies[["movie_id", "title", "genres"]].copy()
        df["score"] = scores
        df = df[~df.movie_id.isin(exclude_ids)]
        return df.sort_values("score", ascending=False).head(n).reset_index(drop=True)

    # ---------- content-based ----------
    def similar_movies(self, title, n=5):
        mid = self.find_movie(title)
        sims = self.content_sim[self.pos[mid]]
        return self._format(sims, {mid}, n)

    def _content_scores(self, user_id):
        hist = self._user_history(user_id)
        liked = hist[hist.rating >= LIKE_THRESHOLD]
        if liked.empty:
            return None
        rows = [self.pos[m] for m in liked.movie_id if m in self.pos]
        profile = np.asarray(self.tfidf[rows].mean(axis=0))
        return cosine_similarity(profile, self.tfidf).ravel()

    # ---------- collaborative filtering ----------
    def _collab_scores(self, user_id):
        if user_id not in self.matrix.index:
            return None
        r = self.matrix.loc[user_id]
        rated = r.notna().values.astype(float)
        centered = (r - self.user_mean[user_id]).fillna(0.0).values
        num = self.item_sim @ centered
        den = np.abs(self.item_sim) @ rated
        pred = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return pred + self.user_mean[user_id]

    # ---------- public recommendation API ----------
    def recommend(self, user_id, n=5, method="hybrid", alpha=0.5):
        if method not in METHODS:
            raise ValueError(f"Method must be one of {METHODS}.")
        seen = set(self._user_history(user_id).movie_id)
        content = self._content_scores(user_id)
        collab = self._collab_scores(user_id)

        if method == "content" and content is not None:
            scores = content
        elif method == "collaborative" and collab is not None:
            scores = collab
        elif method == "hybrid" and content is not None and collab is not None:
            scores = alpha * _minmax(content) + (1 - alpha) * _minmax(collab)
        else:                       # popular, or cold start fallback
            scores = self.popularity
        return self._format(scores, seen, n)

    def user_profile(self, user_id):
        hist = self._user_history(user_id).merge(self.movies, on="movie_id")
        if hist.empty:
            raise ValueError(f"User {user_id} has no watch history.")
        hist = hist.assign(genre=hist.genres.str.split("|")).explode("genre")
        prof = hist.groupby("genre").rating.agg(avg_rating="mean", watched="count")
        return prof.sort_values(["avg_rating", "watched"], ascending=False).round(2).reset_index()


# ---------- evaluation ----------
def evaluate(movies, ratings, k=10, test_frac=0.2, seed=42):
    """Hold out part of each user's ratings, train on the rest, and measure
    Precision@k / Recall@k for hidden movies the user liked (rating >= 4)."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, grp in ratings.groupby("user_id"):
        if len(grp) >= 5:
            test_idx += list(rng.choice(grp.index, size=max(1, int(len(grp) * test_frac)), replace=False))
    test = ratings.loc[test_idx]
    train = ratings.drop(index=test_idx)
    model = MovieRecommender(movies, train)

    results = {m: {"precision": [], "recall": []} for m in METHODS}
    for user, grp in test.groupby("user_id"):
        relevant = set(grp[grp.rating >= LIKE_THRESHOLD].movie_id)
        if not relevant:
            continue
        for m in METHODS:
            recs = set(model.recommend(user, n=k, method=m).movie_id)
            hits = len(recs & relevant)
            results[m]["precision"].append(hits / k)
            results[m]["recall"].append(hits / len(relevant))
    return pd.DataFrame({m: {f"Precision@{k}": np.mean(v["precision"]),
                             f"Recall@{k}": np.mean(v["recall"])}
                         for m, v in results.items()}).T.round(3)


# ---------- CLI ----------
MENU = """
==== Movie Recommender ====
1. Movies similar to a title (content-based)
2. Recommendations for a user
3. Show a user's taste profile
4. Rate a movie (adds watch history)
5. Evaluate accuracy
0. Exit
"""


def main():
    movies, ratings = load_data()
    model = MovieRecommender(movies, ratings)
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()
        try:
            if choice == "1":
                print(model.similar_movies(input("Movie title: "), int(input("How many (e.g. 5): "))).to_string(index=False))
            elif choice == "2":
                uid = int(input("User ID: "))
                method = input(f"Method {METHODS} [hybrid]: ").strip() or "hybrid"
                print(model.recommend(uid, int(input("How many (e.g. 5): ")), method).to_string(index=False))
            elif choice == "3":
                print(model.user_profile(int(input("User ID: "))).to_string(index=False))
            elif choice == "4":
                uid, title = int(input("User ID (new number = new user): ")), input("Movie title: ")
                rating = int(input("Rating 1-5: "))
                if not 1 <= rating <= 5:
                    raise ValueError("Rating must be between 1 and 5.")
                mid = model.find_movie(title)
                ratings = pd.concat([ratings[~((ratings.user_id == uid) & (ratings.movie_id == mid))],
                                     pd.DataFrame([(uid, mid, rating)], columns=ratings.columns)], ignore_index=True)
                ratings.to_csv(RATINGS_FILE, index=False)
                model = MovieRecommender(movies, ratings)
                print("Rating saved.")
            elif choice == "5":
                print(evaluate(movies, ratings))
            elif choice == "0":
                print("Goodbye!")
                break
            else:
                print("Invalid option.")
        except ValueError as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
