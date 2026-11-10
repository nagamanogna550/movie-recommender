from recommender import MovieRecommender, evaluate, load_data


def expect_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_all():
    movies, ratings = load_data()
    rec = MovieRecommender(movies, ratings)

    # content-based: results are sorted, exclude the query movie, respect n
    sim = rec.similar_movies("The Matrix", 5)
    assert len(sim) == 5 and "The Matrix" not in sim.title.values
    assert sim.score.is_monotonic_decreasing
    assert sim.genres.str.contains("Sci-Fi").sum() >= 3

    # partial title match works, unknown title raises
    assert rec.find_movie("matrix") == rec.find_movie("The Matrix")
    expect_error(rec.find_movie, "no such film")

    # recommendations never include already-watched movies
    for method in ("popular", "content", "collaborative", "hybrid"):
        recs = rec.recommend(1, 5, method)
        seen = set(ratings[ratings.user_id == 1].movie_id)
        assert len(recs) == 5 and not (set(recs.movie_id) & seen)
    expect_error(rec.recommend, 1, 5, "magic")

    # cold start: unknown user falls back to popular movies
    assert len(rec.recommend(99999, 3)) == 3

    # taste profile
    assert {"genre", "avg_rating", "watched"} <= set(rec.user_profile(1).columns)
    expect_error(rec.user_profile, 99999)

    # accuracy: personalised models beat the popularity baseline
    res = evaluate(movies, ratings)
    assert res.loc["hybrid", "Precision@10"] > res.loc["popular", "Precision@10"]
    assert res.loc["collaborative", "Precision@10"] > res.loc["popular", "Precision@10"]


if __name__ == "__main__":
    test_all()
    print("All tests passed!")
