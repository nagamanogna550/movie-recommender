"""Generates sample movies.csv and ratings.csv (seeded, so reproducible).

To use real data, replace these files with MovieLens (grouplens.org) converted
to the same columns: movies(movie_id,title,genres,plot), ratings(user_id,movie_id,rating).
"""
import numpy as np
import pandas as pd

MOVIES = [
    ("The Matrix", "Action|Sci-Fi", "A hacker learns his world is a simulation run by machines and joins a rebellion to free humanity."),
    ("Inception", "Action|Sci-Fi|Thriller", "A thief who steals secrets from dreams is hired to plant an idea inside a target's mind."),
    ("Interstellar", "Sci-Fi|Drama", "Astronauts travel through a wormhole to find a new home for humanity as Earth fails."),
    ("Blade Runner 2049", "Sci-Fi|Thriller", "A replicant hunter uncovers a long buried secret that could plunge society into chaos."),
    ("Arrival", "Sci-Fi|Drama", "A linguist is recruited to communicate with alien visitors before the world goes to war."),
    ("Alien", "Sci-Fi|Horror", "The crew of a cargo ship is hunted by a deadly creature that boarded in deep space."),
    ("Die Hard", "Action|Thriller", "A police officer fights terrorists who have taken over a skyscraper during a holiday party."),
    ("Mad Max: Fury Road", "Action|Sci-Fi", "In a desert wasteland a drifter and a rebel driver race across the sands to escape a tyrant."),
    ("John Wick", "Action|Crime|Thriller", "A retired hitman returns to the underworld to take revenge on those who wronged him."),
    ("The Dark Knight", "Action|Crime|Drama", "A masked vigilante faces a chaotic criminal mastermind who pushes a city toward collapse."),
    ("Gladiator", "Action|Drama", "A betrayed Roman general becomes a gladiator and seeks revenge on the corrupt emperor."),
    ("Avengers: Endgame", "Action|Sci-Fi|Fantasy", "Surviving heroes unite for a final battle to undo a catastrophe that erased half of all life."),
    ("The Godfather", "Crime|Drama", "The aging head of a crime family hands control to his reluctant youngest son."),
    ("Goodfellas", "Crime|Drama", "A young man rises through the ranks of the mob and slowly watches his world unravel."),
    ("Se7en", "Crime|Thriller|Horror", "Two detectives hunt a serial killer who stages murders around the seven deadly sins."),
    ("The Silence of the Lambs", "Crime|Thriller|Horror", "A trainee investigator seeks help from an imprisoned cannibal to catch another killer."),
    ("The Conjuring", "Horror|Thriller", "Paranormal investigators help a family terrorized by a dark presence in their farmhouse."),
    ("Get Out", "Horror|Thriller", "A young man visiting his girlfriend's family discovers a disturbing secret behind their warm welcome."),
    ("A Quiet Place", "Horror|Sci-Fi|Thriller", "A family must live in silence to hide from creatures that hunt by sound."),
    ("Jaws", "Thriller|Horror|Adventure", "A police chief and two companions hunt a giant shark terrorizing a beach town."),
    ("Titanic", "Romance|Drama", "A young artist and a wealthy girl fall in love aboard a doomed ocean liner."),
    ("The Notebook", "Romance|Drama", "A man reads the story of a passionate summer romance to a woman with fading memory."),
    ("La La Land", "Romance|Drama|Musical", "A jazz pianist and an aspiring actress chase their dreams and struggle to keep their love."),
    ("Pride and Prejudice", "Romance|Drama", "A spirited young woman and a proud wealthy gentleman overcome their first impressions."),
    ("Forrest Gump", "Drama|Romance|Comedy", "A kind man with a low IQ unwittingly takes part in defining moments of American history."),
    ("The Shawshank Redemption", "Drama|Crime", "A banker sentenced to prison forms a lasting friendship and never gives up hope."),
    ("Parasite", "Drama|Thriller|Comedy", "A poor family schemes its way into the household of a rich family with dark consequences."),
    ("Whiplash", "Drama|Musical", "An ambitious young drummer is pushed to his limits by a ruthless music teacher."),
    ("Superbad", "Comedy", "Two awkward high school friends try to buy alcohol for a party before graduation."),
    ("The Hangover", "Comedy", "Three friends wake up after a bachelor party with no memory and have to find the missing groom."),
    ("Bridesmaids", "Comedy|Romance", "A maid of honor's life unravels as she competes with a rival while planning a wedding."),
    ("Toy Story", "Animation|Comedy|Adventure", "A cowboy doll feels threatened when a new space ranger toy becomes the favorite of their owner."),
    ("Finding Nemo", "Animation|Adventure|Comedy", "An anxious clownfish crosses the ocean with a forgetful friend to rescue his lost son."),
    ("Shrek", "Animation|Comedy|Fantasy", "A grumpy ogre sets out to rescue a princess so he can get his swamp back."),
    ("Frozen", "Animation|Musical|Fantasy", "A fearless princess sets off to find her sister whose icy powers trapped the kingdom in winter."),
    ("Spirited Away", "Animation|Fantasy|Adventure", "A girl wanders into a world of spirits and must work in a bathhouse to free her parents."),
    ("Up", "Animation|Adventure|Comedy", "An elderly widower ties balloons to his house and flies to South America with a young stowaway."),
    ("Coco", "Animation|Musical|Fantasy", "A boy who dreams of music enters the land of the dead to uncover his family's history."),
    ("The Lord of the Rings", "Fantasy|Adventure|Action", "A young hobbit and his allies journey to destroy a powerful ring and defeat a dark lord."),
    ("Harry Potter", "Fantasy|Adventure", "An orphan learns he is a wizard and begins his education at a school of magic."),
    ("Pan's Labyrinth", "Fantasy|Drama|Horror", "In war torn Spain a girl escapes into a dark fantasy world of mythical creatures."),
]
GENRES = sorted({g for _, gs, _ in MOVIES for g in gs.split("|")})


def main(n_users=80, seed=42):
    rng = np.random.default_rng(seed)
    movies = pd.DataFrame(MOVIES, columns=["title", "genres", "plot"])
    movies.insert(0, "movie_id", range(1, len(movies) + 1))
    movies.to_csv("movies.csv", index=False)

    rows = []
    for user in range(1, n_users + 1):
        fav = set(rng.choice(GENRES, size=2, replace=False))
        n = rng.integers(12, 25)
        for mid in rng.choice(movies.movie_id, size=n, replace=False):
            g = set(movies.loc[movies.movie_id == mid, "genres"].iloc[0].split("|"))
            if g & fav and rng.random() < 0.9:
                rating = rng.choice([4, 5], p=[0.4, 0.6])
            else:
                rating = rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            rows.append((user, int(mid), int(rating)))
    pd.DataFrame(rows, columns=["user_id", "movie_id", "rating"]).to_csv("ratings.csv", index=False)
    print(f"Wrote movies.csv ({len(movies)} movies) and ratings.csv ({len(rows)} ratings)")


if __name__ == "__main__":
    main()
