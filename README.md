# Movie Recommendation System

Personalised movie suggestions using Pandas and Scikit-learn.

## Approaches
- **Content-based:** TF-IDF on genres + plot text, cosine similarity between movies
- **Collaborative filtering:** item-item similarity on mean-centred user ratings
- **Hybrid:** weighted blend of both (default alpha = 0.5)
- **Cold start:** unknown users get Bayesian-average popular movies
- **User taste profile:** favourite genres from watch history

## Setup
pip install -r requirements.txt
python make_data.py        (creates sample movies.csv and ratings.csv)

## Run
python recommender.py

## Test
python test_recommender.py

## Evaluation
Per user, 20% of ratings are hidden; models are trained on the rest and scored by
Precision@10 / Recall@10 on hidden movies rated >= 4. Compared against a popularity baseline.

Replace the CSVs with MovieLens data (same columns) to use real data.
