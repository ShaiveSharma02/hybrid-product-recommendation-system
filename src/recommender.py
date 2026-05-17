import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_model(data):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    feature_matrix = vectorizer.fit_transform(
        data["combined_features"]
    )

    return vectorizer, feature_matrix


def get_popular_products(data, top_n=10):
    popular_products = data.copy()

    popular_products["popularity_score"] = (
        popular_products["ratings_clean"] *
        popular_products["no_of_ratings_clean"]
    )

    popular_products = popular_products.sort_values(
        by=["popularity_score", "ratings_clean"],
        ascending=False
    )

    return popular_products.head(top_n)


def recommend_similar_products(
    product_name,
    data,
    vectorizer,
    feature_matrix,
    top_n=10
):
    matching_products = data[
        data["name"].str.contains(
            product_name,
            case=False,
            na=False
        )
    ]

    if matching_products.empty:
        return pd.DataFrame()

    selected_index = matching_products.index[0]

    selected_position = data.index.get_loc(selected_index)

    similarity_scores = cosine_similarity(
        feature_matrix[selected_position],
        feature_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1][1:top_n + 1]

    recommendations = data.iloc[similar_indices].copy()
    recommendations["similarity_score"] = similarity_scores[similar_indices]

    return recommendations


def search_products(data, keyword, top_n=10):
    results = data[
        data["name"].str.contains(
            keyword,
            case=False,
            na=False
        )
    ]

    return results.head(top_n)


def filter_by_category(data, main_category, top_n=10):
    filtered_data = data[
        data["main_category"] == main_category
    ]

    return get_popular_products(filtered_data, top_n)