import streamlit as st

from src.data_preprocessing import load_data, preprocess_data
from src.recommender import (
    build_similarity_model,
    get_popular_products,
    recommend_similar_products,
    search_products,
    filter_by_category
)
from src.visualization import plot_category_distribution


DATA_PATH = "data/sampled_dataset.csv"


@st.cache_resource
def load_recommendation_system():
    raw_data = load_data(DATA_PATH)
    data = preprocess_data(raw_data)

    vectorizer, feature_matrix = build_similarity_model(data)

    return data, vectorizer, feature_matrix


def display_products(products):
    if products.empty:
        st.warning("No products found.")
        return

    for _, product in products.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])

            with col1:
                if product.get("image"):
                    st.image(
                        product["image"],
                        width=120
                    )

            with col2:
                st.subheader(product["name"])

                st.write(
                    f"**Category:** {product['main_category']} | "
                    f"**Sub-category:** {product['sub_category']}"
                )

                st.write(
                    f"**Rating:** {product['ratings_clean']} ⭐ | "
                    f"**Reviews:** {product['no_of_ratings_clean']}"
                )

                st.write(
                    f"**Discount Price:** ₹{product['discount_price_clean']:,.0f}"
                )

                if product.get("actual_price_clean"):
                    st.write(
                        f"**Actual Price:** ₹{product['actual_price_clean']:,.0f}"
                    )

                if product.get("link"):
                    st.link_button(
                        "View Product",
                        product["link"]
                    )

            st.divider()


st.set_page_config(
    page_title="Hybrid Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Hybrid Product Recommendation System")

st.write(
    "An IEEE-published hybrid recommendation system that suggests products "
    "using popularity-based filtering and content-based similarity."
)

data, vectorizer, feature_matrix = load_recommendation_system()

st.sidebar.header("Recommendation Controls")

mode = st.sidebar.selectbox(
    "Choose Recommendation Mode",
    [
        "Popular Products",
        "Search Products",
        "Similar Products",
        "Category Recommendations"
    ]
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

if mode == "Popular Products":
    st.subheader("🔥 Popular Product Recommendations")

    recommendations = get_popular_products(
        data,
        top_n=top_n
    )

    display_products(recommendations)

elif mode == "Search Products":
    st.subheader("🔎 Product Search")

    keyword = st.text_input(
        "Enter product keyword",
        placeholder="Example: shoes, watch, bag"
    )

    if keyword:
        results = search_products(
            data,
            keyword,
            top_n=top_n
        )

        display_products(results)

elif mode == "Similar Products":
    st.subheader("🎯 Similar Product Recommendations")

    product_name = st.text_input(
        "Enter product name or keyword",
        placeholder="Example: shoes, watch, jewellery"
    )

    if product_name:
        recommendations = recommend_similar_products(
            product_name,
            data,
            vectorizer,
            feature_matrix,
            top_n=top_n
        )

        display_products(recommendations)

elif mode == "Category Recommendations":
    st.subheader("📦 Category-Based Recommendations")

    categories = sorted(
        data["main_category"].dropna().unique()
    )

    selected_category = st.selectbox(
        "Select product category",
        categories
    )

    recommendations = filter_by_category(
        data,
        selected_category,
        top_n=top_n
    )

    display_products(recommendations)

st.divider()

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Products", f"{len(data):,}")

with col2:
    st.metric("Main Categories", data["main_category"].nunique())

with col3:
    st.metric("Sub Categories", data["sub_category"].nunique())

st.subheader("Category Distribution")

fig = plot_category_distribution(data)
st.pyplot(fig)