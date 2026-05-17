import pandas as pd


PRICE_COLUMNS = [
    "discount_price",
    "actual_price"
]

REQUIRED_COLUMNS = [
    "name",
    "main_category",
    "sub_category",
    "image",
    "link",
    "ratings",
    "no_of_ratings",
    "discount_price",
    "actual_price"
]


def clean_price(price):
    if pd.isna(price):
        return None

    price = str(price)
    price = price.replace("₹", "")
    price = price.replace(",", "")
    price = price.strip()

    if price == "":
        return None

    try:
        return float(price)
    except ValueError:
        return None


def clean_rating(rating):
    if pd.isna(rating):
        return 0.0

    rating = str(rating).strip()

    try:
        return float(rating)
    except ValueError:
        return 0.0


def clean_rating_count(count):
    if pd.isna(count):
        return 0

    count = str(count)
    count = count.replace(",", "")
    count = count.strip()

    try:
        return int(float(count))
    except ValueError:
        return 0


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    available_columns = [
        column for column in REQUIRED_COLUMNS
        if column in data.columns
    ]

    data = data[available_columns]

    return data


def preprocess_data(data):
    data = data.copy()

    data["name"] = data["name"].fillna("Unknown Product")
    data["main_category"] = data["main_category"].fillna("Unknown")
    data["sub_category"] = data["sub_category"].fillna("Unknown")
    data["image"] = data["image"].fillna("")
    data["link"] = data["link"].fillna("")

    data["discount_price_clean"] = data["discount_price"].apply(clean_price)
    data["actual_price_clean"] = data["actual_price"].apply(clean_price)
    data["ratings_clean"] = data["ratings"].apply(clean_rating)
    data["no_of_ratings_clean"] = data["no_of_ratings"].apply(clean_rating_count)

    data["combined_features"] = (
        data["name"].astype(str) + " " +
        data["main_category"].astype(str) + " " +
        data["sub_category"].astype(str)
    )

    data = data.dropna(
        subset=["discount_price_clean"]
    )

    return data