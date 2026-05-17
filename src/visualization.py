import matplotlib.pyplot as plt


def plot_category_distribution(data):
    category_counts = data["main_category"].value_counts().head(10)

    fig, ax = plt.subplots()

    ax.barh(
        category_counts.index,
        category_counts.values
    )

    ax.set_title("Top Product Categories")
    ax.set_xlabel("Number of Products")
    ax.set_ylabel("Category")

    return fig