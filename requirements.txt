# 🛍️ Hybrid Product Recommendation System

An IEEE-published hybrid recommendation system that suggests products using popularity-based filtering and content-based similarity techniques.

Built using Scikit-learn, TF-IDF vectorization, cosine similarity, and Streamlit.

---

## 📄 IEEE Publication

This project was published as part of an IEEE research paper.

**IEEE Link:** [Read the full paper on IEEE Xplore](https://ieeexplore.ieee.org/document/10452564)

---

## 📸 Application Preview

### Streamlit Dashboard

![Streamlit Dashboard](assets/app-ui.png)

---

### Product Recommendation Results

![Product Recommendation Results](assets/recommendation-result.png)

---

## 🚀 Features

- Hybrid recommendation system
- Popularity-based filtering
- Content-based filtering using TF-IDF
- Cosine similarity recommendation engine
- Product search functionality
- Category-based recommendations
- Interactive Streamlit dashboard
- Product popularity ranking

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- Pandas
- NumPy
- Matplotlib

---

## 📂 Project Structure

```bash
hybrid-product-recommendation-system/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   └── sampled_dataset.csv
├── assets/
│   ├── app-ui.png
│   └── recommendation-result.png
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── recommender.py
    └── visualization.py

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/ShaiveSharma02/hybrid-product-recommendation-system.git
cd hybrid-product-recommendation-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
python -m streamlit run app.py
```

---

## 🧠 Recommendation Workflow

1. Load and preprocess product dataset
2. Generate combined product features
3. Apply TF-IDF vectorization
4. Compute cosine similarity scores
5. Generate personalized recommendations
6. Rank products using popularity metrics
7. Display recommendations in Streamlit dashboard

---

## 📈 Recommendation Techniques

### Popularity-Based Filtering

Ranks products based on:
- Product ratings
- Number of reviews
- Overall popularity score

### Content-Based Filtering

Uses:
- TF-IDF Vectorization
- Product metadata analysis
- Cosine similarity scoring

---

## ⚠️ Disclaimer

This project is intended for educational, research, and portfolio purposes only.

Product recommendations may vary depending on dataset quality and preprocessing.

---

## 👨‍💻 Author

Shaive Sharma
