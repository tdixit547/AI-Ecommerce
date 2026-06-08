# 🛒 ApniDukaan — Product Recommendation Engine

> A **content-based product recommendation system** built for [ApniDukaan](https://github.com), a fictional Indian e-commerce platform. The engine suggests relevant products to users based on product attributes using machine learning.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Project Overview

ApniDukaan serves millions of Indian consumers across categories like **Electronics**, **Fashion**, **Books**, **Home & Kitchen**, and **Sports**. This recommendation engine powers the _"Customers who viewed this also liked..."_ feature by analyzing product text attributes and finding similar items.

### How It Works

The engine uses a **content-based filtering** approach:

1. **Feature Engineering** — Multiple text columns (product name, category, brand, description, tags) are combined into a single "soup" text per product.
2. **TF-IDF Vectorization** — The text soup is transformed into numerical feature vectors using [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (Term Frequency–Inverse Document Frequency), which captures the importance of words relative to the entire product catalog.
3. **Cosine Similarity** — Pairwise cosine similarity is computed between all product vectors, producing a similarity matrix where each entry represents how "similar" two products are (0 = unrelated, 1 = identical).
4. **Ranking** — For any query product, the top-N most similar products are returned, excluding the query itself.

---

##  Project Structure

```
apni-dukaan-recommender/
├── data/
│   └── products.csv            # Product catalog (300 products, 9 columns)
├── src/
│   ├── __init__.py
│   ├── data_processor.py       # Data loading, cleaning, soup construction
│   ├── recommender.py          # TF-IDF + cosine similarity engine
│   └── evaluate.py             # Quality evaluation test suite
├── main.py                     # End-to-end pipeline demo
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```

---

##  Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/apni-dukaan-recommender.git
cd apni-dukaan-recommender

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

##  Usage

### Run the Full Pipeline

```bash
python main.py
```

This runs the complete pipeline: **Load Data → Clean & Preprocess → Vectorize → Query → Display Results**.

### Run the Evaluation Suite

```bash
python -m src.evaluate
```

This runs 8 structured test cases covering exact matches, partial searches, cross-category behavior, error handling, and score validation.

### Run Individual Modules

```bash
# Test data processing only
python -m src.data_processor

# Test recommender standalone
python -m src.recommender
```

---

##  Example Output

```
══════════════════════════════════════════════════════════════════════════
  📋 RECOMMENDATION DEMOS
══════════════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────────────────
  🔍 Query: 'Samsung Galaxy S24 Ultra'
──────────────────────────────────────────────────────────────────────────
  Rank   Product                             Category           Score      Price
  ────── ─────────────────────────────────── ────────────────── ──────── ────────
  1      oneplus nord ce 4                   electronics        0.3842   ₹24,999
  2      samsung galaxy a15                  electronics        0.3521   ₹13,999
  3      xiaomi 14 ultra                     electronics        0.3198   ₹89,999
  4      realme narzo 70x                    electronics        0.2876   ₹11,999
  5      apple iphone 15 pro                 electronics        0.2654   ₹1,34,900
```

*(Actual scores will vary based on the dataset)*

---

##  Evaluation Results

The evaluation suite (`evaluate.py`) runs **8 test cases**:

| # | Test Case | Description |
|---|-----------|-------------|
| 1 | **Exact Match Query** | Validates recommendations for a known product |
| 2 | **Partial Name Search** | Tests fuzzy product name lookup |
| 3 | **Same-Category Relevance** | Checks if recommendations stay in-category |
| 4 | **Cross-Category Separation** | Ensures books don't recommend electronics |
| 5 | **Unknown Product Handling** | Graceful `KeyError` for missing products |
| 6 | **Empty Query Handling** | Graceful `ValueError` for empty input |
| 7 | **Score Bounds [0, 1]** | All similarity scores within valid range |
| 8 | **Self-Exclusion Check** | Product never recommends itself |

---

##  Technical Details

### Dataset
- **300 products** across 5 categories and 25+ sub-categories
- Realistic Indian and international brands
- Prices in INR with category-appropriate ranges
- Deliberately includes ~5% missing values and whitespace noise for realistic data cleaning

### ML Pipeline
- **TF-IDF Vectorizer**: `max_features=5000`, `ngram_range=(1, 2)`, English stop words removed
- **Cosine Similarity**: Computed on the full TF-IDF matrix — O(n²) space
- **Soup Construction**: Intentional combination of `name + category + sub_category + brand + description + tags` to capture multi-dimensional product similarity

### Design Decisions
- **Stateful Recommender**: The `ContentBasedRecommender` class caches the similarity matrix after `fit()`, enabling fast repeated queries without recomputation
- **Modular Architecture**: Each file has a single responsibility — data I/O, model logic, evaluation, and orchestration are fully separated
- **Production Patterns**: Guard clauses, input validation, score bounds checking, and comprehensive error handling

---

##  Future Improvements

- [ ] **Collaborative Filtering** — Incorporate user purchase/browsing history for hybrid recommendations
- [ ] **User Profiles** — Build user preference vectors based on interaction history
- [ ] **Real-time Updates** — Incremental model updates as new products are added
- [ ] **Flask/FastAPI** — RESTful API endpoint for integration with frontend
- [ ] **A/B Testing Framework** — Compare recommendation strategies in production
- [ ] **Price-aware Recommendations** — Weight similarity by price range proximity
- [ ] **Image Embeddings** — Use product images alongside text features
- [ ] **Deployment** — Containerize with Docker and deploy on AWS/GCP

---

##  License

This project is open-source under the [MIT License](LICENSE).

---

##  Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

<p align="center">
  Built with ❤️ for the ApniDukaan platform
</p>
