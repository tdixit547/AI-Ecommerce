"""
recommender.py — Content-Based Recommendation Engine

Responsible for:
- TF-IDF vectorization of product text features
- Computing cosine similarity between all product pairs
- Returning top-N similar products for a given query

Design Note:
    The recommender is intentionally stateful — once fit() is called,
    the similarity matrix is cached in memory for fast repeated queries.
    This mirrors a production pattern where the model is trained once
    and served many times.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-based product recommender using TF-IDF + Cosine Similarity.
    
    This class encapsulates the full recommendation workflow:
        1. Fit a TF-IDF vectorizer on product 'soup' text.
        2. Compute the pairwise cosine similarity matrix.
        3. Serve top-N recommendations for any product query.
    
    Attributes:
        tfidf_vectorizer: Fitted TfidfVectorizer instance.
        tfidf_matrix: Sparse TF-IDF matrix (n_products × n_features).
        similarity_matrix: Dense cosine similarity matrix (n_products × n_products).
        df: Reference to the processed product DataFrame.
        product_indices: Mapping from lowercase product name → DataFrame index.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """Initialize the recommender with TF-IDF hyperparameters.
        
        Args:
            max_features: Maximum number of TF-IDF features to extract.
                Higher values capture more vocabulary but increase memory.
            ngram_range: Range of n-grams to consider. (1, 2) captures
                both unigrams and bigrams for richer phrase matching.
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",  # Remove common English stop words
            # strip_accents and lowercase are handled by data_processor,
            # but we keep lowercase=True here as a safety net
            lowercase=True,
            dtype=np.float32,  # Use float32 to reduce memory footprint
        )
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.df = None
        self.product_indices = None
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        """Fit the recommender on a processed product DataFrame.
        
        This method:
            1. Validates that the 'soup' column exists.
            2. Fits and transforms the TF-IDF vectorizer.
            3. Computes the full cosine similarity matrix.
            4. Builds a name → index lookup for fast querying.
        
        Args:
            df: Processed DataFrame with a 'soup' column (from data_processor).
            
        Returns:
            self (for method chaining).
            
        Raises:
            ValueError: If 'soup' column is missing from the DataFrame.
        """
        if "soup" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'soup' column. "
                "Run data_processor.process_pipeline() first."
            )
        
        self.df = df.reset_index(drop=True)
        
        # Step 1: TF-IDF Vectorization
        # Transform the combined text 'soup' into a numerical feature matrix.
        # Each row becomes a sparse vector of TF-IDF weights.
        print("[Recommender] Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df["soup"])
        print(f"[Recommender] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Step 2: Cosine Similarity
        # Compute pairwise similarity between all products.
        # Result: a symmetric matrix where entry (i, j) is the cosine
        # similarity between product i and product j.
        print("[Recommender] Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Validate: all similarity scores must be in [0, 1]
        assert self.similarity_matrix.min() >= 0.0, "Similarity scores below 0 detected!"
        assert self.similarity_matrix.max() <= 1.0 + 1e-6, "Similarity scores above 1 detected!"
        print(f"[Recommender] Similarity matrix shape: {self.similarity_matrix.shape}")
        
        # Step 3: Build product name → index mapping (lowercase for case-insensitive lookup)
        self.product_indices = pd.Series(
            self.df.index,
            index=self.df["product_name"].str.strip().str.lower()
        )
        
        # Handle duplicate product names: keep the first occurrence
        self.product_indices = self.product_indices[~self.product_indices.index.duplicated(keep="first")]
        
        self._is_fitted = True
        print(f"[Recommender] Model fitted on {len(self.df)} products.\n")
        return self
    
    def get_recommendations(
        self, 
        product_name: str, 
        top_n: int = 5
    ) -> pd.DataFrame:
        """Get top-N product recommendations for a given product.
        
        Args:
            product_name: Name of the query product (case-insensitive).
            top_n: Number of recommendations to return. Defaults to 5.
            
        Returns:
            DataFrame with columns: product_id, product_name, category,
            sub_category, brand, price, rating, similarity_score.
            Sorted by similarity_score descending.
            
        Raises:
            RuntimeError: If the model has not been fitted yet.
            ValueError: If product_name is empty or None.
            KeyError: If the product is not found in the dataset.
        """
        # --- Guard clauses ---
        if not self._is_fitted:
            raise RuntimeError(
                "Recommender is not fitted. Call .fit(df) first."
            )
        
        if not product_name or not product_name.strip():
            raise ValueError("Product name cannot be empty.")
        
        # Normalize query to lowercase for case-insensitive matching
        query = product_name.strip().lower()
        
        if query not in self.product_indices.index:
            raise KeyError(
                f"Product '{product_name}' not found in the dataset. "
                f"Available products: {len(self.product_indices)}"
            )
        
        # --- Core recommendation logic ---
        # 1. Get the index of the query product
        idx = self.product_indices[query]
        
        # 2. Get pairwise similarity scores for this product vs all others
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # 3. Sort by similarity (descending), skip the first one (itself, score=1.0)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # 4. CRITICAL: Exclude the query product itself from recommendations
        sim_scores = [
            (i, score) for i, score in sim_scores
            if i != idx
        ]
        
        # 5. Take top N
        top_matches = sim_scores[:top_n]
        
        # 6. Build result DataFrame
        result_indices = [i for i, _ in top_matches]
        result_scores = [score for _, score in top_matches]
        
        result_df = self.df.iloc[result_indices][
            ["product_id", "product_name", "category", "sub_category", 
             "brand", "price", "rating"]
        ].copy()
        result_df["similarity_score"] = result_scores
        result_df = result_df.reset_index(drop=True)
        
        # Final validation: ensure all scores are in [0, 1]
        assert result_df["similarity_score"].between(0, 1).all(), \
            "ERROR: Similarity scores outside [0, 1] range detected!"
        
        return result_df
    
    def find_product(self, partial_name: str, limit: int = 10) -> list[str]:
        """Search for products by partial name match.
        
        Useful when the exact product name is unknown. Returns matching
        product names that can be passed to get_recommendations().
        
        Args:
            partial_name: Partial product name to search for (case-insensitive).
            limit: Maximum number of matches to return.
            
        Returns:
            List of matching product names.
        """
        if not partial_name or not partial_name.strip():
            return []
        
        query = partial_name.strip().lower()
        matches = [
            name for name in self.product_indices.index
            if query in name
        ]
        return matches[:limit]
    
    def get_model_stats(self) -> dict:
        """Return summary statistics about the fitted model.
        
        Returns:
            Dictionary with model statistics including vocabulary size,
            number of products, and similarity matrix stats.
        """
        if not self._is_fitted:
            return {"status": "not fitted"}
        
        return {
            "status": "fitted",
            "n_products": len(self.df),
            "n_features": self.tfidf_matrix.shape[1],
            "vocabulary_size": len(self.tfidf_vectorizer.vocabulary_),
            "avg_similarity": float(np.mean(self.similarity_matrix)),
            "max_similarity": float(np.max(self.similarity_matrix)),
            "min_similarity": float(np.min(self.similarity_matrix)),
            "sparsity": float(1.0 - (self.tfidf_matrix.nnz / np.prod(self.tfidf_matrix.shape))),
        }


if __name__ == "__main__":
    # Quick standalone test
    from data_processor import process_pipeline
    
    df = process_pipeline()
    rec = ContentBasedRecommender()
    rec.fit(df)
    
    print("\nModel Stats:")
    for k, v in rec.get_model_stats().items():
        print(f"  {k}: {v}")
    
    print("\nSample Recommendations:")
    results = rec.get_recommendations("Samsung Galaxy S24 Ultra", top_n=5)
    print(results.to_string(index=False))
