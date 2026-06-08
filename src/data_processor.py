"""
data_processor.py — Data Loading & Preprocessing Pipeline

Responsible for:
- Loading raw CSV product data
- Cleaning missing values, whitespace, and text normalization
- Constructing the combined 'soup' feature column for TF-IDF vectorization

Design Note:
    This module intentionally keeps data I/O and transformation separate
    from the recommendation logic (recommender.py) to maintain single-
    responsibility and testability.
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "products.csv")

# Columns used to build the text 'soup' for vectorization.
# Each column is chosen for a specific reason:
#   - product_name:  captures the core identity of the product
#   - category:      groups products by broad domain (Electronics, Books, …)
#   - sub_category:  adds finer granularity within a category
#   - brand:         brand affinity is a strong signal for similarity
#   - description:   rich free-text with features, materials, use-cases
#   - tags:          curated keywords that highlight key attributes
SOUP_COLUMNS = ["product_name", "category", "sub_category", "brand", "description", "tags"]


def load_data(filepath: str = DEFAULT_CSV) -> pd.DataFrame:
    """Load the raw product CSV into a DataFrame.
    
    Args:
        filepath: Path to the CSV file. Defaults to data/products.csv.
        
    Returns:
        Raw DataFrame as read from disk.
        
    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[DataProcessor] Loaded {len(df)} products from {filepath}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the product DataFrame.
    
    Steps:
        1. Strip leading/trailing whitespace from all string columns.
        2. Fill missing text fields with empty strings.
        3. Normalize text columns to lowercase for consistent vectorization.
        4. Ensure price and rating are numeric; fill missing numerics with median.
    
    Args:
        df: Raw product DataFrame.
        
    Returns:
        Cleaned DataFrame with normalized text and no missing values.
    """
    df = df.copy()  # avoid mutating the original
    
    # --- Step 1 & 2: Strip whitespace and fill missing text ---
    text_cols = ["product_name", "category", "sub_category", "brand", "description", "tags"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna("")        # replace NaN with empty string
                .astype(str)
                .str.strip()       # remove leading/trailing whitespace
                .str.lower()       # normalize to lowercase
            )
    
    # --- Step 3: Ensure numeric columns are clean ---
    for num_col in ["price", "rating"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            median_val = df[num_col].median()
            df[num_col] = df[num_col].fillna(median_val)
    
    missing_count = df[text_cols].eq("").sum().sum()
    print(f"[DataProcessor] Cleaning complete. Remaining empty text fields: {missing_count}")
    return df


def build_soup(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the combined 'soup' text feature for TF-IDF vectorization.
    
    The soup column concatenates multiple text columns into a single string
    per product. This gives the TF-IDF vectorizer a richer representation
    that captures product identity, category context, brand affinity,
    descriptive features, and curated tags — all in one place.
    
    Why each column is included:
        - product_name:  captures the core product identity and key terms
        - category:      ensures same-domain products share vocabulary
        - sub_category:  provides finer-grained grouping signal
        - brand:         users often prefer similar brands; encodes affinity
        - description:   free-text with features, materials, and use-cases
        - tags:          curated keywords amplify important attributes
    
    Args:
        df: Cleaned product DataFrame.
        
    Returns:
        DataFrame with an added 'soup' column.
    """
    df = df.copy()
    
    # Concatenate all soup columns with a space separator.
    # Each column contributes different semantic signals to the combined text.
    df["soup"] = df[SOUP_COLUMNS].apply(
        lambda row: " ".join(row.values.astype(str)),
        axis=1
    )
    
    # Remove extra whitespace that may result from empty fields
    df["soup"] = df["soup"].str.replace(r"\s+", " ", regex=True).str.strip()
    
    print(f"[DataProcessor] Built 'soup' column. Average length: {df['soup'].str.len().mean():.0f} chars")
    return df


def process_pipeline(filepath: str = DEFAULT_CSV) -> pd.DataFrame:
    """Run the full data processing pipeline: Load → Clean → Build Soup.
    
    This is the primary entry point for other modules. It returns a fully
    processed DataFrame ready for TF-IDF vectorization.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        Processed DataFrame with clean text and a 'soup' column.
    """
    print("\n" + "=" * 60)
    print("  ApniDukaan — Data Processing Pipeline")
    print("=" * 60)
    
    df = load_data(filepath)
    df = clean_data(df)
    df = build_soup(df)
    
    print(f"[DataProcessor] Pipeline complete. Shape: {df.shape}")
    print("=" * 60 + "\n")
    return df


if __name__ == "__main__":
    # Quick standalone test
    df = process_pipeline()
    print(df[["product_id", "product_name", "soup"]].head(10).to_string())
