"""
main.py — ApniDukaan Recommendation Engine: End-to-End Pipeline Demo

This script demonstrates the complete recommendation pipeline:
    1. Load raw product data from CSV
    2. Clean and preprocess text features
    3. Build TF-IDF vectors and cosine similarity matrix
    4. Query recommendations for sample products
    5. Display formatted results

Usage:
    python main.py
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_processor import process_pipeline
from src.recommender import ContentBasedRecommender


def display_recommendations(recommender: ContentBasedRecommender, 
                            product_name: str, 
                            top_n: int = 5) -> None:
    """Display formatted recommendations for a given product.
    
    Args:
        recommender: Fitted ContentBasedRecommender instance.
        product_name: Name of the product to query.
        top_n: Number of recommendations to display.
    """
    print(f"\n{'─' * 70}")
    print(f"  🔍 Query: '{product_name}'")
    print(f"{'─' * 70}")
    
    try:
        results = recommender.get_recommendations(product_name, top_n=top_n)
        
        print(f"  {'Rank':<6} {'Product':<35} {'Category':<18} {'Score':<8} {'Price':>8}")
        print(f"  {'─'*6} {'─'*35} {'─'*18} {'─'*8} {'─'*8}")
        
        for i, row in results.iterrows():
            rank = i + 1
            name = row['product_name'][:33]
            category = row['category'][:16]
            score = row['similarity_score']
            price = f"₹{row['price']:,.0f}"
            print(f"  {rank:<6} {name:<35} {category:<18} {score:<8.4f} {price:>8}")
        
        avg_score = results['similarity_score'].mean()
        print(f"\n  📊 Average similarity score: {avg_score:.4f}")
    
    except KeyError as e:
        print(f"  ❌ Product not found: {e}")
    except ValueError as e:
        print(f"  ⚠️  Invalid query: {e}")


def main() -> None:
    """Run the complete ApniDukaan recommendation pipeline."""
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  🛒 ApniDukaan — Product Recommendation Engine".ljust(68) + "║")
    print("║" + "     Content-Based Filtering with TF-IDF + Cosine Similarity".ljust(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # ── Step 1: Data Processing Pipeline ──────────────────────────────────
    df = process_pipeline()
    
    # ── Step 2: Fit Recommendation Model ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  Fitting Recommendation Model")
    print("=" * 60)
    
    recommender = ContentBasedRecommender(
        max_features=5000,     # Vocabulary cap for TF-IDF
        ngram_range=(1, 2),    # Unigrams + bigrams for richer matching
    )
    recommender.fit(df)
    
    # Print model statistics
    stats = recommender.get_model_stats()
    print("Model Statistics:")
    print(f"  • Products indexed:  {stats['n_products']}")
    print(f"  • TF-IDF features:   {stats['n_features']}")
    print(f"  • Vocabulary size:   {stats['vocabulary_size']}")
    print(f"  • Matrix sparsity:   {stats['sparsity']:.2%}")
    print(f"  • Avg similarity:    {stats['avg_similarity']:.4f}")
    
    # ── Step 3: Demo Recommendations ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  📋 RECOMMENDATION DEMOS")
    print("=" * 70)
    
    # Demo 1: Electronics — Smartphone query
    display_recommendations(recommender, "Samsung Galaxy S24 Ultra")
    
    # Demo 2: Fashion — Footwear query
    display_recommendations(recommender, "Nike Air Max 270")
    
    # Demo 3: Books — Self-help query
    display_recommendations(recommender, "Atomic Habits by James Clear")
    
    # Demo 4: Home & Kitchen query
    display_recommendations(recommender, "Prestige Omega Deluxe Tawa")
    
    # ── Step 4: Partial Search Demo ───────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  🔎 PARTIAL SEARCH DEMO")
    print(f"{'═' * 70}")
    
    search_terms = ["sony", "nike", "yoga"]
    for term in search_terms:
        matches = recommender.find_product(term, limit=5)
        print(f"\n  Search '{term}' → {len(matches)} results:")
        for match in matches:
            print(f"    • {match}")
    
    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  ✅ Pipeline complete! All recommendations generated successfully.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
