"""
evaluate.py — Recommendation Quality Evaluation Suite

Responsible for:
- Running structured test cases against the recommendation engine
- Validating similarity scores and edge-case handling
- Producing a formatted evaluation report

Design Note:
    This module tests the recommendation quality from multiple angles:
    exact matches, partial matches, cross-category behavior, and error
    handling. Each test case is designed to validate a specific aspect
    of the recommendation engine's behavior.
"""

import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_processor import process_pipeline
from src.recommender import ContentBasedRecommender


# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------
class TestResult:
    """Container for a single test case result."""
    
    def __init__(self, name: str, passed: bool, details: str = ""):
        self.name = name
        self.passed = passed
        self.details = details
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"  [{status}] {self.name}\n          {self.details}"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_exact_match(recommender: ContentBasedRecommender) -> TestResult:
    """Test 1: Exact product name match.
    
    Verifies that querying with a known product name returns valid
    recommendations with reasonable similarity scores.
    """
    test_name = "Exact Match Query"
    try:
        query = "Samsung Galaxy S24 Ultra"
        results = recommender.get_recommendations(query, top_n=5)
        
        # Validate: got results
        assert len(results) > 0, "No recommendations returned"
        
        # Validate: query product is NOT in results
        result_names = results["product_name"].str.lower().tolist()
        assert query.lower() not in result_names, \
            "Query product should not appear in its own recommendations"
        
        # Validate: all similarity scores in [0, 1]
        assert results["similarity_score"].between(0, 1).all(), \
            "Similarity scores outside [0, 1]"
        
        # Validate: scores are sorted descending
        scores = results["similarity_score"].tolist()
        assert scores == sorted(scores, reverse=True), \
            "Results are not sorted by similarity score"
        
        top_score = scores[0]
        top_product = results.iloc[0]["product_name"]
        details = (
            f"Query: '{query}' → Top match: '{top_product}' "
            f"(score: {top_score:.4f})\n"
            f"          All scores: {[f'{s:.4f}' for s in scores]}"
        )
        return TestResult(test_name, True, details)
    
    except Exception as e:
        return TestResult(test_name, False, f"Error: {e}")


def test_partial_name_search(recommender: ContentBasedRecommender) -> TestResult:
    """Test 2: Partial name search functionality.
    
    Verifies that the find_product() method correctly returns products
    matching a partial name query.
    """
    test_name = "Partial Name Search"
    try:
        # Search for 'samsung' should return Samsung products
        matches = recommender.find_product("samsung")
        assert len(matches) > 0, "No matches found for 'samsung'"
        assert all("samsung" in m for m in matches), \
            "Not all matches contain 'samsung'"
        
        # Search for 'book' or a known book keyword
        book_matches = recommender.find_product("atomic habits")
        
        details = (
            f"'samsung' → {len(matches)} matches: {matches[:3]}...\n"
            f"          'atomic habits' → {len(book_matches)} matches: {book_matches[:3]}"
        )
        return TestResult(test_name, True, details)
    
    except Exception as e:
        return TestResult(test_name, False, f"Error: {e}")


def test_same_category_relevance(recommender: ContentBasedRecommender) -> TestResult:
    """Test 3: Same-category relevance check.
    
    Verifies that recommendations for a product are predominantly from
    the same category, demonstrating that the content-based approach
    captures category-level similarity.
    """
    test_name = "Same-Category Relevance"
    try:
        query = "boAt Rockerz 450"
        results = recommender.get_recommendations(query, top_n=5)
        
        # Get the query product's category
        query_category = recommender.df[
            recommender.df["product_name"].str.lower() == query.lower()
        ].iloc[0]["category"]
        
        # Count how many recommendations are from the same category
        same_category_count = (results["category"] == query_category).sum()
        relevance_ratio = same_category_count / len(results)
        
        # We expect at least 60% of recommendations to be same-category
        # (content-based filtering should strongly prefer same category)
        passed = relevance_ratio >= 0.6
        
        details = (
            f"Query: '{query}' (category: '{query_category}')\n"
            f"          Same-category ratio: {same_category_count}/{len(results)} "
            f"({relevance_ratio:.0%}) — threshold: 60%"
        )
        return TestResult(test_name, passed, details)
    
    except Exception as e:
        return TestResult(test_name, False, f"Error: {e}")


def test_cross_category_query(recommender: ContentBasedRecommender) -> TestResult:
    """Test 4: Cross-category query (Books vs Electronics).
    
    Verifies that a book query does NOT return electronics products
    as top recommendations, demonstrating that the model distinguishes
    between fundamentally different product domains.
    """
    test_name = "Cross-Category Separation"
    try:
        query = "Atomic Habits by James Clear"
        results = recommender.get_recommendations(query, top_n=5)
        
        # Books should NOT be recommended electronics primarily
        electronics_count = (results["category"] == "electronics").sum()
        
        # We expect fewer than 40% electronics in book recommendations
        passed = electronics_count < 2
        
        categories = results["category"].tolist()
        scores = results["similarity_score"].tolist()
        details = (
            f"Query: '{query}' (category: Books)\n"
            f"          Result categories: {categories}\n"
            f"          Electronics in results: {electronics_count}/5\n"
            f"          Scores: {[f'{s:.4f}' for s in scores]}"
        )
        return TestResult(test_name, passed, details)
    
    except Exception as e:
        return TestResult(test_name, False, f"Error: {e}")


def test_unknown_product(recommender: ContentBasedRecommender) -> TestResult:
    """Test 5: Unknown product — graceful error handling.
    
    Verifies that querying for a non-existent product raises a KeyError
    with a helpful message rather than crashing.
    """
    test_name = "Unknown Product Handling"
    try:
        try:
            recommender.get_recommendations("NonExistent Product XYZ 9999")
            return TestResult(test_name, False, "Should have raised KeyError")
        except KeyError as e:
            details = f"Correctly raised KeyError: {str(e)[:100]}"
            return TestResult(test_name, True, details)
    
    except Exception as e:
        return TestResult(test_name, False, f"Unexpected error type: {type(e).__name__}: {e}")


def test_empty_query(recommender: ContentBasedRecommender) -> TestResult:
    """Test 6: Empty query — input validation.
    
    Verifies that passing an empty string raises ValueError.
    """
    test_name = "Empty Query Handling"
    try:
        try:
            recommender.get_recommendations("")
            return TestResult(test_name, False, "Should have raised ValueError")
        except ValueError as e:
            details = f"Correctly raised ValueError: {str(e)[:100]}"
            return TestResult(test_name, True, details)
    
    except Exception as e:
        return TestResult(test_name, False, f"Unexpected error type: {type(e).__name__}: {e}")


def test_score_bounds(recommender: ContentBasedRecommender) -> TestResult:
    """Test 7: Similarity score bounds validation.
    
    Verifies that ALL entries in the similarity matrix are within [0, 1].
    This is a fundamental property of cosine similarity on TF-IDF vectors.
    """
    test_name = "Score Bounds [0, 1]"
    try:
        sim_matrix = recommender.similarity_matrix
        min_score = sim_matrix.min()
        max_score = sim_matrix.max()
        
        passed = (min_score >= 0.0) and (max_score <= 1.0 + 1e-6)
        
        details = (
            f"Min similarity: {min_score:.6f}, Max similarity: {max_score:.6f}\n"
            f"          All {sim_matrix.shape[0] * sim_matrix.shape[1]:,} entries within bounds: {passed}"
        )
        return TestResult(test_name, passed, details)
    
    except Exception as e:
        return TestResult(test_name, False, f"Error: {e}")


def test_self_exclusion(recommender: ContentBasedRecommender) -> TestResult:
    """Test 8: Self-exclusion validation.
    
    Verifies that no product ever recommends itself, across a sample
    of products from different categories.
    """
    test_name = "Self-Exclusion Check"
    try:
        # Sample products from different categories
        sample_queries = [
            "Samsung Galaxy S24 Ultra",
            "Nike Air Max 270",
            "Atomic Habits by James Clear",
            "Prestige Omega Deluxe Tawa",
            "Yonex Nanoray Light 18i",
        ]
        
        for query in sample_queries:
            results = recommender.get_recommendations(query, top_n=10)
            result_names = results["product_name"].str.lower().tolist()
            assert query.lower() not in result_names, \
                f"'{query}' appeared in its own recommendations!"
        
        details = f"Verified self-exclusion across {len(sample_queries)} products"
        return TestResult(test_name, True, details)
    
    except AssertionError as e:
        return TestResult(test_name, False, str(e))
    except Exception as e:
        return TestResult(test_name, False, f"Error: {e}")


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------
def run_evaluation() -> None:
    """Run all evaluation test cases and print a formatted report.
    
    This function:
        1. Loads and processes the dataset.
        2. Fits the recommendation model.
        3. Runs all test cases.
        4. Prints a formatted evaluation report with pass/fail results.
    """
    print("\n" + "=" * 70)
    print("  ApniDukaan — Recommendation Engine Evaluation Report")
    print("=" * 70 + "\n")
    
    # --- Setup ---
    print("Setting up evaluation pipeline...\n")
    df = process_pipeline()
    recommender = ContentBasedRecommender()
    recommender.fit(df)
    
    # --- Run all tests ---
    tests = [
        test_exact_match,
        test_partial_name_search,
        test_same_category_relevance,
        test_cross_category_query,
        test_unknown_product,
        test_empty_query,
        test_score_bounds,
        test_self_exclusion,
    ]
    
    print("-" * 70)
    print("  TEST RESULTS")
    print("-" * 70)
    
    results = []
    for test_fn in tests:
        result = test_fn(recommender)
        results.append(result)
        print(result)
        print()
    
    # --- Summary ---
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print("=" * 70)
    print(f"  SUMMARY: {passed}/{total} tests passed")
    if passed == total:
        print("  ✓ All tests passed! Recommendation engine is working correctly.")
    else:
        failed = [r.name for r in results if not r.passed]
        print(f"  ✗ Failed tests: {', '.join(failed)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()
