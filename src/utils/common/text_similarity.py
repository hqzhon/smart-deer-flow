# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Text similarity utilities for comparing text content.

This module provides common text similarity calculation functions
used across the codebase.
"""

import re
from typing import List
from collections import Counter


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Converts to lowercase, removes extra whitespace, and removes punctuation.

    Args:
        text: Input text to normalize.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Input text to tokenize.

    Returns:
        List of tokens (words).
    """
    normalized = normalize_text(text)
    return normalized.split()


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.

    Jaccard similarity = |intersection| / |union| of word sets.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity score between 0 and 1.
    """
    if not text1 or not text2:
        return 0.0

    words1 = set(tokenize(text1))
    words2 = set(tokenize(text2))

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union) if union else 0.0


def cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts.

    Uses word frequency vectors for comparison.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Similarity score between 0 and 1.
    """
    if not text1 or not text2:
        return 0.0

    words1 = tokenize(text1)
    words2 = tokenize(text2)

    if not words1 or not words2:
        return 0.0

    counter1 = Counter(words1)
    counter2 = Counter(words2)

    all_words = set(counter1.keys()).union(set(counter2.keys()))

    dot_product = sum(
        counter1.get(word, 0) * counter2.get(word, 0) for word in all_words
    )

    magnitude1 = sum(count**2 for count in counter1.values()) ** 0.5
    magnitude2 = sum(count**2 for count in counter2.values()) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def calculate_text_similarity(
    text1: str,
    text2: str,
    method: str = "jaccard",
) -> float:
    """
    Calculate text similarity using the specified method.

    Args:
        text1: First text.
        text2: Second text.
        method: Similarity method - "jaccard" or "cosine".

    Returns:
        Similarity score between 0 and 1.
    """
    if method == "cosine":
        return cosine_similarity(text1, text2)
    else:
        return jaccard_similarity(text1, text2)


def find_similar_texts(
    query: str,
    candidates: List[str],
    threshold: float = 0.5,
    method: str = "jaccard",
) -> List[tuple[int, float, str]]:
    """
    Find texts similar to a query from a list of candidates.

    Args:
        query: The query text to compare against.
        candidates: List of candidate texts.
        threshold: Minimum similarity threshold.
        method: Similarity method - "jaccard" or "cosine".

    Returns:
        List of tuples (index, similarity_score, candidate_text) for matches.
    """
    results = []
    for idx, candidate in enumerate(candidates):
        similarity = calculate_text_similarity(query, candidate, method)
        if similarity >= threshold:
            results.append((idx, similarity, candidate))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def calculate_overlap_ratio(text1: str, text2: str) -> float:
    """
    Calculate the overlap ratio between two texts.

    Returns the ratio of overlapping words to the length of the shorter text.

    Args:
        text1: First text.
        text2: Second text.

    Returns:
        Overlap ratio between 0 and 1.
    """
    if not text1 or not text2:
        return 0.0

    words1 = set(tokenize(text1))
    words2 = set(tokenize(text2))

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    min_length = min(len(words1), len(words2))

    return len(intersection) / min_length if min_length > 0 else 0.0
