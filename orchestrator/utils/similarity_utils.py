from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

__all__ = [
    'calculate_cosine_similarity',
    'get_similar_users',
    'filter_recommendations',
    'calculate_item_similarity',
    'get_similar_items',
    'calculate_cooccurrence_matrix'
]

def calculate_cosine_similarity(user_features: np.ndarray) -> np.ndarray:
    """
    Calculate raw cosine similarity matrix
    Input matrix shape: (n_users, n_features)
    Output matrix shape: (n_users, n_users)
    """
    return cosine_similarity(user_features)

def get_similar_users(similarity_matrix: np.ndarray, 
                     user_index: int,
                     exclude_self: bool = True) -> List[Dict]:
    """Get all similar users with their similarity scores
    No filtering, just raw similarities"""
    user_similarities = similarity_matrix[user_index]
    
    similar_users = []
    for idx, similarity in enumerate(user_similarities):
        if exclude_self and idx == user_index:
            continue
        similar_users.append({
            'index': idx,
            'similarity': float(similarity)
        })
    
    return similar_users

def filter_recommendations(candidate_products: List[str],
                         user_products: List[str]) -> List[str]:
    """Remove products already used by target user"""
    return [product for product in candidate_products 
            if product not in set(user_products)]

def calculate_item_similarity(item_features: np.ndarray) -> np.ndarray:
    """Calculate similarity matrix between items"""
    return cosine_similarity(item_features)

def get_similar_items(similarity_matrix: np.ndarray,
                     item_index: int,
                     exclude_self: bool = True) -> List[Dict]:
    """Get similar items based on similarity matrix"""
    similarities = similarity_matrix[item_index]
    
    similar_items = []
    for idx, similarity in enumerate(similarities):
        if exclude_self and idx == item_index:
            continue
        similar_items.append({
            'index': idx,
            'similarity': float(similarity)
        })
    
    # Sort by similarity
    similar_items.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similar_items

def calculate_cooccurrence_matrix(df: pd.DataFrame, 
                                user_col: str,
                                item_col: str) -> np.ndarray:
    """Calculate item co-occurrence matrix"""
    items = df[item_col].unique()
    item_indices = {item: idx for idx, item in enumerate(items)}
    n_items = len(items)
    
    cooccurrence = np.zeros((n_items, n_items))
    
    for user in df[user_col].unique():
        user_items = df[df[user_col] == user][item_col].unique()
        for i in user_items:
            for j in user_items:
                if i != j:
                    idx_i = item_indices[i]
                    idx_j = item_indices[j]
                    cooccurrence[idx_i, idx_j] += 1
    
    return cooccurrence 