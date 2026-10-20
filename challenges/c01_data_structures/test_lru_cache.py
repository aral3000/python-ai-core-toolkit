"""
Test suite for Challenge 01: LRUCache
Run with: pytest challenges/01_data_structures/test_lru_cache.py
"""

import pytest
from challenges.c01_data_structures.lru_cache import LRUCache


def test_basic_put_get():
    cache = LRUCache(2)
    cache.put("token_a", [0.1, 0.2])
    cache.put("token_b", [0.3, 0.4])
    
    assert cache.get("token_a") == [0.1, 0.2]
    assert cache.get("token_b") == [0.3, 0.4]


def test_eviction_policy():
    cache = LRUCache(2)
    cache.put(1, "one")
    cache.put(2, "two")
    # Evicts key 1 because capacity is 2
    cache.put(3, "three")
    
    assert cache.get(1) in [-1, None]
    assert cache.get(2) == "two"
    assert cache.get(3) == "three"


def test_recently_used_order():
    cache = LRUCache(2)
    cache.put(1, "one")
    cache.put(2, "two")
    # Access key 1 to make it most recently used
    _ = cache.get(1)
    # Adding key 3 should now evict key 2 (least recently used)
    cache.put(3, "three")
    
    assert cache.get(2) in [-1, None]
    assert cache.get(1) == "one"
    assert cache.get(3) == "three"


def test_update_existing_key():
    cache = LRUCache(2)
    cache.put("alpha", 100)
    cache.put("alpha", 200)
    
    assert cache.get("alpha") == 200
    assert len(cache) == 1


def test_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache(0)
    with pytest.raises(ValueError):
        LRUCache(-5)
