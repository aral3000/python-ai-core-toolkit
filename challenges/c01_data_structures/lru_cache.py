"""
Challenge 01: O(1) LRU (Least Recently Used) Cache for AI Token & Embedding Memory
Author: M. Athallah Rafif Aldera

Task:
Implement an LRU Cache from scratch.
All operations `get` and `put` must run in O(1) average time complexity.
"""

from collections import OrderedDict
from typing import Any


class LRUCache:
    def __init__(self, capacity: int):
        """
        Initializes an LRU Cache with a positive integer capacity.
        Raises ValueError if capacity <= 0.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer greater than 0.")
        self.capacity = capacity
        self._cache: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Any:
        """
        Returns the value of the key if it exists in the cache, otherwise returns -1.
        Accessing the key marks it as the Most Recently Used.
        Complexity: O(1).
        """
        if key not in self._cache:
            return -1
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: Any, value: Any) -> None:
        """
        Updates the value of the key if key exists.
        Otherwise adds the key-value pair to the cache.
        If the number of keys exceeds capacity from this operation,
        evict the least recently used key.
        Complexity: O(1).
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        """Returns the current number of items stored in cache."""
        return len(self._cache)
