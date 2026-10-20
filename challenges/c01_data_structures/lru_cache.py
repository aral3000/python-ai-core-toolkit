"""
Challenge 01: O(1) LRU (Least Recently Used) Cache for AI Token & Embedding Memory
Author: M. Athallah Rafif Aldera

Task:
Implement an LRU Cache from scratch.
All operations `get` and `put` must run in O(1) average time complexity.
You can implement this using a Hash Map + Doubly Linked List, or an OrderedDict.
"""

from typing import Any, Optional


class Node:
    """Doubly Linked List Node (optional helper if building from scratch)."""
    def __init__(self, key: Any = None, value: Any = None):
        self.key = key
        self.value = value
        self.prev: Optional['Node'] = None
        self.next: Optional['Node'] = None


class LRUCache:
    def __init__(self, capacity: int):
        """
        Initializes an LRU Cache with a positive integer capacity.
        Raises ValueError if capacity <= 0.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer greater than 0.")
        self.capacity = capacity
        # TODO: Athallah - Initialize your data structures here (e.g. dict, head/tail dummy nodes)
        self._cache = {}

    def get(self, key: Any) -> Any:
        """
        Returns the value of the key if it exists in the cache, otherwise returns -1 (or None).
        Accessing the key marks it as the Most Recently Used.
        Complexity: O(1).
        """
        # TODO: Athallah - implement get logic
        raise NotImplementedError("TODO: Athallah - implement get(key)")

    def put(self, key: Any, value: Any) -> None:
        """
        Updates the value of the key if key exists.
        Otherwise adds the key-value pair to the cache.
        If the number of keys exceeds capacity from this operation,
        evict the least recently used key.
        Complexity: O(1).
        """
        # TODO: Athallah - implement put logic
        raise NotImplementedError("TODO: Athallah - implement put(key, value)")

    def __len__(self) -> int:
        """Returns the current number of items stored in cache."""
        return len(self._cache)
