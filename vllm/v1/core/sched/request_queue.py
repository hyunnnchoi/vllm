# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from vllm.v1.request import Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    # [NOTE, hyunnnchoi, 2025.12.01] ELIS ISRTF scheduling policy
    # Based on: https://arxiv.org/abs/2505.09142
    ISRTF = "isrtf"
    # LTR (Least Time Remaining) scheduling policy
    LTR = "ltr"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))


# [NOTE, hyunnnchoi, 2025.12.01] ELIS ISRTF Request Queue
# Based on: https://arxiv.org/abs/2505.09142
class ISRTFRequestQueue(RequestQueue):
    """
    Iterative Shortest Remaining Time First (ISRTF) queue for ELIS scheduling.
    
    Requests with smaller predicted_remaining_tokens are processed first.
    If multiple requests have the same predicted tokens, the one with the 
    earlier arrival_time is processed first.
    
    This implements the ISRTF scheduling from the ELIS paper:
    - Requests are prioritized by predicted remaining output tokens
    - Predictions are updated every 50 tokens (handled by scheduler)
    - Lower remaining tokens = higher priority
    """

    def __init__(self) -> None:
        # Heap entries: (predicted_remaining_tokens, arrival_time, request)
        self._heap: list[tuple[float, float, Request]] = []
        # Track requests for efficient removal
        self._request_set: set[str] = set()

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to ISRTF policy."""
        if request.request_id in self._request_set:
            return  # Already in queue
        heapq.heappush(
            self._heap,
            (request.predicted_remaining_tokens, request.arrival_time, request)
        )
        self._request_set.add(request.request_id)

    def pop_request(self) -> Request:
        """Pop a request with shortest predicted remaining tokens."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        while self._heap:
            _, _, request = heapq.heappop(self._heap)
            if request.request_id in self._request_set:
                self._request_set.remove(request.request_id)
                return request
        raise IndexError("pop from empty heap")

    def peek_request(self) -> Request:
        """Peek at the request with shortest predicted remaining tokens."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        # Skip removed requests
        while self._heap and self._heap[0][2].request_id not in self._request_set:
            heapq.heappop(self._heap)
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to ISRTF policy.
        
        Note: In ISRTF, there is no concept of prepending. Requests are 
        ordered by (predicted_remaining_tokens, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to ISRTF policy."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._request_set.discard(request.request_id)
        # Lazy removal - actual heap entry will be skipped in pop/peek

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        for request in requests:
            self._request_set.discard(request.request_id)

    def update_request_priority(self, request: Request) -> None:
        """
        Update request priority after prediction update.
        
        This is called when the predicted_remaining_tokens is updated
        (every 50 tokens as per ELIS paper).
        """
        if request.request_id not in self._request_set:
            return
        # Re-add with updated priority (old entry will be skipped via lazy removal)
        heapq.heappush(
            self._heap,
            (request.predicted_remaining_tokens, request.arrival_time, request)
        )

    def _cleanup_heap(self) -> None:
        """Remove stale entries from the heap."""
        self._heap = [
            (p, t, r) for p, t, r in self._heap 
            if r.request_id in self._request_set
        ]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._request_set)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._request_set)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to ISRTF policy."""
        # Create sorted list by (predicted_remaining_tokens, arrival_time)
        valid_requests = [
            (p, t, r) for p, t, r in self._heap 
            if r.request_id in self._request_set
        ]
        valid_requests.sort(key=lambda x: (x[0], x[1]))
        seen = set()
        for _, _, request in valid_requests:
            if request.request_id not in seen:
                seen.add(request.request_id)
                yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse ISRTF order."""
        return reversed(list(self))


class LTRRequestQueue(RequestQueue):
    """
    Least Time Remaining (LTR) queue for scheduling.
    
    Requests with smaller estimated remaining time are processed first.
    This is similar to ISRTF but focuses on time-based scheduling.
    """

    def __init__(self) -> None:
        # Heap entries: (estimated_remaining_time, arrival_time, request)
        self._heap: list[tuple[float, float, Request]] = []
        # Track requests for efficient removal
        self._request_set: set[str] = set()

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to LTR policy."""
        if request.request_id in self._request_set:
            return  # Already in queue
        # Use predicted_remaining_tokens as a proxy for remaining time
        remaining_time = request.predicted_remaining_tokens
        heapq.heappush(
            self._heap,
            (remaining_time, request.arrival_time, request)
        )
        self._request_set.add(request.request_id)

    def pop_request(self) -> Request:
        """Pop a request with least remaining time."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        while self._heap:
            _, _, request = heapq.heappop(self._heap)
            if request.request_id in self._request_set:
                self._request_set.remove(request.request_id)
                return request
        raise IndexError("pop from empty heap")

    def peek_request(self) -> Request:
        """Peek at the request with least remaining time."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        # Skip removed requests
        while self._heap and self._heap[0][2].request_id not in self._request_set:
            heapq.heappop(self._heap)
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request back to the queue according to LTR policy.
        
        Note: In LTR (priority-based queue), requests cannot truly be 
        "prepended" to the front. Instead, they are added according to 
        their remaining time priority. This ensures LTR ordering is maintained.
        This method exists to satisfy the RequestQueue interface."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add requests back to the queue according to LTR policy.
        
        Note: In LTR (priority-based queue), requests cannot truly be 
        "prepended" to the front. Instead, they are added according to 
        their remaining time priority. This ensures LTR ordering is maintained.
        This method exists to satisfy the RequestQueue interface."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._request_set.discard(request.request_id)
        # Lazy removal - actual heap entry will be skipped in pop/peek

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        for request in requests:
            self._request_set.discard(request.request_id)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._request_set)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._request_set)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to LTR policy.
        
        Note: Duplicates are prevented because add_request() checks
        self._request_set before adding. Only requests currently in
        the queue are yielded."""
        # Create sorted list by (remaining_time, arrival_time)
        # Only include requests that are still in the queue (not removed)
        valid_requests = [
            (p, t, r) for p, t, r in self._heap 
            if r.request_id in self._request_set
        ]
        valid_requests.sort(key=lambda x: (x[0], x[1]))
        for _, _, request in valid_requests:
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse LTR order."""
        return reversed(list(self))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    # [NOTE, hyunnnchoi, 2025.12.01] ELIS ISRTF scheduling
    elif policy == SchedulingPolicy.ISRTF:
        return ISRTFRequestQueue()
    elif policy == SchedulingPolicy.LTR:
        return LTRRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
