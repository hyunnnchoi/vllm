# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
LTR (Least Time Remaining) Scheduler implementation.

This scheduler extends the base Scheduler to use the LTR scheduling policy,
which prioritizes requests with the least remaining time to completion.
"""

from __future__ import annotations

import copy
from typing import Optional

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager


class SchedulerLTR(Scheduler):
    """
    Scheduler that uses Least Time Remaining (LTR) policy.
    
    LTR scheduling prioritizes requests based on their estimated remaining
    execution time. Requests with less remaining time are scheduled first,
    which can help minimize average response time and improve throughput.
    
    This scheduler initializes the base Scheduler with FCFS policy to avoid
    issues during base initialization, then switches to LTR policy and
    recreates the waiting queue appropriately.
    
    Use this scheduler when:
    - You want to minimize average response time
    - Request completion time estimates are available
    - Shorter requests should be prioritized over longer ones
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Initialize the LTR scheduler.
        
        Args:
            vllm_config: Configuration for vLLM
            kv_cache_config: KV cache configuration
            structured_output_manager: Manager for structured outputs
            mm_registry: Multimodal registry
            include_finished_set: Whether to track finished request IDs
            log_stats: Whether to log statistics
        """
        # Avoid mutating the provided vllm_config object because it's shared/global;
        # create a shallow copy of the config and the scheduler_config so we can
        # pass a valid fcfs policy to the base Scheduler init without changing
        # global state or introducing races.
        # 
        # We use copy.copy (shallow copy) instead of deepcopy for efficiency,
        # as we only need to avoid mutating the top-level config objects.
        # The nested objects (model_config, cache_config, etc.) don't need
        # to be copied since we're not modifying them.
        patched_config = copy.copy(vllm_config)
        patched_sched_cfg = copy.copy(vllm_config.scheduler_config)
        # Set to "fcfs" for base initialization.
        # The base Scheduler expects a known policy string, so we use "fcfs"
        # which is the simplest and most predictable for initialization.
        patched_sched_cfg.policy = "fcfs"
        patched_config.scheduler_config = patched_sched_cfg

        # Initialize the base Scheduler with the patched config
        super().__init__(
            vllm_config=patched_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        # Now override to LTR and recreate the waiting queue for LTR policy
        # This ensures that all requests are scheduled according to LTR policy
        # (Least Time Remaining), where requests with less remaining time
        # are prioritized.
        self.policy = SchedulingPolicy.LTR
        self.waiting = create_request_queue(self.policy)
