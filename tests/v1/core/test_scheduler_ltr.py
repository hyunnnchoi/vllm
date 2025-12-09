# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for SchedulerLTR to ensure safe config handling.
"""

import pytest

from vllm.config import (CacheConfig, ModelConfig, SchedulerConfig, VllmConfig)
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler_ltr import SchedulerLTR
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.structured_output import StructuredOutputManager


def create_test_config(policy: str = "ltr") -> VllmConfig:
    """Create a test configuration for the scheduler."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="auto",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="float16",
        revision=None,
        skip_tokenizer_init=True,
    )
    
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=512,
        max_num_seqs=64,
        max_model_len=512,
        policy=policy,
    )
    
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space_bytes=0,
        cache_dtype="auto",
    )
    
    return VllmConfig(
        model_config=model_config,
        scheduler_config=scheduler_config,
        cache_config=cache_config,
    )


def create_test_kv_cache_config() -> KVCacheConfig:
    """Create a test KV cache configuration."""
    return KVCacheConfig(
        kv_cache_groups=[
            KVCacheGroupSpec(
                num_kv_heads=12,
                head_size=64,
                num_layers=12,
                num_blocks=1000,
                attention_spec=FullAttentionSpec(),
            )
        ]
    )


def test_scheduler_ltr_initialization():
    """Test that SchedulerLTR initializes correctly."""
    vllm_config = create_test_config(policy="ltr")
    kv_cache_config = create_test_kv_cache_config()
    structured_output_manager = StructuredOutputManager()
    
    scheduler = SchedulerLTR(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=structured_output_manager,
    )
    
    # Verify that the scheduler has the LTR policy set
    assert scheduler.policy == SchedulingPolicy.LTR
    
    # Verify that the waiting queue is created for LTR policy
    assert scheduler.waiting is not None
    

def test_scheduler_ltr_does_not_mutate_global_config():
    """Test that SchedulerLTR does not mutate the provided vllm_config."""
    original_policy = "ltr"
    vllm_config = create_test_config(policy=original_policy)
    kv_cache_config = create_test_kv_cache_config()
    structured_output_manager = StructuredOutputManager()
    
    # Store the original policy value
    original_scheduler_policy = vllm_config.scheduler_config.policy
    
    # Create the scheduler
    scheduler = SchedulerLTR(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=structured_output_manager,
    )
    
    # Verify that the original config's policy is unchanged
    assert vllm_config.scheduler_config.policy == original_scheduler_policy
    assert vllm_config.scheduler_config.policy == original_policy
    
    # Verify that the scheduler itself has LTR policy
    assert scheduler.policy == SchedulingPolicy.LTR


def test_scheduler_ltr_with_fcfs_config():
    """Test that SchedulerLTR works even when config has fcfs policy."""
    # Create a config with FCFS policy
    vllm_config = create_test_config(policy="fcfs")
    kv_cache_config = create_test_kv_cache_config()
    structured_output_manager = StructuredOutputManager()
    
    # Store the original policy
    original_policy = vllm_config.scheduler_config.policy
    assert original_policy == "fcfs"
    
    # Create the LTR scheduler
    scheduler = SchedulerLTR(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=structured_output_manager,
    )
    
    # Verify that the scheduler has LTR policy, not FCFS
    assert scheduler.policy == SchedulingPolicy.LTR
    
    # Verify that the original config is unchanged
    assert vllm_config.scheduler_config.policy == original_policy
    assert vllm_config.scheduler_config.policy == "fcfs"


def test_scheduler_ltr_config_isolation():
    """Test that multiple SchedulerLTR instances don't interfere."""
    vllm_config = create_test_config(policy="priority")
    kv_cache_config = create_test_kv_cache_config()
    structured_output_manager = StructuredOutputManager()
    
    # Create first scheduler
    scheduler1 = SchedulerLTR(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=structured_output_manager,
    )
    
    # Verify original config is unchanged
    assert vllm_config.scheduler_config.policy == "priority"
    
    # Create second scheduler with same config
    scheduler2 = SchedulerLTR(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(),
    )
    
    # Verify both schedulers have LTR policy
    assert scheduler1.policy == SchedulingPolicy.LTR
    assert scheduler2.policy == SchedulingPolicy.LTR
    
    # Verify original config is still unchanged
    assert vllm_config.scheduler_config.policy == "priority"
