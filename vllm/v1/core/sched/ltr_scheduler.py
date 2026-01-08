# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import csv
import itertools
import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)

# [NOTE, hjhoon03, 2025.12.17]: scheduler for learning-to-rank policy
class LTRScheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        self.dcp_world_size = \
            vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): The scheduler’s block_size must be multiplied
        # by dcp_world_size, since block hashes are computed on the
        # original full token sequence at a granularity of
        # original_block_size × dcp_world_size.
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        self.scheduler_config.policy = "ltr"
        if self.scheduler_config.policy == "ltr":
            self.policy = SchedulingPolicy.LTR
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        
        # [NOTE, hjhoon03, 2025-12-19] Initialize LTR request queue and predictor model
        logger.info("[LTR] Initializing predictor from %s", self.scheduler_config.predictor_model_path)

        # Priority queues for requests.
        self.waiting = create_request_queue(
            self.policy, 
            target_model=self.vllm_config.model_config.model, 
            predictor_model_name=self.scheduler_config.predictor_model_name, 
            predictor_model_path=self.scheduler_config.predictor_model_path
        )
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        # [NOTE, hyunnnchoi, 2025.11.03] Initialize tokenizer for logging purposes if not skipped
        tokenizer = None
        if not self.vllm_config.model_config.skip_tokenizer_init:
            try:
                tokenizer = init_tokenizer_from_configs(
                    model_config=self.vllm_config.model_config)
            except Exception:
                # If tokenizer initialization fails, continue without it
                logger.warning("Failed to initialize tokenizer for prefix cache logging")
                tokenizer = None
        
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            tokenizer=tokenizer,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        # CSV logging setup
        # 환경변수로 지정하거나, 기본 경로 사용
        csv_log_enabled = os.environ.get("VLLM_SCHEDULER_CSV_LOG", "0") == "1"
        self.csv_log_dir = os.environ.get(
            "VLLM_SCHEDULER_CSV_LOG_DIR",
            "/tmp/vllm_scheduler_logs" if csv_log_enabled else None
        )
        self.batch_csv_file = None
        self.request_csv_file = None
        self.batch_csv_writer = None
        self.request_csv_writer = None
        self.iteration_counter = 0
        self.csv_rows_written = 0  # Track rows written to current file
        self.csv_max_rows = int(os.environ.get("VLLM_SCHEDULER_CSV_MAX_ROWS", "1000000"))  # Default 1M rows per file
        self.csv_file_counter = 0  # Counter for file rotation
        
        if self.csv_log_dir:
            os.makedirs(self.csv_log_dir, exist_ok=True)
            self._open_csv_files()
            
            logger.info(f"CSV logging enabled. Logs will be saved to {self.csv_log_dir}")

        # For learning to rank scheduling
        self.starvation_threshold: int = 256
        self.priority_quantum: int = 32

    def _open_csv_files(self) -> None:
        """Open CSV files for logging. Creates new files or rotates existing ones."""
        if not self.csv_log_dir:
            return
        
        # Close existing files if open
        self._close_csv_files()
        
        # Batch log CSV - iteration마다 한 행
        batch_csv_path = os.path.join(
            self.csv_log_dir, 
            f"batch_log_dp{self.parallel_config.data_parallel_rank}_part{self.csv_file_counter}.csv"
        )
        self.batch_csv_file = open(batch_csv_path, 'w', newline='', buffering=1)  # Line buffering
        self.batch_csv_writer = csv.writer(self.batch_csv_file)
        self.batch_csv_writer.writerow([
            'iteration', 'timestamp', 'iteration_time_ms',
            'new_reqs', 'running_reqs', 'resumed_reqs', 'total_reqs',
            'total_tokens', 'waiting_reqs', 'preempted_reqs',
            'prefill_reqs', 'decode_reqs', 'prefill_ratio'
        ])
        self.batch_csv_file.flush()
        
        # Request log CSV - iteration마다 각 request별로 한 행씩
        request_csv_path = os.path.join(
            self.csv_log_dir,
            f"request_log_dp{self.parallel_config.data_parallel_rank}_part{self.csv_file_counter}.csv"
        )
        self.request_csv_file = open(request_csv_path, 'w', newline='', buffering=1)  # Line buffering
        self.request_csv_writer = csv.writer(self.request_csv_file)
        self.request_csv_writer.writerow([
            'iteration', 'timestamp', 'request_id', 'phase',
            'num_prompt_tokens', 'num_output_tokens', 'num_computed_tokens',
            'chunk_start', 'chunk_end', 'num_scheduled_tokens'
        ])
        self.request_csv_file.flush()
        
        self.csv_rows_written = 0
        logger.info(f"Opened CSV files part {self.csv_file_counter}: {batch_csv_path}, {request_csv_path}")

    def _close_csv_files(self) -> None:
        """Close CSV files if they are open."""
        if self.batch_csv_file is not None:
            try:
                self.batch_csv_file.flush()
                self.batch_csv_file.close()
            except Exception as e:
                logger.warning(f"Error closing batch CSV file: {e}")
            finally:
                self.batch_csv_file = None
                self.batch_csv_writer = None
        
        if self.request_csv_file is not None:
            try:
                self.request_csv_file.flush()
                self.request_csv_file.close()
            except Exception as e:
                logger.warning(f"Error closing request CSV file: {e}")
            finally:
                self.request_csv_file = None
                self.request_csv_writer = None

    def _check_and_rotate_csv_files(self) -> None:
        """Check if CSV files need to be rotated based on row count."""
        if not self.csv_log_dir:
            return
        
        # Rotate when we reach the max rows limit
        # Note: This tracks request rows (more accurate) but applies to both files
        if self.csv_rows_written >= self.csv_max_rows:
            logger.info(
                f"CSV file rotation: {self.csv_rows_written} rows written, "
                f"rotating to part {self.csv_file_counter + 1}"
            )
            self.csv_file_counter += 1
            self._open_csv_files()

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # score에 따라 requests 정렬
        # promote: starvation 방지용
        # waiting, running request 구별 없이 정렬
        sorted_requests = sorted(self.waiting.requests + self.running, key=lambda req: (req.promote, req.score), reverse=True)

        # define for ltr scheduling
        prev_run = filter(lambda req: req.status == RequestStatus.RUNNING, sorted_requests)
        new_run = []
        can_schedule = True

        # batch requests
        req_index = 0
        while req_index < len(sorted_requests) and token_budget > 0 and len(new_run) < self.max_num_running_reqs and can_schedule:
            request = sorted_requests[req_index]

            # KVTransfer: skip request if still waiting for remote kvs.
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                is_ready = self._update_waiting_for_remote_kv(request)
                if is_ready:
                    request.status = RequestStatus.WAITING
                else:
                    logger.debug(
                        "%s is still in WAITING_FOR_REMOTE_KVS state.",
                        request.request_id)
                    req_index += 1
                continue

            # Skip request if the structured output request is still waiting
            # for FSM compilation.
            elif request.status == RequestStatus.WAITING_FOR_FSM:
                structured_output_req = request.structured_output_request
                if structured_output_req and structured_output_req.grammar:
                    request.status = RequestStatus.WAITING
                else:
                    req_index += 1
                continue

            elif request.status == RequestStatus.RUNNING:
                num_new_tokens = (request.num_tokens_with_spec +
                                request.num_output_placeholders -
                                request.num_computed_tokens)
                if (0 < self.scheduler_config.long_prefill_token_threshold <
                        num_new_tokens):
                    num_new_tokens = (
                        self.scheduler_config.long_prefill_token_threshold)
                num_new_tokens = min(num_new_tokens, token_budget)

                # Make sure the input position does not exceed the max model len.
                # This is necessary when using spec decoding.
                num_new_tokens = min(
                    num_new_tokens,
                    self.max_model_len - 1 - request.num_computed_tokens)

                if num_new_tokens == 0:
                    # The request cannot be scheduled because one of the following
                    # reasons:
                    # 1. No new tokens to schedule. This may happen when
                    #    (1) PP>1 and we have already scheduled all prompt tokens
                    #    but they are not finished yet.
                    #    (2) Async scheduling and the request has reached to either
                    #    its max_total_tokens or max_model_len.
                    # 2. The encoder budget is exhausted.
                    # 3. The encoder cache is exhausted.
                    # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                    # we do not strictly follow the FCFS scheduling policy and
                    # allow the lower-priority requests to be scheduled.
                    new_run.append(request)
                    req_index += 1
                    continue

            elif request.status in [RequestStatus.WAITING, RequestStatus.PREEMPTED]:
                num_external_computed_tokens = 0
                load_kv_async = False

                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))
                        
                        if num_external_computed_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            req_index += 1
                            continue

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        req_index += 1
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)
                
                num_encoder_tokens = 0
            else:
                raise RuntimeError(
                    f"Invalid request status: {request.status}")

            # allocate KV blocks
            while True:
                if request.status in [RequestStatus.WAITING, RequestStatus.PREEMPTED]:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens + num_external_computed_tokens,
                        num_new_local_computed_tokens,
                        new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        delay_cache_blocks=load_kv_async,
                        num_encoder_tokens=num_encoder_tokens,
                    )
                elif request.status == RequestStatus.RUNNING:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")
                
                if new_blocks is None:
                    if not prev_run:
                        can_schedule = False
                        break

                    preempted_req = prev_run.pop()
                        
                    self.kv_cache_manager.free(preempted_req)
                    self.encoder_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    break
            assert new_blocks is not None

            if request.status in [RequestStatus.WAITING, RequestStatus.PREEMPTED]:
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                    
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens

            elif request.status == RequestStatus.RUNNING:
                req_to_new_blocks[request.request_id] = new_blocks
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens

            else:
                raise RuntimeError(
                    f"Invalid request status: {request.status}")

            # starvation
            request.starvation = 0
            if request.promote:
                if request.quantum is None:
                    request.quantum = self.priority_quantum
                
                request.quantum -= 1

                if request.quantum == 0:
                    request.promote = False
                    request.quantum = None

            new_run.append(request)
            if request.status == RequestStatus.RUNNING:
                scheduled_running_reqs.append(request)
            elif request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(
                    f"Invalid request status: {request.status}")
            request.status = RequestStatus.RUNNING
            req_index += 1

        self.running = new_run
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))
        
        self.waiting.requests = []
        for request in sorted_requests:
            if request.status == RequestStatus.RUNNING:
                if request in self.running:
                    continue
                else:
                    self.kv_cache_manager.free(request)
                    self.encoder_cache_manager.free(request)
                    request.status = RequestStatus.PREEMPTED
                    request.num_computed_tokens = 0
                    if self.log_stats:
                        request.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)
                    preempted_reqs.append(request)

            request.starvation += 1
            if request.starvation == self.starvation_threshold:
                request.promote = True

            self.waiting.requests.append(request)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids())
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_blocks,
        )
        scheduled_requests = (scheduled_new_reqs + scheduled_running_reqs +
                              scheduled_resumed_reqs)
        structured_output_request_ids, grammar_bitmask = (
            self.get_grammar_bitmask(scheduled_requests,
                                     scheduled_spec_decode_tokens))
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.
            get_freed_mm_hashes(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # NOTE(hyunnnchoi,2025-10-30): Log batch scheduling details for debugging
        if logger.isEnabledFor(logging.INFO):
            # 1. Iteration 시간 계산
            iteration_time_ms = (time.monotonic() - scheduled_timestamp) * 1000
            
            scheduled_req_ids = (
                [r.request_id for r in scheduled_new_reqs] +
                [r.request_id for r in scheduled_running_reqs] +
                [r.request_id for r in scheduled_resumed_reqs]
            )
            
            # 2. Running request들의 토큰 인덱스 정보
            all_scheduled_reqs = scheduled_new_reqs + scheduled_running_reqs + scheduled_resumed_reqs
            running_req_token_info = {}
            prefill_count = 0
            decode_count = 0
            
            for req in all_scheduled_reqs:
                is_prefill = req.num_computed_tokens < req.num_prompt_tokens
                if is_prefill:
                    prefill_count += 1
                else:
                    decode_count += 1
                
                # Prefill phase 분류
                # scheduled_new_reqs에 포함된 request는 첫 번째 스케줄링
                is_first_prefill = req in scheduled_new_reqs and is_prefill
                
                if is_prefill and is_first_prefill and req.num_computed_tokens > 0:
                    # 첫 prefill이지만 prefix cache로 인해 일부 토큰이 이미 계산됨
                    chunk_start = req.num_computed_tokens
                    chunk_end = req.num_computed_tokens + num_scheduled_tokens.get(req.request_id, 0)
                    running_req_token_info[req.request_id] = {
                        'computed': req.num_computed_tokens,
                        'input_tokens': req.num_prompt_tokens,
                        'output_tokens': req.num_output_tokens,
                        'phase': f'prefill_first_cached[{chunk_start}:{chunk_end}/{req.num_prompt_tokens}]'
                    }
                elif is_prefill and req.num_computed_tokens > 0:
                    # Chunked prefill 중 (이미 처리 중인 request)
                    chunk_start = req.num_computed_tokens
                    chunk_end = req.num_computed_tokens + num_scheduled_tokens.get(req.request_id, 0)
                    running_req_token_info[req.request_id] = {
                        'computed': req.num_computed_tokens,
                        'input_tokens': req.num_prompt_tokens,
                        'output_tokens': req.num_output_tokens,
                        'phase': f'prefill_chunk[{chunk_start}:{chunk_end}/{req.num_prompt_tokens}]'
                    }
                elif is_prefill:
                    # 첫 prefill (prefix cache 없음)
                    chunk_end = num_scheduled_tokens.get(req.request_id, 0)
                    running_req_token_info[req.request_id] = {
                        'computed': req.num_computed_tokens,
                        'input_tokens': req.num_prompt_tokens,
                        'output_tokens': req.num_output_tokens,
                        'phase': f'prefill_first[0:{chunk_end}/{req.num_prompt_tokens}]'
                    }
                else:
                    # Decode phase
                    running_req_token_info[req.request_id] = {
                        'computed': req.num_computed_tokens,
                        'input_tokens': req.num_prompt_tokens,
                        'output_tokens': req.num_output_tokens,
                        'phase': f'decode[token_{req.num_output_tokens + 1}]'
                    }
            
            # 3. 새로 들어온 request의 input token 정보
            new_req_info = [(r.request_id, r.num_prompt_tokens) for r in scheduled_new_reqs]
            
            # 4. Prefill/decode 비율
            total_phase_reqs = prefill_count + decode_count
            prefill_ratio = (prefill_count / total_phase_reqs * 100) if total_phase_reqs > 0 else 0
            
            # 5. Preempted request 상세 정보
            preempted_req_ids = [r.request_id for r in preempted_reqs]
            
            # 상세 로그 출력
            logger.info(
                f"Scheduled batch [iter_time={iteration_time_ms:.2f}ms]: "
                f"new_reqs={len(scheduled_new_reqs)} {[r.request_id for r in scheduled_new_reqs]}, "
                f"running_reqs={len(scheduled_running_reqs)} {[r.request_id for r in scheduled_running_reqs]}, "
                f"resumed_reqs={len(scheduled_resumed_reqs)} {[r.request_id for r in scheduled_resumed_reqs]}, "
                f"total_reqs={len(scheduled_req_ids)}, "
                f"total_tokens={total_num_scheduled_tokens}, "
                f"waiting_reqs={len(self.waiting)}, "
                f"preempted_reqs={len(preempted_reqs)} {preempted_req_ids}, "
                f"prefill_reqs={prefill_count}, "
                f"decode_reqs={decode_count}, "
                f"prefill_ratio={prefill_ratio:.1f}%"
            )
            
            # 새 request input token 정보
            if new_req_info:
                logger.info(f"  New requests input tokens: {new_req_info}")
            
            # 각 request의 상세 정보
            if running_req_token_info:
                logger.info(f"  Request details:")
                for req_id, info in running_req_token_info.items():
                    logger.info(
                        f"    {req_id}: {info['phase']}, "
                        f"input={info['input_tokens']}, "
                        f"output={info['output_tokens']}, "
                        f"computed={info['computed']}"
                    )
            
            # CSV 로깅 (환경변수로 활성화된 경우)
            if self.csv_log_dir and self.batch_csv_writer and self.request_csv_writer:
                # Check if file rotation is needed before writing
                self._check_and_rotate_csv_files()
                
                current_timestamp = time.time()
                
                # Batch log - iteration당 한 행
                self.batch_csv_writer.writerow([
                    self.iteration_counter,
                    current_timestamp,
                    f"{iteration_time_ms:.2f}",
                    len(scheduled_new_reqs),
                    len(scheduled_running_reqs),
                    len(scheduled_resumed_reqs),
                    len(scheduled_req_ids),
                    total_num_scheduled_tokens,
                    len(self.waiting),
                    len(preempted_reqs),
                    prefill_count,
                    decode_count,
                    f"{prefill_ratio:.1f}"
                ])
                self.batch_csv_file.flush()
                
                # Request log - 각 request별로 한 행씩
                request_rows_written = 0
                for req in all_scheduled_reqs:
                    info = running_req_token_info[req.request_id]
                    
                    # chunk_start, chunk_end 추출
                    is_prefill = req.num_computed_tokens < req.num_prompt_tokens
                    is_first_prefill = req in scheduled_new_reqs and is_prefill
                    
                    if is_prefill and is_first_prefill and req.num_computed_tokens > 0:
                        # 첫 prefill이지만 prefix cache로 인해 일부 토큰이 이미 계산됨
                        chunk_start = req.num_computed_tokens
                        chunk_end = req.num_computed_tokens + num_scheduled_tokens.get(req.request_id, 0)
                    elif is_prefill and req.num_computed_tokens > 0:
                        # Chunked prefill
                        chunk_start = req.num_computed_tokens
                        chunk_end = req.num_computed_tokens + num_scheduled_tokens.get(req.request_id, 0)
                    elif is_prefill:
                        # First prefill (prefix cache 없음)
                        chunk_start = 0
                        chunk_end = num_scheduled_tokens.get(req.request_id, 0)
                    else:
                        # Decode
                        chunk_start = -1
                        chunk_end = -1
                    
                    # phase는 prefill_first/prefill_first_cached/prefill_chunk/decode 중 하나
                    if is_prefill and is_first_prefill and req.num_computed_tokens > 0:
                        phase = "prefill_first_cached"
                    elif is_prefill and req.num_computed_tokens > 0:
                        phase = "prefill_chunk"
                    elif is_prefill:
                        phase = "prefill_first"
                    else:
                        phase = "decode"
                    
                    self.request_csv_writer.writerow([
                        self.iteration_counter,
                        current_timestamp,
                        req.request_id,
                        phase,
                        info['input_tokens'],
                        info['output_tokens'],
                        info['computed'],
                        chunk_start,
                        chunk_end,
                        num_scheduled_tokens.get(req.request_id, 0)
                    ])
                    request_rows_written += 1
                self.request_csv_file.flush()
                
                # Increment row counter (track both batch and request rows)
                # Use request rows as the primary counter since it's more accurate
                self.csv_rows_written += request_rows_written
                self.iteration_counter += 1

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[Optional[tuple[list[int], ...]]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = (num_scheduled_tokens[req_id] -
                          len(spec_decode_tokens.get(req_id, ())))
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_tokens_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used.")
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_features[i].identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(
                        request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if not self.encoder_cache_manager.can_allocate(
                    request, i, encoder_compute_budget,
                    num_tokens_to_schedule):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            num_tokens_to_schedule += num_encoder_tokens
            encoder_compute_budget -= num_encoder_tokens
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
        )

    def get_grammar_bitmask(
        self,
        requests: list[Request],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to its index in the batch.
        # This will help us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}
        for i, req in enumerate(requests):
            if req.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[req.request_id] = i

        if not structured_output_request_ids:
            bitmask = None
        else:
            bitmask = self.structured_output_manager.grammar_bitmask(
                self.requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )
        return structured_output_request_ids, bitmask

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # NOTE(hyunnnchoi,2025-10-30): Record iteration timestamp for decode step timing
        iteration_timestamp_ms = time.time() * 1000.0
        
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None
        kv_connector_stats = (kv_connector_output.kv_connector_stats
                              if kv_connector_output else None)

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                request.num_computed_tokens -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)
                
                # NOTE(hyunnnchoi,2025-10-30): Record decode step timing for each token
                for token_id in new_token_ids:
                    num_output_tokens = len(request.output_token_ids)
                    request.decode_step_timings.append(
                        (token_id, iteration_timestamp_ms, num_output_tokens)
                    )
                    # Record first token timestamp
                    if request.first_token_timestamp is None:
                        request.first_token_timestamp = iteration_timestamp_ms

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                # NOTE(hyunnnchoi,2025-10-30): Save decode step timings to file
                self._save_decode_timings(request)
                
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # checked above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    ))
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats,
                                     kv_connector_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
                draft_token_ids.req_ids,
                draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if not spec_token_ids:
                # NOTE(woosuk): request.spec_token_ids should be updated.
                request.spec_token_ids.clear()
            elif self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids)
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    # waiting queue에 request 추가.
    # score prediction 등은 waiting queue에서 진행
    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _save_decode_timings(self, request: Request) -> None:
        """Save decode step timings to a JSON file for analysis.
        
        NOTE(hyunnnchoi,2025-10-30): This saves detailed per-token timing
        information for debugging and performance analysis.
        """
        if not request.decode_step_timings:
            return
        
        import json
        import os
        from datetime import datetime
        
        # Get output directory from environment variable or use default
        # Set VLLM_DECODE_TIMINGS_DIR to match your benchmark log location
        output_dir = os.environ.get(
            "VLLM_DECODE_TIMINGS_DIR", 
            os.path.join(os.getcwd(), "decode_timings")
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert arrival_time (Unix timestamp in seconds) to human-readable format
        arrival_time_unix = request.arrival_time
        arrival_time_str = datetime.fromtimestamp(arrival_time_unix).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        
        # Prepare data
        timings_data = {
            "request_id": request.request_id,
            "arrival_time_unix": arrival_time_unix,
            "arrival_time_str": arrival_time_str,
            "arrival_time_ms": arrival_time_unix * 1000.0,
            "first_token_timestamp_ms": request.first_token_timestamp,
            "num_prompt_tokens": request.num_prompt_tokens,
            "num_output_tokens": len(request.output_token_ids),
            "decode_steps": [
                {
                    "token_id": token_id,
                    "timestamp_ms": timestamp_ms,
                    "output_token_index": token_idx,
                    "time_since_arrival_ms": timestamp_ms - arrival_time_unix * 1000.0,
                }
                for token_id, timestamp_ms, token_idx in request.decode_step_timings
            ]
        }
        
        # Calculate statistics
        if len(request.decode_step_timings) > 1:
            timings = [t[1] for t in request.decode_step_timings]
            time_to_first_token_ms = (
                request.first_token_timestamp - request.arrival_time * 1000.0
                if request.first_token_timestamp else None
            )
            inter_token_latencies = [
                timings[i] - timings[i-1] 
                for i in range(1, len(timings))
            ]
            
            total_time_ms = timings[-1] - arrival_time_unix * 1000.0
            
            timings_data["statistics"] = {
                "time_to_first_token_ms": time_to_first_token_ms,
                "total_generation_time_ms": total_time_ms,
                "inter_token_latencies_ms": inter_token_latencies,
                "avg_inter_token_latency_ms": (
                    sum(inter_token_latencies) / len(inter_token_latencies)
                    if inter_token_latencies else None
                ),
                "min_inter_token_latency_ms": (
                    min(inter_token_latencies) if inter_token_latencies else None
                ),
                "max_inter_token_latency_ms": (
                    max(inter_token_latencies) if inter_token_latencies else None
                ),
                "throughput_tokens_per_sec": (
                    len(request.output_token_ids) / (total_time_ms / 1000.0)
                    if total_time_ms > 0 else None
                ),
            }
        
        # Save to file
        filename = os.path.join(
            output_dir, 
            f"decode_timing_{request.request_id}.json"
        )
        with open(filename, "w") as f:
            json.dump(timings_data, f, indent=2)
        
        logger.info(
            f"Saved decode timings for request {request.request_id} "
            f"({len(request.decode_step_timings)} tokens) to {filename}"
        )

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
        kv_connector_stats: Optional[KVConnectorStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(num_running_reqs=len(self.running),
                              num_waiting_reqs=len(self.waiting),
                              kv_cache_usage=self.kv_cache_manager.usage,
                              prefix_cache_stats=prefix_cache_stats,
                              spec_decoding_stats=spec_decoding_stats,
                              num_corrupted_reqs=sum(req.is_output_corrupted
                                                     for req in self.running),
                              kv_connector_stats=kv_connector_stats.data
                              if kv_connector_stats else None)

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()
        # Close CSV files
        self._close_csv_files()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less than one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in (kv_connector_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (kv_connector_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            if req_id not in self.requests:
                logger.warning(
                    "Got finished sending KV transfer for request %s,"
                    "but the request is already freed.", req_id)
            else:
                self._free_blocks(self.requests[req_id])