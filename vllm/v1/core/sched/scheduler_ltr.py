# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [NOTE, hyunnnchoi, 2025.12.09] Learning-to-Rank Scheduler
# Complete port of jaehoon-ltr branch scheduling logic

from __future__ import annotations

import os
import time
from typing import Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVEventBatch
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData
from vllm.v1.core.sched.predictor import LTRPredictor
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class LTRScheduler(Scheduler):
    """
    Learning-to-Rank Scheduler (jaehoon-ltr port).
    
    Complete implementation matching jaehoon-ltr's scheduling logic:
    - Combines waiting + running queues
    - Sorts by (promote, score) descending
    - Starvation prevention with quantum
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
        # [NOTE, hyunnnchoi, 2025.12.09] Temporarily set policy to FCFS for base init
        original_policy = vllm_config.scheduler_config.policy
        vllm_config.scheduler_config.policy = "fcfs"
        
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        
        vllm_config.scheduler_config.policy = original_policy
        self.policy = SchedulingPolicy.LTR
        
        # For LTR we need direct access to requests list
        from vllm.v1.core.sched.request_queue import FCFSRequestQueue
        self.waiting = FCFSRequestQueue()
        
        # Initialize LTR predictor
        predictor_path = os.environ.get("VLLM_LTR_PREDICTOR_PATH")
        if predictor_path is None:
            raise ValueError(
                "VLLM_LTR_PREDICTOR_PATH must be set for LTR scheduling"
            )
        
        logger.info("[LTR] Initializing predictor from %s", predictor_path)
        
        self.ltr_predictor = LTRPredictor(
            target_model=vllm_config.model_config.model,
            predictor_model_path=predictor_path
        )
        
        # Starvation prevention (from jaehoon-ltr)
        self.starvation_threshold = int(os.environ.get(
            "VLLM_LTR_STARVATION_THRESHOLD", "256"
        ))
        self.priority_quantum = int(os.environ.get(
            "VLLM_LTR_PRIORITY_QUANTUM", "32"
        ))
        
        logger.info(
            "[LTR] Starvation threshold: %d, Priority quantum: %d",
            self.starvation_threshold, self.priority_quantum
        )
    
    def add_request(self, request: Request) -> None:
        """Add request with LTR score and starvation tracking."""
        # Compute LTR score
        try:
            score = self.ltr_predictor.get_score(request.prompt_token_ids)
            request.score = score
            logger.debug(
                "[LTR] Request %s score: %.3f",
                request.request_id[:8], score
            )
        except Exception as e:
            logger.warning("[LTR] Score computation failed: %s", e)
            request.score = 0.0
        
        # Initialize jaehoon-ltr attributes
        request.promote = False
        request.starvation = 0
        request.quantum = None
        
        # Call parent
        super().add_request(request)
    
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

        # ltr scheduling
        sorted_requests = sorted(self.waiting.requests + self.running, key=lambda req: (req.promote, req.score), reverse=True)
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

