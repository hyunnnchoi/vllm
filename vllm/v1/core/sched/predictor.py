# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# [NOTE, hyunnnchoi, 2025.12.09] Learning-to-Rank predictor for LTR scheduling
# This module provides prediction capabilities for LTR scheduling policy.

import os
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from vllm.logger import init_logger

logger = init_logger(__name__)


class LTRPredictor:
    """
    Learning-to-Rank predictor for LTR scheduling.
    
    Uses a fine-tuned language model to predict the latency/priority score
    for incoming requests based on their prompts.
    
    Args:
        target_model: The target model being served (e.g., "facebook/opt-125m")
        predictor_model_path: Path to the fine-tuned predictor model checkpoint
    """
    
    def __init__(
        self, 
        target_model: str, 
        predictor_model_path: Optional[str] = None
    ) -> None:
        # [NOTE, hyunnnchoi, 2025.12.09] Allow environment variable override
        if predictor_model_path is None:
            predictor_model_path = os.environ.get(
                "VLLM_LTR_PREDICTOR_PATH",
                None
            )
        
        if predictor_model_path is None:
            raise ValueError(
                "predictor_model_path must be provided either as argument or "
                "via VLLM_LTR_PREDICTOR_PATH environment variable"
            )
        
        logger.info(f"[LTR] Initializing predictor from {predictor_model_path}")
        logger.info(f"[LTR] Target model: {target_model}")
        
        # Determine tokenizer based on predictor model
        self.predictor_tokenizer = self._get_predictor_tokenizer(
            predictor_model_path
        )
        
        # Target model tokenizer (for decoding prompts)
        try:
            self.target_tokenizer = AutoTokenizer.from_pretrained(target_model)
        except Exception as e:
            logger.warning(
                f"[LTR] Failed to load target tokenizer: {e}. "
                "Will use predictor tokenizer for both."
            )
            self.target_tokenizer = self.predictor_tokenizer
        
        # Load predictor model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.predictor = AutoModelForSequenceClassification.from_pretrained(
                predictor_model_path, 
                local_files_only=True
            ).eval().to(device)
            self.device = device
            logger.info(f"[LTR] Predictor loaded successfully on {device}")
        except Exception as e:
            logger.error(f"[LTR] Failed to load predictor model: {e}")
            raise
    
    def _get_predictor_tokenizer(self, predictor_model_path: str):
        """Determine which tokenizer to use for the predictor model."""
        # Try to infer from path
        if "opt-125m" in predictor_model_path.lower():
            return AutoTokenizer.from_pretrained("facebook/opt-125m")
        elif "opt-350m" in predictor_model_path.lower():
            return AutoTokenizer.from_pretrained("facebook/opt-350m")
        else:
            # Try to load from the predictor path itself
            try:
                return AutoTokenizer.from_pretrained(
                    predictor_model_path,
                    local_files_only=True
                )
            except Exception:
                logger.error(
                    f"[LTR] Could not load tokenizer from {predictor_model_path}. "
                    "Please ensure the predictor model directory contains a valid tokenizer configuration (e.g., tokenizer_config.json)."
                )
                raise RuntimeError(f"Failed to load tokenizer for predictor model at {predictor_model_path}.")
    
    def get_score(self, prompt_token_ids: list[int]) -> float:
        """
        Compute priority score for a request based on its prompt.
        
        Args:
            prompt_token_ids: Token IDs of the prompt
            
        Returns:
            Float score (higher score = higher priority)
        """
        try:
            # Decode prompt using target tokenizer
            prompt = self.target_tokenizer.decode(
                prompt_token_ids, 
                skip_special_tokens=True
            )
            
            # Tokenize for predictor
            input_tokens = self.predictor_tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512  # Reasonable limit
            ).to(self.device)
            
            # Get score
            with torch.no_grad():
                score = self.predictor(
                    input_tokens['input_ids'], 
                    input_tokens['attention_mask']
                ).logits.item()
            
            return score
            
        except Exception as e:
            logger.warning(f"[LTR] Failed to compute score: {e}")
            # Return neutral score on error
            return 0.0

