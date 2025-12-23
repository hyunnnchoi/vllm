import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# [NOTE, hjhoon03, 2025.12.11]: predictor for learning-to-rank policy
class LTRPredictor():
    def __init__(self, target_model: str, predictor_model_name: str, predictor_model_path: str) -> None:
        # tokenizer 생성
        if "opt-125m" in predictor_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        elif "opt-350m" in predictor_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        else:
            raise NotImplementedError
        
        # target model tokenizer (decode 용)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model)
        # predictor
        self.predictor = AutoModelForSequenceClassification.from_pretrained(predictor_model_path, local_files_only=True).eval().to("cuda")


    def get_score(self, prompt_token_ids: list[int]) -> float:
        prompt = self.target_tokenizer.decode(prompt_token_ids) # token id 형태의 input을 문장 형태로 decode
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to("cuda") # predictor에 맞는 형태로 tokenize
        score = self.predictor(input_tokens['input_ids'], input_tokens['attention_mask']).logits.item() # prediction

        return score