from reader.utils import INSTRUCTION_STR, truncate_prompt
from utils import complete_model_names
from litellm import batch_completion 

time_map = {}
        
class Reader:
    def __init__(self, model_identifier=None, hosted_api_endpoint=None, tokenizer=None):
        self.model_identifier = model_identifier
        self.tokenizer = tokenizer
        self.hosted_api_endpoint = hosted_api_endpoint
    
    def batch_generate(self, texts, max_new_tokens=10):
        messages = [[{"role":"user", "content":text}] for text in texts]
        return batch_completion(model=complete_model_names[self.model_identifier], 
                                messages=messages, 
                                api_base=self.hosted_api_endpoint,
                                max_tokens = max_new_tokens
                                )


    def generate(self, prompts, max_new_tokens=10, truncate=2000):
        total_tokens = truncate
        modified_prompts = []
        context_length_changes = []
        instruction_str_tokens = self.tokenizer(INSTRUCTION_STR)["input_ids"]
        for prompt in prompts:
            modified_prompt, context_length_change_info = truncate_prompt(prompt, self.tokenizer, instruction_str_tokens, total_tokens, max_new_tokens)
            modified_prompts.append(modified_prompt)
            context_length_changes.append(context_length_change_info)

        responses = self.batch_generate(modified_prompts, max_new_tokens, truncate)
        return [r.choices[0].message.content for r in responses], context_length_changes
