import torch 
import time
from transformers import T5Tokenizer
from pprint import pprint
import asyncio
from readers.utils import CONTEXT_PROMPT, create_prompt
import nest_asyncio
import text_generation as tg


device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)

time_map = {}
        
class FlanT5Reader:
    def __init__(self, hosted_api_path=None):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        self.hosted_api_endpoint = hosted_api_path
        nest_asyncio.apply()
        self.async_client = tg.AsyncClient(self.hosted_api_endpoint)

    
    async def batch_generate(self, texts, max_new_tokens=10, truncate=2000):
        return await asyncio.gather(*[self.async_client.generate(text, max_new_tokens=max_new_tokens, truncate=truncate) for text in texts])


    def generate(self, prompts, max_new_tokens=10, truncate=2000):
        print("In flan t5 generation")

        total_tokens = 2000
        prompt_strs = []
        context_length_changes = []
        context_prompt_tokenized = self.tokenizer(CONTEXT_PROMPT)
        for prompt in prompts:
            # print(prompt)
            question_tokenized = self.tokenizer(prompt["question"])
            remaining_length = total_tokens-len(context_prompt_tokenized)-len(question_tokenized)-10
            context_tokenized_without_truncation = self.tokenizer(prompt["context"], add_special_tokens=False)
            context_tokenized = self.tokenizer(prompt["context"], max_length=remaining_length, truncation=True, add_special_tokens=False)
            context_after_truncation = self.tokenizer.decode(context_tokenized["input_ids"])

            context_length_changes.append([len(prompt["context"]), len(context_after_truncation), len(context_tokenized_without_truncation["input_ids"]), len(context_tokenized["input_ids"])])

            prompt_strs.append(create_prompt(question=prompt["question"], context=context_after_truncation))

        responses = asyncio.run(self.batch_generate(prompt_strs, max_new_tokens, truncate))
        return [r.generated_text for r in responses], context_length_changes

            


