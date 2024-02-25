import asyncio
from reader.utils import INSTRUCTION_STR, truncate_prompt
import nest_asyncio
import text_generation as tg

time_map = {}
        
class Reader:
    def __init__(self, hosted_api_endpoint=None, tokenizer=None):
        self.tokenizer = tokenizer
        self.hosted_api_endpoint = hosted_api_endpoint

        # initialize async text generation inference client
        nest_asyncio.apply()
        self.async_client = tg.AsyncClient(self.hosted_api_endpoint)

    
    async def batch_generate(self, texts, max_new_tokens=10, truncate=2000):
        return await asyncio.gather(*[self.async_client.generate(text, max_new_tokens=max_new_tokens, truncate=truncate) for text in texts])


    def generate(self, prompts, max_new_tokens=10, truncate=2000):
        total_tokens = truncate
        modified_prompts = []
        context_length_changes = []
        instruction_str_tokens = self.tokenizer(INSTRUCTION_STR)["input_ids"]
        for prompt in prompts:
            modified_prompt, context_length_change_info = truncate_prompt(prompt, self.tokenizer, instruction_str_tokens, total_tokens)
            modified_prompts.append(modified_prompt)
            context_length_changes.append(context_length_change_info)

        responses = asyncio.run(self.batch_generate(modified_prompts, max_new_tokens, truncate))
        return [r.generated_text for r in responses], context_length_changes