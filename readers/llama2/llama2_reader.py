import torch 
import time
from transformers import LlamaTokenizer, AutoModelForCausalLM
from pprint import pprint
import asyncio
from readers.utils import CONTEXT_PROMPT, create_prompt
import nest_asyncio
import text_generation as tg


device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)

time_map = {}
        
class LlamaReader:
    def __init__(self, hosted_api_path=None, tokenizer_path="/data/datasets/models/meta-ai/llama2/weights/", model_path="/data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf"):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        if not hosted_api_path:
            start_time = time.time()
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
            current_time1 = time.time()
            time_to_load_tokenizer = current_time1-start_time
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            current_time2 = time.time()
            time_to_load_model = current_time2-current_time1
            self.model = self.model.to(device)
            current_time3 = time.time()
            time_to_put_model_to_device = current_time3-current_time2

            time_map["time_to_load_tokenizer"]=time_to_load_tokenizer
            time_map["time_to_load_model"]=time_to_load_model
            time_map["time_to_put_model_to_device"] = time_to_put_model_to_device
        else:
            self.hosted_api_endpoint = hosted_api_path
            nest_asyncio.apply()
            self.async_client = tg.AsyncClient(self.hosted_api_endpoint)

    
    async def batch_generate(self, texts, max_new_tokens=20, truncate=4000):
        return await asyncio.gather(*[self.async_client.generate(text, max_new_tokens=max_new_tokens, truncate=truncate) for text in texts])


    def generate(self, prompts, max_new_tokens=20, truncate=4000):

        total_tokens = 4000
        prompt_strs = []
        context_length_changes = []
        context_prompt_tokenized = self.tokenizer(CONTEXT_PROMPT)
        for prompt in prompts:
            question_tokenized = self.tokenizer(prompt["question"])
            remaining_length = total_tokens-len(context_prompt_tokenized)-len(question_tokenized)-10
            context_tokenized = self.tokenizer(prompt["context"], max_length=remaining_length, truncation=True, add_special_tokens=False)
            context_after_truncation = self.tokenizer.decode(context_tokenized["input_ids"])
            context_length_changes.append([len(prompt["context"]), len(context_after_truncation)])

            prompt_strs.append(create_prompt(question=prompt["question"], context=context_after_truncation))

        responses = asyncio.run(self.batch_generate(prompt_strs, max_new_tokens, truncate))
        return [r.generated_text for r in responses], context_length_changes
    
        # if not self.hosted_api_endpoint:
        #     final_outputs = []
        #     for combined_prompt in combined_prompts:
        #         inputs = self.tokenizer(combined_prompt, return_tensors="pt")
        #         inputs = inputs.to(device)

            
        #         # Generate
        #         generate_ids = self.model.generate(inputs.input_ids, max_length=max_length)
        #         result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #         print("result: ", result, "\n")
        #         final_output = remove_substring(result, combined_prompt)
        #         final_outputs.append(final_output)
        #     return final_outputs
        # else:
            


