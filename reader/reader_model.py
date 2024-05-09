from reader.reader_utils import INSTRUCTION_STR, NO_CONTEXT_INSTRUCTION_STR, truncate_prompt, num_gpt_tokens_per_content, num_gpt_tokens_per_message
from reader.reader_utils import complete_model_names
from litellm import batch_completion, Router
import os
import tiktoken
import pdb
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

        responses = self.batch_generate(modified_prompts, max_new_tokens)
        return [r.choices[0].message.content for r in responses], context_length_changes

class GPT_Reader():
    def __init__(self, model_identifier, api_key):
        self.model_identifier = model_identifier
        os.environ["OPENAI_API_KEY"]=api_key
        self.api_key = api_key
        try:
            self.tokenizer = tiktoken.encoding_for_model(complete_model_names[self.model_identifier])
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    
    def truncate_prompt(self, prompt, total_tokens, max_new_tokens):

        system_prompt = f"{INSTRUCTION_STR}\nContext: " if prompt['context'] else NO_CONTEXT_INSTRUCTION_STR
        user_prompt = f"Question: {prompt['question']}\nAnswer: "
        system_prompt_len = num_gpt_tokens_per_message([{"role": "system", "content": system_prompt}], complete_model_names[self.model_identifier])
        user_prompt_len = num_gpt_tokens_per_message([{"role": "user", "content": user_prompt}], complete_model_names[self.model_identifier])
        # instruction_len, _ = num_gpt_tokens_per_content(f"{INSTRUCTION_STR}\nContext: ", complete_model_names[self.model_identifier])
        response_header_len = 3
        remaining_length = total_tokens - system_prompt_len - user_prompt_len - max_new_tokens - response_header_len - 5 #additional buffer of 5

        # hard truncate
        context_len, encoded_context = num_gpt_tokens_per_content(prompt["context"], complete_model_names[self.model_identifier])

        # truncate the encoded context
        if context_len > remaining_length:
            trun_decoded_context = self.tokenizer.decode(encoded_context[:remaining_length])

            trunc_context_len = num_gpt_tokens_per_content(trun_decoded_context, complete_model_names[self.model_identifier])
        else:
            trunc_context_len = context_len
            trun_decoded_context = prompt["context"]

        context_length_change_info = {
            "original_context_str_length": len(prompt['context']),
            "context_str_length_after_truncation": len(trun_decoded_context),
            "original_context_token_length": context_len,
            "context_token_length_after_truncation": trunc_context_len
        }
    
        messages = self.create_prompt(question=prompt["question"], context=trun_decoded_context)
        return messages, context_length_change_info
    
    def create_prompt(self, question, context):
        messages = []

        if context:
            system_prompt = f"{INSTRUCTION_STR}\nContext: {context}".strip()
            user_prompt = f"Question: {question}\nAnswer: ".strip()
            # system_prompt =  f"{INSTRUCTION_STR}\nContext: {context}".strip()
        else:
            system_prompt = f"{NO_CONTEXT_INSTRUCTION_STR}".strip()
            user_prompt = f"Question: {question}\nAnswer: ".strip()
            # system_prompt = f"{NO_CONTEXT_INSTRUCTION_STR}".strip()
    
        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages
       

    def generate(self, prompts, max_new_tokens=10, truncate=2000):
        total_tokens = truncate
        modified_prompts = []
        context_length_changes = []

        for prompt in prompts:
            modified_prompt, context_length_change_info = self.truncate_prompt(prompt, total_tokens, max_new_tokens)
            modified_prompts.append(modified_prompt)
            context_length_changes.append(context_length_change_info)

        model_list = [{
            "model_name": self.model_identifier,
            "litellm_params":{"model": complete_model_names[self.model_identifier],\
                "api_key": self.api_key,\
                "rpm": 10_000,
                "tpm": 2_000_000
            }
        }]
        router = Router(model_list = model_list,\
                        num_retries = 6, \
                        allowed_fails = 6,\
                        cooldown_time = 1)
        responses = []


        for messages in modified_prompts:
            response = router.completion(model = self.model_identifier, messages = messages, temperature = 0.0)
            responses.append(response)


            

        # responses = self.batch_generate(modified_prompts, max_new_tokens)
        return [r.choices[0].message.content for r in responses], context_length_changes
    

# class Claude_Reader():
#     def __init__(self, model_identifier, api_key):
#         self.model_identifier = model_identifier
#         os.environ["OPENAI_API_KEY"]=api_key
#         try:
#             self.tokenizer = tiktoken.encoding_for_model(complete_model_names[self.model_identifier])
#         except KeyError:
#             print("Warning: model not found. Using cl100k_base encoding.")
#             self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    
#     def truncate_prompt(self, prompt, total_tokens, max_new_tokens):

#         question_len = num_gpt_tokens_per_message([{"role": "user", "content": f"Question: {prompt["question"]}\nAnswer: "}], self.model_identifier)
#         instruction_len = num_gpt_tokens_per_content(f"{INSTRUCTION_STR}\nContext: ", self.model_identifier)
#         response_header_len = 3
#         remaining_length = total_tokens - instruction_len - question_len - max_new_tokens - response_header_len - 5 #additional buffer of 5

#         # hard truncate
#         context_len, encoded_context = num_gpt_tokens_per_content(prompt["context"], self.model_identifier)

#         # truncate the encoded context
#         if context_len > remaining_length:
#             trun_decoded_context = self.tokenizer.decode(encoded_context[:remaining_length])

#             trunc_context_len = num_gpt_tokens_per_content(trun_decoded_context, self.model_identifier)
#         else:
#             trunc_context_len = context_len
#             trun_decoded_context = prompt["context"]

#         context_length_change_info = {
#             "original_context_str_length": len(prompt['context']),
#             "context_str_length_after_truncation": len(trun_decoded_context),
#             "original_context_token_length": context_len,
#             "context_token_length_after_truncation": trunc_context_len
#         }
        

#         messages = self.create_prompt(question=prompt["question"], context=trun_decoded_context)
#         return messages, context_length_change_info
    
#     def create_prompt(self, question, context):
#         messages = []

#         if context:
#             system_prompt =  f"{INSTRUCTION_STR}\nContext: {context}".strip()
#         else:
#             system_prompt = f"{NO_CONTEXT_INSTRUCTION_STR}".strip()
        
    
#         messages.append({"role": "system", "content":system_prompt})
#         messages.append({"role": "user", "content": f"Question: {question}\nAnswer: "})
#         return messages
       

#     def generate(self, prompts, max_new_tokens=10, truncate=2000):
#         total_tokens = truncate
#         modified_prompts = []
#         context_length_changes = []

#         for prompt in prompts:
#             modified_prompt, context_length_change_info = self.truncate_prompt(prompt, total_tokens, max_new_tokens)
#             modified_prompts.append(modified_prompt)
#             context_length_changes.append(context_length_change_info)

#         model_list = [{
#             "model_name": self.model_identifier,
#             "litellm_params":{"model": complete_model_names[self.model_identifier],\
#                 "api_key": self.api_key,\
#                 "rpm": 10_000,
#                 "tpm": 2_000_000
#             }
#         }]
#         router = Router(model_list = model_list,\
#                         num_retries = 6, \
#                         allowed_fails = 6,\
#                         cooldown_time = 1)
#         responses = []
#         for messages in modified_prompts:
#             response = router.completion(model = self.model_identifier, messages = messages)
#             responses.append(response)
#         # responses = self.batch_generate(modified_prompts, max_new_tokens)
#         return [r.choices[0].message.content for r in responses], context_length_changes