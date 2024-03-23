import os
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import asyncio
from dataclasses import asdict
from typing import List, Union, Optional
from dotenv import load_dotenv
import random
import async_timeout
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time
from typing import Dict, Any

from vllm import LLM as vllm_LLM, SamplingParams

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.price import cost_count
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry

@LLMRegistry.register('CustomLLM')
@LLMRegistry.register('CustomLLM')

class CustomLLM(LLM):
    model = None
    tokenizer = None
    model_name = "google/gemma-7B-it" #Should be modifiable later

    def __init__(self):
        super().__init__()
        print("We are using custom LLM class")
        if CustomLLM.model is None:
            print("Load Model...")
            CustomLLM.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16).to("cuda")
           # CustomLLM.model = vllm_LLM(model="google/gemma-7B-it", dtype="half", max_model_len=5888)
        if CustomLLM.tokenizer is None:
            print("Load Tokenizer")
            CustomLLM.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def process_messages(self,messages: List[Message]) -> List[Message]:
        processed_messages = []
        system_message = None

        for message in messages:
            if message.role == 'system':
                system_message = message.content
            elif message.role == 'user':
                if system_message:
                    message.content = system_message + ' ' + message.content
                    system_message = None
                processed_messages.append(message)

        # Handle the case where the last message is a system message
        if system_message:
            processed_messages.append(Message(role='user', content=system_message))

        return processed_messages

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        else:
            messages = self.process_messages(messages)
        #print a fancy console seperater
        prompt = CustomLLM.tokenizer.apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        prompt = CustomLLM.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        prompt_len = len(prompt[0])
        outputs = CustomLLM.model.generate(
            prompt,
            do_sample=True,
            max_length=max_tokens + prompt_len,
            temperature=temperature,
            num_return_sequences=num_comps,
            top_k=50,
            top_p=1.0
        )
        output_text = CustomLLM.tokenizer.decode(outputs[0][prompt.shape[-1]:],skip_special_tokens=True)
        return output_text
        '''
        sampling_params = SamplingParams(temperature=temperature, top_p=1, top_k=50,max_tokens=max_tokens)
        output = CustomLLM.model.generate(prompt, sampling_params, use_tqdm=False)
        print(output[0].outputs[0].text)
        return output[0].outputs[0].text
        '''


    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        

        prompt = CustomLLM.tokenizer.apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        prompt = CustomLLM.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        prompt_len = len(prompt[0])
        outputs = CustomLLM.model.generate(
            prompt,
            do_sample=True,
            max_length=max_tokens + prompt_len,
            temperature=temperature,
            num_return_sequences=num_comps,
            top_k=50,
            top_p=1.0
        )
        output_text = CustomLLM.tokenizer.decode(outputs[0][prompt.shape[-1]:],skip_special_tokens=True)
        return output_text
