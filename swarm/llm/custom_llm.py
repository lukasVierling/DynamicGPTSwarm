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
    model_name = "google/gemma-2b-it" #Should be modifiable later

    def __init__(self):
        super().__init__()
        print("We are using custom LLM class")
        if CustomLLM.model is None:
            print("Load Model...")
            CustomLLM.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16).to("cuda")
        if CustomLLM.tokenizer is None:
            print("Load Tokenizer")
            CustomLLM.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


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

        prompt = CustomLLM.tokenizer.apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        prompt = CustomLLM.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
        outputs = CustomLLM.model.generate(
            prompt,
            do_sample=True,
            max_length=max_tokens,
            temperature=temperature,
            num_return_sequences=num_comps,
            top_k=50,
            top_p=1.0
        )
        output_text = CustomLLM.tokenizer.decode(outputs[0][prompt.shape[-1]:],skip_special_tokens=True)
        #print(output_text)
        return output_text

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
        outputs = CustomLLM.model.generate(
            prompt,
            do_sample=True,
            max_length=max_tokens,
            temperature=temperature,
            num_return_sequences=num_comps,
            top_k=50,
            top_p=1.0
        )
        output_text = CustomLLM.tokenizer.decode(outputs[0][prompt.shape[-1]:],skip_special_tokens=True)
        #print(output_text)
        return output_text
