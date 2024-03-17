from transformers import AutoTokenizer, pipeline
import torch
import asyncio
import os
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

import pdb

#from huggingface_hub import login 

@LLMRegistry.register('CustomLLM')
class CustomLLM(LLM):
    def __init__(self, model_name: str):
        print("We are using custom LLM class")
        self.model_name = model_name
        path = f"./models/{self.model_name}/pipeline"
        # Check if the path exists
        if os.path.exists(path):
            # Iterate over the files in the folder
            for filename in os.listdir(path):
                # Print each file name
                print("Folder path does exist.")
        else:
            print("Folder path does not exist.")

        self.pipeline = pipeline(
            "text-generation",
            path
        )
        #Old settings but not working because hf token not working
        '''
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        '''

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

        prompt = pipeline.tokenizer.apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=1.0
        )
        print("We are using the custom llm in asynch lets gooo")
        return outputs[0]["generated_text"][len(prompt):]

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

        prompt = self.pipeline.tokenizer.apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=1.0
        )
        print("We are using the custom llm in synch lets gooo")

        return outputs[0]["generated_text"][len(prompt):]
