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

from swarm.utils.select_gpu import select_gpu


@LLMRegistry.register('CustomLLM')

class CustomLLM(LLM):
    models = {} #2-dim after alowing for several models on different gpus
    tokenizers = {}
    devices = {} #2-dim after alowing for several models on different gpus
    MAX_LLMS = 1
    

    def __init__(self, model_name: Optional[str] = "google/gemma-7B-it"):
        super().__init__()
        print(f"We are using custom LLM class, model_name: {model_name}")
        self.model_name = model_name
        if self.model_name not in CustomLLM.models:
            print("Load Model...")
            CustomLLM.models[self.model_name] = {}
            CustomLLM.devices[self.model_name] = {}
            for i in range (CustomLLM.MAX_LLMS):
                CustomLLM.devices[self.model_name][i] = select_gpu()
                CustomLLM.models[self.model_name][i] = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(f"cuda:{CustomLLM.devices[self.model_name][i]}")
        if self.model_name not in CustomLLM.tokenizers:
            print("Load Tokenizer...")
            CustomLLM.tokenizers[self.model_name] = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
            if CustomLLM.tokenizers[self.model_name].pad_token is None:
                CustomLLM.tokenizers[self.model_name].pad_token = CustomLLM.tokenizers[self.model_name].eos_token
            if self.model_name == "vivo-ai/BlueLM-7B-Chat":
                CustomLLM.tokenizers[self.model_name].apply_chat_template = BlueLM_apply_chat_template
        self.terminators = [CustomLLM.tokenizers[self.model_name].eos_token_id]
        if self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            self.terminators += [CustomLLM.tokenizers[self.model_name].convert_tokens_to_ids("<|eot_id|>")]


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
            if self.model_name in ["google/gemma-2B-it","google/gemma-7B-it","vivo-ai/BlueLM-7B-Chat"]:
                messages = self.process_messages(messages)
        
        prompt = CustomLLM.tokenizers[self.model_name].apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        idx = random.randrange(0,CustomLLM.MAX_LLMS) #get a random worker to do the job
        prompt = CustomLLM.tokenizers[self.model_name].encode(prompt, add_special_tokens=False, return_tensors="pt").to(f"cuda:{CustomLLM.devices[self.model_name][idx]}") #add special token for blue apparently true...?
        prompt_len = len(prompt[0])
        outputs = CustomLLM.models[self.model_name][idx].generate(
            prompt,
            do_sample=True,
            max_length=max_tokens + prompt_len,
            temperature=temperature,
            num_return_sequences=num_comps,
            top_k=50,
            top_p=0.9, #changed from 1.0 -> 0.9
            eos_token_id=self.terminators,
            pad_token_id =CustomLLM.tokenizers[self.model_name].pad_token_id
        )
        output_text = CustomLLM.tokenizers[self.model_name].decode(outputs[0][prompt.shape[-1]:],skip_special_tokens=True)
        #print("These are the outputs after generation: ")
        #print(output_text)
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
        else:
            messages = self.process_messages(messages)

        prompt = CustomLLM.tokenizers[self.model_name].apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        idx = random.randrange(0,CustomLLM.MAX_LLMS) #get a random worker to do the job
        prompt = CustomLLM.tokenizers[self.model_name].encode(prompt, add_special_tokens=False, return_tensors="pt").to(f"cuda:{CustomLLM.devices[self.model_name][idx]}") #add special token for blue apparently true...?
        prompt_len = len(prompt[0])
        outputs = CustomLLM.models[self.model_name][idx].generate(
            prompt,
            do_sample=True,
            max_length=max_tokens + prompt_len,
            temperature=temperature,
            num_return_sequences=num_comps,
            top_k=50,
            top_p=0.9, #changed from 1.0 -> 0.9
        )
        output_text = CustomLLM.tokenizers[self.model_name].decode(outputs[0][prompt.shape[-1]:],skip_special_tokens=True)

        return output_text

def BlueLM_apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs):
    if add_generation_prompt:
        if messages[-1]["role"] == "user":
            messages.append({"role": "assistant", "content": ""})
    if tokenize:
        raise NotImplementedError("Tokenization not implemented for BlueLM")
    chat = ""
    for message in messages:
        role, content = message.values()
        if role == "user":
            chat += f"[|Human|]:{content}"
        elif role == "assistant":
            chat += f"[|AI|]:{content}"
        else:
            raise ValueError(f"Unknown role {role}")
    return chat