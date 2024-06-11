import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
from dataclasses import asdict
from typing import List, Union, Optional, Dict, Any
from dotenv import load_dotenv
import random
import async_timeout
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.price import cost_count
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry

from swarm.utils.select_gpu import select_gpu


@LLMRegistry.register('CustomLLM')
class CustomLLM(LLM):
    """
    CustomLLM is a class that handles loading, managing, and utilizing language models for text generation.

    Attributes:
        models (Dict[str, Dict[int, AutoModelForCausalLM]]): Loaded models indexed by model name and device ID.
        tokenizers (Dict[str, AutoTokenizer]): Tokenizers for each model.
        devices (Dict[str, Dict[int, int]]): Device IDs for each model.
        MAX_LLMS (int): Maximum number of models to load.
        tokens (Dict[str, Dict[str, Dict[str, int]]]): Token usage statistics.
        count_tokens (bool): Flag to indicate whether to count tokens.
    """
    models = {}  # Loaded models indexed by model name and device ID
    tokenizers = {}
    devices = {}  # Device IDs for each model
    MAX_LLMS = 1  # Maximum number of models to load
    tokens = {}
    count_tokens = False

    @staticmethod
    def start_counter():
        """Starts the token counting mechanism."""
        CustomLLM.count_tokens = True
        CustomLLM.tokens = {model: {"actual": {"input": 0, "output": 0}, "max": {"input": 0, "output": 0}} for model in CustomLLM.models}

    @staticmethod
    def end_counter():
        """Ends the token counting mechanism."""
        CustomLLM.count_tokens = False

    @staticmethod
    def reset_counter():
        """Resets the token counters."""
        CustomLLM.tokens = {model: {"actual": {"input": 0, "output": 0}, "max": {"input": 0, "output": 0}} for model in CustomLLM.models}

    @staticmethod
    def get_tokens() -> Dict[str, Dict[str, Dict[str, int]]]:
        """Gets the current token counts.

        Returns:
            Dict[str, Dict[str, Dict[str, int]]]: Current token counts.
        """
        return CustomLLM.tokens

    @staticmethod
    def get_price(prices: Dict[str, Dict[str, float]]) -> float:
        """Calculates the cost of the tokens used based on a price list.

        Args:
            prices (Dict[str, Dict[str, float]]): Price list in USD per million tokens.

        Returns:
            float: Calculated cost.
        """
        cost = 0
        for model, tokens in CustomLLM.tokens.items():
            price_input = prices.get(model, {}).get("input", 0)
            price_output = prices.get(model, {}).get("output", 0)
            cost += price_input * tokens["actual"]["input"] / 1_000_000 + price_output * tokens["actual"]["output"] / 1_000_000
        return cost

    @staticmethod
    def get_max_price(prices: Dict[str, Dict[str, float]]) -> float:
        """Calculates the maximum possible cost of the tokens used based on a price list.

        Args:
            prices (Dict[str, Dict[str, float]]): Price list in USD per million tokens.

        Returns:
            float: Calculated maximum cost.
        """
        cost = 0
        for model, tokens in CustomLLM.tokens.items():
            price_input = prices.get(model, {}).get("input", 0)
            price_output = prices.get(model, {}).get("output", 0)
            cost += price_input * tokens["max"]["input"] / 1_000_000 + price_output * tokens["max"]["output"] / 1_000_000
        return cost

    def __init__(self, model_name: Optional[str] = "google/gemma-7B-it"):
        """Initializes the CustomLLM class.

        Args:
            model_name (Optional[str], optional): Name of the model to load. Defaults to "google/gemma-7B-it".
        """
        super().__init__()
        print(f"Using custom LLM class, model_name: {model_name}")
        self.model_name = model_name

        if self.model_name not in CustomLLM.models:
            print("Loading model...")
            CustomLLM.models[self.model_name] = {}
            CustomLLM.devices[self.model_name] = {}
            for i in range(CustomLLM.MAX_LLMS):
                CustomLLM.devices[self.model_name][i] = select_gpu()
                CustomLLM.models[self.model_name][i] = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
                ).to(f"cuda:{CustomLLM.devices[self.model_name][i]}")

        if self.model_name not in CustomLLM.tokenizers:
            print("Loading tokenizer...")
            CustomLLM.tokenizers[self.model_name] = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
            if CustomLLM.tokenizers[self.model_name].pad_token is None:
                CustomLLM.tokenizers[self.model_name].pad_token = CustomLLM.tokenizers[self.model_name].eos_token
            if self.model_name == "vivo-ai/BlueLM-7B-Chat":
                CustomLLM.tokenizers[self.model_name].apply_chat_template = BlueLM_apply_chat_template

        self.terminators = [CustomLLM.tokenizers[self.model_name].eos_token_id]
        if self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            self.terminators += [CustomLLM.tokenizers[self.model_name].convert_tokens_to_ids("")]

    def process_messages(self, messages: List[Message]) -> List[Message]:
        """Processes a list of messages, combining system messages with user messages.

        Args:
            messages (List[Message]): List of messages to process.

        Returns:
            List[Message]: Processed list of messages.
        """
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
        """Asynchronously generates text based on a list of messages.

        Args:
            messages (List[Message]): List of input messages.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Defaults to None.
            num_comps (Optional[int], optional): Number of completions to generate. Defaults to None.

        Returns:
            Union[List[str], str]: Generated text.
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFAULT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        else:
            if self.model_name in ["google/gemma-2B-it", "google/gemma-7B-it", "vivo-ai/BlueLM-7B-Chat"]:
                messages = self.process_messages(messages)

        prompt = CustomLLM.tokenizers[self.model_name].apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        idx = random.randrange(0, CustomLLM.MAX_LLMS)
        prompt = CustomLLM.tokenizers[self.model_name].encode(prompt, add_special_tokens=False, return_tensors="pt").to(f"cuda:{CustomLLM.devices[self.model_name][idx]}")
        prompt_len = len(prompt[0])

        if CustomLLM.count_tokens:
            CustomLLM.tokens[self.model_name]["actual"]["input"] += prompt_len
            CustomLLM.tokens[self.model_name]["max"]["input"] += prompt_len

        outputs = CustomLLM.models[self.model_name][idx].generate(
            prompt,
            do_sample=True,
            max_length=max_tokens + prompt_len,
            temperature=temperature,
            num_return_sequences=num_comps,
            top_k=50,
            top_p=0.9,
            eos_token_id=self.terminators,
            pad_token_id=CustomLLM.tokenizers[self.model_name].pad_token_id,
        )

        if CustomLLM.count_tokens:
            CustomLLM.tokens[self.model_name]["actual"]["output"] += len(outputs[0])
            CustomLLM.tokens[self.model_name]["max"]["output"] += max_tokens + prompt_len

        output_text = CustomLLM.tokenizers[self.model_name].decode(outputs[0][prompt.shape[-1]:], skip_special_tokens=True)
        return output_text

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Generates text based on a list of messages.

        Args:
            messages (List[Message]): List of input messages.
            max_tokens (Optional[int], optional): Maximum number of tokens to generate. Defaults to None.
            temperature (Optional[float], optional): Sampling temperature. Defaults to None.
            num_comps (Optional[int], optional): Number of completions to generate. Defaults to None.

        Returns:
            Union[List[str], str]: Generated text.
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFAULT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        else:
            messages = self.process_messages(messages)

        prompt = CustomLLM.tokenizers[self.model_name].apply_chat_template([asdict(message) for message in messages], tokenize=False, add_generation_prompt=True)
        idx = random.randrange(0, CustomLLM.MAX_LLMS)
        prompt = CustomLLM.tokenizers[self.model_name].encode(prompt, add_special_tokens=False, return_tensors="pt").to(f"cuda:{CustomLLM.devices[self.model_name][idx]}")
        prompt_len = len(prompt[0])

        if CustomLLM.count_tokens:
            CustomLLM.tokens[self.model_name]["actual"]["input"] += prompt_len
            CustomLLM.tokens[self.model_name]["max"]["input"] += prompt_len

        outputs = CustomLLM.models[self.model_name][idx].generate(
            prompt,
            do_sample=True,
            max_length=max_tokens + prompt_len,
            temperature=temperature,
            num_return_sequences=num_comps,
            top_k=50,
            top_p=0.9,
            eos_token_id=self.terminators,
            pad_token_id=CustomLLM.tokenizers[self.model_name].pad_token_id,
        )

        if CustomLLM.count_tokens:
            CustomLLM.tokens[self.model_name]["actual"]["output"] += len(outputs[0])
            CustomLLM.tokens[self.model_name]["max"]["output"] += max_tokens + prompt_len

        output_text = CustomLLM.tokenizers[self.model_name].decode(outputs[0][prompt.shape[-1]:], skip_special_tokens=True)
        return output_text


def BlueLM_apply_chat_template(messages: List[Dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True, **kwargs) -> str:
    """Applies a chat template for BlueLM.

    Args:
        messages (List[Dict[str, str]]): List of messages to format.
        tokenize (bool, optional): Flag to tokenize messages. Defaults to False.
        add_generation_prompt (bool, optional): Flag to add generation prompt. Defaults to True.

    Returns:
        str: Formatted chat string.
    """
    if add_generation_prompt and messages[-1]["role"] == "user":
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
