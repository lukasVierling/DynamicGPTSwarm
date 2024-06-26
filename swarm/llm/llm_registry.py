import sys
sys.path.append('/opt/homebrew/lib/python3.9/site-packages')
from typing import Optional
from class_registry import ClassRegistry

from swarm.llm.llm import LLM

CUSTOM_LLMS = ["google/gemma-2B-it","google/gemma-7B-it","vivo-ai/BlueLM-7B-Chat","meta-llama/Meta-Llama-3-8B-Instruct"]

class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None:
            model_name = "gpt-4-1106-preview"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif model_name in CUSTOM_LLMS:
            # for custom model get the smalle gemma model and run on our gpus
            model = cls.registry.get('CustomLLM', model_name)
        else: # any version of GPTChat like "gpt-4-1106-preview"
            model = cls.registry.get('GPTChat', model_name)

        return model
