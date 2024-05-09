#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from typing import List, Any, Optional
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer



class EmptyNode(Node): 
    def __init__(self, 
                 domain: str,
                 operation_description: str = "Output empty string",
                 id=None):
        super().__init__(operation_description, id, True)

        self.domain = domain
        self.prompt_set = PromptSetRegistry.get(self.domain)


    @property
    def node_name(self):
        return self.__class__.__name__
    
    async def node_optimize(self, input, meta_optmize=False):
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        if meta_optmize:
            print("node optimization for empty node not supported")

        return role, constraint



    async def _execute(self, inputs: List[Any] = [], **kwargs):
            
            node_inputs = self.process_input(inputs)
            outputs = []
            '''
            #check if inputs has "domain" key and if then get the prompt set for that domain
            for input_dict in node_inputs:
                if "domain" in input_dict:
                    prompt_set = PromptSetRegistry.get(input_dict["domain"])
                    break
            
            '''

            for input in node_inputs:
                task = input["task"]
                role, constraint = await self.node_optimize(input, meta_optmize=False)
                prompt = self.prompt_set.get_answer_prompt(question=task)    
                response = "" #empty response

                execution = {
                    "operation": self.node_name,
                    "task": task,
                    "files": input.get("files", []),
                    "input": task,
                    "role": role,
                    "constraint": constraint,
                    "prompt": prompt,
                    "output": response,
                    "ground_truth": input.get("GT", []),
                    "format": "natural language"
                }
                outputs.append(execution)
                self.memory.add(self.id, execution)

            # self.log()
            return outputs 
