#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from typing import List, Any, Optional

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry

import random


class AdversarialAnswer(Node):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Directly output an answer.",
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

    @property
    def node_name(self):
        return self.__class__.__name__

    def meta_prompt(self, input, meta_init=False):

        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()
        prompt = self.prompt_set.get_adversarial_answer_prompt(question=task)    

        if meta_init:
            pass #TODO

        return role, constraint, prompt


    async def _execute(self, inputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        inputs = []
        for input in node_inputs:
            role, constraint, prompt= self.meta_prompt(input, meta_init=False)
            message = [Message(role="system", content=f"You are a {role}. {constraint}"),
                    Message(role="user", content=prompt)]
            #response = await self.llm.agen(message)
            #idx = random.randint(0, 3) to support random decision
            idx = 0 #always say A no matter the question
            options = ['A', 'B', 'C', 'D'] #modified the adversarial agent because he can't lie lol
            response = options[idx]
            _memory = {
                "operation": self.node_name,
                #"task_id": input["task_id"], 
                "task": input["task"],
                "files": input.get("files", []),
                "input": input["task"],
                "subtask": prompt,
                "output": response,
                "format": "natural language"
            }

            # self.log()
            inputs.append(_memory)
        return inputs

