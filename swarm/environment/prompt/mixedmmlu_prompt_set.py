#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from swarm.environment.prompt.prompt_set import PromptSet
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.prompt.common import get_combine_materials

from swarm.environment.prompt.cmmlu_prompt_set import CMMLUPromptSet
from swarm.environment.prompt.mmlu_prompt_set import MMLUPromptSet

import warnings

#@PromptSetRegistry.register('mixedmmlu')
class MixedMMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option question answering.
    """
    @staticmethod
    def warn_and_raise():
        warnings.warn("This prompt set should not be used. Use MMLU or CMMLU prompt set instead.")
        raise NotImplementedError("This prompt set should not be used. Use MMLU or CMMLU prompt set instead.")

    @staticmethod
    def warn_and_return_empty():
        warnings.warn("This prompt set should not be used. Use MMLU or CMMLU prompt set instead.")
        return ""

    @staticmethod
    def get_role():
        return MixedMMLUPromptSet.warn_and_return_empty()

    @staticmethod
    def get_constraint():
        return MixedMMLUPromptSet.warn_and_return_empty()

    @staticmethod
    def get_format():
        return MixedMMLUPromptSet.warn_and_return_empty()


    @staticmethod
    def get_answer_prompt(question):
        return MixedMMLUPromptSet.warn_and_raise()

    @staticmethod
    def get_query_prompt(question):
        return MixedMMLUPromptSet.warn_and_raise()

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return MixedMMLUPromptSet.warn_and_raise()

    @staticmethod
    def get_websearch_prompt(query):
        return MixedMMLUPromptSet.warn_and_raise()

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return MixedMMLUPromptSet.warn_and_raise()

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        return MixedMMLUPromptSet.warn_and_raise()

    @staticmethod
    def get_reflect_prompt(question, answer):
        return MixedMMLUPromptSet.warn_and_raise()

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return MixedMMLUPromptSet.warn_and_raise()

    @staticmethod
    def get_task_with_hint(task, hint):
        return MixedMMLUPromptSet.warn_and_raise()