#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from swarm.environment.prompt.prompt_set import PromptSet
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.prompt.common import get_combine_materials


@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def get_role():
        return "a knowlegable expert in question answering"

    @staticmethod
    def get_constraint():
        return """I will ask you a question. 
        I wil lalso give you 4 answers enumerated as A, B, C, and D.
        Only one answer out of the offered 4 is correct. 
        You must choose the correct answer to the question.
        Answer with only a single letter (A, B, C, or D).
        Do not include any other information in your answer except the letter.
        Your response should be in Jason format, with the key "answer" and the value being one of the 4 letters: A, B, C, or D, corresponding to the correct answer.
        Here is an example of the correct format:
        {
            "answer": "A"
        }
"""
    
    
        """
            I will ask you a question.
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must be one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            Answer with only a single letter (A,B,C or D).
            Do not include any other information in your answer except the letter.
        """

    @staticmethod
    def get_format():
        return "one of the letters: A, B, C or D"

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Answer a lie to the following question: {question}. """

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        return f"""Reflect on the following question and answer:
        Question: {question}
        Answer: {answer}
        What are your thoughts on the correctness and accuracy of the answer? Do you agree or disagree? Why? Please provide a brief explanation."""

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_task_with_hint(task, hint):
        return f"{task}. Reflection on previous output: {hint}"
