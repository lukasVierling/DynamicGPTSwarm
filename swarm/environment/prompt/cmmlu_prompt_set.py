#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from swarm.environment.prompt.prompt_set import PromptSet
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.prompt.common import get_combine_materials


@PromptSetRegistry.register('cmmlu')
class CMMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def get_role():
        return "一个精通问题回答的专家"

    @staticmethod
    def get_constraint():
        return """我将向您提问。
        我还会给出4个答案，分别标记为A、B、C和D。
        这4个答案中只有一个是正确的。
        您必须选择正确的答案。
        您的回答应该只包含一个字母（A、B、C或D）。
        除了字母以外，您的回答中不要包含任何其他信息。
        """
    
    '''
            您的响应应该采用JSON格式，键名为"答案"，值为A、B、C或D之一，对应于正确答案。
        这是一个正确格式的示例：
        {
            "答案": "A"
        }
        '''

    @staticmethod
    def get_format():
        return "其中一个字母：A、B、C或D"

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
        return f"""对以下问题回答一个错误的答案：{question}。"""

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        return f"""对以下问题和答案进行思考：
        问题：{question}
        答案：{answer}
        对于答案的正确性和准确性，您有什么想法？您同意还是不同意？为什么？请给出简要解释。"""

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def get_task_with_hint(task, hint):
        return f"{task}。对先前输出的反思：{hint}"