from experiments.evaluator.datasets.base_dataset import BaseDataset, SwarmInput
from experiments.evaluator.datasets.cmmlu_dataset import CMMLUDataset
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset


from typing import Union, List, Literal

MMLU_PROMPTS = True

class MixedMMLUDataset(BaseDataset):
    def __init__(self, split: Union[Literal['dev'], Literal['test']]) -> None:
        self.cmmlu_dataset = CMMLUDataset(split)
        self.mmlu_dataset = MMLUDataset(split)
        self._split = split
        print("cmmlu dataset length: ", len(self.cmmlu_dataset))
        print("mmlu dataset length: ", len(self.mmlu_dataset))
        # Balance the lengths of the datasets
        min_len = min(len(self.cmmlu_dataset), len(self.mmlu_dataset))
        self.cmmlu_dataset._total_df = self.cmmlu_dataset._total_df[:min_len]
        self.mmlu_dataset._total_df = self.mmlu_dataset._total_df[:min_len]
        print("cmmlu dataset length: ", len(self.cmmlu_dataset))
        print("mmlu dataset length: ", len(self.mmlu_dataset))
        
        self.counter = {"mmlu":0, "cmmlu":0}

    @staticmethod
    def get_domain() -> str:
        return 'mixed_mmlu'
    
    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self.cmmlu_dataset) + len(self.mmlu_dataset)

    def __getitem__(self, index: int) -> dict:
        if index < len(self.cmmlu_dataset):
            record = self.cmmlu_dataset[index]
            self.counter['cmmlu'] += 1
        else:
            record = self.mmlu_dataset[index - len(self.cmmlu_dataset)]
            self.counter['mmlu'] += 1
        return record

    @staticmethod
    def record_to_swarm_input(record: dict) -> SwarmInput:
        if not(MMLU_PROMPTS):
            return  CMMLUDataset.record_to_swarm_input(record)
        else:
            return MMLUDataset.record_to_swarm_input(record)

    @staticmethod
    def record_to_target_answer(record: dict) -> str:
        if not(MMLU_PROMPTS):
            return CMMLUDataset.record_to_target_answer(record)
        else:
            return MMLUDataset.record_to_target_answer(record)
        
    
        
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if not(MMLU_PROMPTS):
            return self.cmmlu_dataset.postprocess_answer(answer)
        else:
            return self.mmlu_dataset.postprocess_answer(answer)
        
    #static postprocess
    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        print(answer)
        if not(MMLU_PROMPTS):
            return CMMLUDataset.postprocess_answer(answer)
        else:
            return MMLUDataset.postprocess_answer(answer)
        
    def postprocess_answer_list(self, answer: Union[str, List[str]]) -> List[str]:
        if not(MMLU_PROMPTS):
            return self.cmmlu_dataset.postprocess_answer(answer)
        else:
            return self.mmlu_dataset.postprocess_answer(answer)