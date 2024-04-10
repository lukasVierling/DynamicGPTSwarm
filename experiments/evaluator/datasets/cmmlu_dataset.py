import glob
import pandas as pd
from typing import Union, List, Literal
import numpy as np
import json
import re

from experiments.evaluator.datasets.base_dataset import BaseDataset, SwarmInput


class CMMLUDataset(BaseDataset):
    def __init__(self,
        split: Union[Literal['dev'],Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"dataset/CMMLU/data/{self._split}/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'cmmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names)
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_swarm_input(record: pd.DataFrame) -> SwarmInput:
        demo_question = (
            f"{record['question']}\n"
            f"选项A:{record['A']}\n"
            f"选项B:{record['B']}\n"
            f"选项C:{record['C']}\n"
            f"选项D:{record['D']}\n"
        )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")

        if len(answer) > 0:
            letters = re.findall('[A-D]', answer)
            if len(letters) != 0:
                letter = letters[0]
            else:
                letter = ""  # or some default value
        return letter
    
    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        letter = ""
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            letters = re.findall('[A-D]', answer)
            if len(letters) != 0:
                letter = letters[0]
            else:
                letter = ""  # or some default value
        return letter
    

    def postprocess_answer_list(self, answer: Union[str, List[str]]) -> List[str]:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")

        if len(answer) > 0:
            letters = re.findall('[A-D]', answer)
            if len(letters) != 0:
                letter = letters[0]
            else:
                letter = ""  # or some default value
        return letter

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
