import os
import asyncio
import pandas as pd
from typing import Iterable, Optional, Iterator, Union, Literal, List, Dict, Any
from tqdm import tqdm
import torch
import time
import datetime
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import json
import math
import random

from swarm.graph import Graph
from swarm.environment.agents import IO
from swarm.graph.swarm import Swarm
from experiments.evaluator.datasets.base_dataset import BaseDataset
from experiments.evaluator.accuracy import Accuracy
from swarm.environment.agents import AgentRegistry


class Evaluator():
    def __init__(
            self,
            swarm: Optional[Swarm],
            train_dataset: BaseDataset,
            val_dataset: BaseDataset,
            model_name: Optional[str] = None,
            enable_tensorboard: bool = False,
            enable_artifacts: bool = False,
            tensorboard_tag: Optional[str] = None,
        ) -> None:

        self._swarm: Optional[Swarm] = swarm
        self._train_dataset: BaseDataset = train_dataset
        self._val_dataset: BaseDataset = val_dataset
        self._model_name: Optional[str] = model_name

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        art_dir_name = (f"{datetime_str}" +
                        (f"_{tensorboard_tag}" if tensorboard_tag is not None else ""))

        if enable_artifacts or enable_tensorboard:
            self._art_dir_name = os.path.join("runs", art_dir_name)
            os.makedirs(self._art_dir_name, exist_ok=True)
        else:
            self._art_dir_name = None

        if enable_tensorboard:
            self._logger = SummaryWriter(log_dir=self._art_dir_name)
        else:
            self._logger = None

    async def evaluate_agent(self,
            limit_questions: Optional[int] = None,
            agent: str = "DirectAnswer",
            ) -> float:

        dataset = self._val_dataset

        print(f"Evaluating DirectAnswer on {dataset.get_domain()} split {dataset.split}")
        
        agent = AgentRegistry.get(agent,dataset.get_domain(), self._model_name)

        accuracy = Accuracy()

        for i_question, record in tqdm(enumerate(dataset)):
            print(80*'-')
            if limit_questions is not None:
                if i_question >= limit_questions:
                    break

            input_dict = dataset.record_to_swarm_input(record)
            #print(input_dict)

            raw_answer = await agent.run(input_dict)

            print("Raw answer:", raw_answer)
            try:
                answer = dataset.postprocess_answer(raw_answer)
            except Exception as e:
                print("Exception while postprocessing answer:", e)
                answer = ['']
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            accuracy.update(answer, correct_answer)
            accuracy.print()

        print("Final accuracy:")
        accuracy.print()

        self._dump_eval_results(dict(
            accuracy=accuracy.get(),
            limit_questions=limit_questions))

        print("Done!")
        return accuracy.get()

    async def evaluate_swarm(
            self,
            mode: Union[
                Literal['full_connected_swarm'],
                Literal['randomly_connected_swarm'],
                Literal['external_edge_probs'],
                ],
            edge_probs: Optional[torch.Tensor] = None,
            limit_questions: Optional[int] = None,
            eval_batch_size: int = 4,
            edge_network_enable: bool = False,
            ) -> float:

        assert self._swarm is not None

        #edge network in eval mode
        if edge_network_enable:
            self._swarm.connection_dist.model.eval()

        dataset = self._val_dataset

        print(f"Evaluating swarm on {dataset.__class__.__name__} split {dataset.split}")

        realized_graph: Optional[Graph]
        if mode == 'full_connected_swarm':
            realized_graph = self._swarm.connection_dist.realize_full(self._swarm.composite_graph)
        elif mode == 'external_edge_probs':
            assert edge_probs is not None
            edge_mask = edge_probs > 0.5
            print("Edge Mask:", edge_mask)
            realized_graph = self._swarm.connection_dist.realize_mask(self._swarm.composite_graph, edge_mask)
            realized_graph.display()
        else:
            realized_graph = None

        accuracy = Accuracy()

        def eval_loader(batch_size: int) -> Iterator[List[Any]]:
            records = []
            for i_record, record in enumerate(dataset):
                if limit_questions is not None:
                    if i_record >= limit_questions:
                        break
                records.append(record)
                if len(records) >= batch_size:
                    yield records
                    records = []
            if len(records) > 0:
                yield records
            return

        data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
        num_batches = int(math.ceil(data_len / eval_batch_size))

        for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
            print(80*'-')

            start_ts = time.time()

            future_answers = []
            for record in record_batch:
                if mode == 'randomly_connected_swarm':
                    realized_graph, _ = self._swarm.connection_dist.realize(self._swarm.composite_graph)
                #assert realized_graph is not None

                input_dict = dataset.record_to_swarm_input(record)
                #print("input_dict:", input_dict)
                if edge_network_enable:
                    realized_graph, log_prob, edge_probs = self._swarm.connection_dist.realize(self._swarm.composite_graph, inputs=input_dict)
                else:
                    realized_graph, log_prob = self._swarm.connection_dist.realize(
                        self._swarm.composite_graph,
                        # temperature=3.0, # DEBUG
                        )
                #print(input_dict)

                future_answer = self._swarm.arun(input_dict, realized_graph)
                future_answers.append(future_answer)

            raw_answers = await asyncio.gather(*future_answers)
            print(f"Batch time {time.time() - start_ts:.3f}")

            for raw_answer, record in zip(raw_answers, record_batch):
                print("Raw answer:", raw_answer)
                
                answer = dataset.postprocess_answer(raw_answer)
                print("Postprocessed answer:", answer)
                correct_answer = dataset.record_to_target_answer(record)
                print("Correct answer:", correct_answer)
                accuracy.update(answer, correct_answer)
                accuracy.print()

        accuracy.print()
        print("Done!")
        
        self._dump_eval_results(dict(
            accuracy=accuracy.get(),
            limit_questions=limit_questions))

        return accuracy.get()

    def _dump_eval_results(self, dct: Dict[str, Any]) -> None:
        if self._art_dir_name is not None:
            eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
            with open(eval_json_name, "w") as f:
                json.dump(dct, f)

    def _print_conns(self, edge_probs: torch.Tensor, save_to_file: bool = False):
        assert self._swarm is not None
        msgs = []
        for i_conn, (conn, prob) in enumerate(zip(
                self._swarm.connection_dist.potential_connections, edge_probs)):
            src_id, dst_id = conn
            src_node = self._swarm.composite_graph.find_node(src_id)
            dst_node = self._swarm.composite_graph.find_node(dst_id)
            msg = (f"{i_conn}: src={src_node.model_name if hasattr(src_node, 'model_name') else src_node.node_name}({src_node.id}), "
                f"dst={dst_node.model_name if hasattr(dst_node, 'model_name') else dst_node.node_name}({dst_node.id}), prob={prob.item():.3f}")
            print(msg)
        if save_to_file:
            if self._art_dir_name is not None:
                txt_name = os.path.join(self._art_dir_name, "connections.txt")
                with open(txt_name, "w") as f:
                    f.writelines(msgs)

    async def optimize_swarm(
            self,
            num_iters: int,
            lr: float,
            batch_size: int = 4,
            edge_network_enable: bool = False,
            reduce_edges: bool =False,
            delta: float = 0.2,
            reduce_cost: bool=False,
            epsilon: float = 0.1,
            ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset

        print(f"Optimizing swarm on {dataset.__class__.__name__} split {dataset.split}")

        optimizer = torch.optim.Adam(self._swarm.connection_dist.parameters(), lr=lr)

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(lr=lr,
                               batch_size=batch_size,
                               num_iters=num_iters,
                               model_name=self._model_name
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        #to normalize the costs we want to compute the highest amount of cost possible
        random_idx = random.randint(0,len(dataset)-1)
        if reduce_cost:
            edge_probs = torch.ones(len(self._swarm.connection_dist.potential_connections))
            realized_graph = self._swarm.connection_dist.realize_mask(self._swarm.composite_graph, edge_probs)
            _,max_cost = await self._swarm.arun(dataset.record_to_swarm_input(dataset[random_idx]), realized_graph, return_cost=False, return_max_cost=True)

            print("max:", max_cost)

            print("We normalize the graph costs with: ", max_cost)


        edge_probs = None
        for i_iter in range(num_iters):
            print(f"Iter {i_iter}", 80*'-')

            start_ts = time.time()

            future_answers = []
            log_probs = []
            correct_answers = []
            edge_probs = []
            costs = []
            for i_record, record in zip(range(batch_size), loader):
                input_dict = dataset.record_to_swarm_input(record)
                if edge_network_enable:
                    realized_graph, log_prob, edge_prob = self._swarm.connection_dist.realize(self._swarm.composite_graph, inputs=input_dict)
                else:
                    realized_graph, log_prob = self._swarm.connection_dist.realize(
                        self._swarm.composite_graph,
                        # temperature=3.0, # DEBUG
                        )
                    edge_prob = torch.sigmoid(self._swarm.connection_dist.edge_logits)
                answer = self._swarm.arun(input_dict, realized_graph, return_cost=reduce_cost) #add dataset for further processing in later steps
                future_answers.append(answer)
                log_probs.append(log_prob)
                edge_probs.append(edge_prob)
                correct_answer = dataset.record_to_target_answer(record) #should work the same way for both datasets
                correct_answers.append(correct_answer)


            # With this loop:
            raw_answers = []
            for future in future_answers:
                raw_answer = await future
                raw_answers.append(raw_answer)
            print(raw_answers)
            #raw_answers = await asyncio.gather(*future_answers)
            costs = [raw_answer[1] if reduce_cost else 0 for raw_answer in raw_answers]
            print(costs)
            raw_answers = [raw_answer[0] for raw_answer in raw_answers]
            print(f"Batch time {time.time() - start_ts:.3f}")

            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            for raw_answer, log_prob, correct_answer, cost in zip(raw_answers, log_probs, correct_answers, costs):
                
                print("Raw answer:", raw_answer)
                print("Correct answer:", correct_answer)
                answer = dataset.postprocess_answer(raw_answer)
                print("Postprocessed answer:", answer)
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                if reduce_cost:
                    utility -= min(epsilon * cost/max_cost,0.99)#pow(len(self._swarm.composite_graph.nodes),2) #run a fully connected graph once and get a good approximation of the max price for nomrlaization
                    print("cost: ", cost)
                    print("normalized cost: ", cost/max_cost)
                    utility = max(0,utility)
                    #give a small reward if correct because an expensive correct answer is better than a wrong
                single_loss = - log_prob * utility
                utilities.append(utility)
                loss_list.append(single_loss)
            

            print("utilities:", utilities)
            mean_utility = np.mean(np.array(utilities))
            total_loss = torch.mean(torch.stack(loss_list))
            if reduce_edges:
                total_loss += delta * torch.sum(torch.abs(torch.stack(edge_probs)))/len(edge_probs) #calc the average edge prob and subtract it as part of the loss

            print("loss:", total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            if edge_network_enable:
                print("Grad: ", self._swarm.connection_dist.model.linear.weight.grad)
                print("Grad: ", self._swarm.connection_dist.model.linear.bias.grad)
                optimizer.step()
            else:
                print("Grad:", self._swarm.connection_dist.edge_logits.grad)

                print("edge_logits:", self._swarm.connection_dist.edge_logits)
                optimizer.step()
                
                edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
                print("edge_probs:", edge_probs)

                self._print_conns(edge_probs)

            if self._logger is not None:
                self._logger.add_scalar("train/loss", total_loss.item(), i_iter)
                self._logger.add_scalar("train/utility", mean_utility.item(), i_iter)
            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(iter=i_iter, train_loss=total_loss.item(), train_utility=mean_utility.item()), f)
                    f.write("\n")
            print("end of iteration")

        if edge_probs is not None and not(edge_network_enable):
            self._print_conns(edge_probs, save_to_file=True)

        print("Done!")
        if not(edge_network_enable):
            print("Edge probs:", edge_probs)
            edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
            return edge_probs
        else:
            return None
