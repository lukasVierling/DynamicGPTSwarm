import asyncio
import os
import time

import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"
from typing import Union, Literal, Optional
import argparse

from swarm.graph.swarm import Swarm
from swarm.environment.operations.final_decision import MergingStrategy
from experiments.evaluator.evaluator import Evaluator
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='OptimizedSwarm',
                        choices=['DirectAnswer', 'FullConnectedSwarm', 'RandomSwarm', 'OptimizedSwarm'],
                        help="Mode of operation. Default is 'OptimizedSwarm'.")

    parser.add_argument('--num-truthful-agents', type=int, default=1,
                        help="Number of truthful agents. The total will be N truthful and N adversarial.")

    parser.add_argument('--num-iterations', type=int, default=200,
                        help="Number of optimization iterations. Default 200.")

    parser.add_argument('--model_name', type=str, default="google/gemma-7B-it",
                        help="Model name, None runs the default ChatGPT4.")

    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="Set for a quick debug cycle")
    
    parser.add_argument('--edge_network_enable', action='store_true', default=False,
                        help="Enable edge network")

    args = parser.parse_args()
    return args


async def main():

    args = parse_args()

    debug: bool = args.debug

    model_name: Optional[str] = args.model_name

    edge_network_enable = args.edge_network_enable


    mode: Union[Literal['DirectAnswer'],
                Literal['FullConnectedSwarm'],
                Literal['RandomSwarm'],
                Literal['OptimizedSwarm']]

    mode = args.mode

    strategy = MergingStrategy.MajorityVote

    domain: str = args.domain

    if mode == 'DirectAnswer':
        swarm_name = None
        swarm = None
    else:
        N = args.num_truthful_agents
        M = N
        agent_name_list = N * ["IO"] + M * ["AdversarialAgent"]

        #agent_name_list = ["COT"]

        swarm_name = f"{N}true_{M}adv"

        #swarm_name = "MMLUReflection_CoT_IO"


        swarm = Swarm(
            agent_name_list,
            domain,
            model_name=model_name,
            final_node_class="FinalDecision",
            final_node_kwargs=dict(strategy=strategy),
            edge_optimize=True,
            edge_network_enable=edge_network_enable,
            llm_backbone_name="google/gemma-2B"
        )

    tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}_{time.time()}"

    download()

    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')

    evaluator = Evaluator(
        swarm,
        dataset_train,
        dataset_val,
        model_name=model_name,
        enable_tensorboard = mode=='OptimizedSwarm',
        enable_artifacts=True,
        tensorboard_tag=tag)

    limit_questions = 5 if debug else 153

    if mode == 'DirectAnswer':
        score = await evaluator.evaluate_direct_answer(
            limit_questions=limit_questions)
    elif mode == 'FullConnectedSwarm':
        score = await evaluator.evaluate_swarm(
            mode='full_connected_swarm',
            limit_questions=limit_questions)
    elif mode == 'RandomSwarm':
        score = await evaluator.evaluate_swarm(
            mode='randomly_connected_swarm',
            limit_questions=limit_questions)
    elif mode == 'OptimizedSwarm':

        num_iters = 3 if debug else args.num_iterations
        if edge_network_enable:
            lr = 0.0001 #0.001
        else: 
            lr = 0.1

        edge_probs = await evaluator.optimize_swarm(num_iters=num_iters, lr=lr, edge_network_enable=edge_network_enable)
        mode = 'edge_network' if edge_network_enable else 'external_edge_probs_swarm'
        score = await evaluator.evaluate_swarm(
            mode=mode,
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            edge_network_enable=edge_network_enable
        )
        #store the swarm in result folder
        
    else:
        raise Exception(f"Unsupported mode {mode}")
    
    if not os.path.exists("result/mmlu"):
        os.makedirs("result/mmlu")
    with open(f"result/mmlu/{tag}.pkl", "wb") as f:
        pickle.dump(swarm, f) #store the swarm in result folder

    print(f"Score: {score}")


if __name__ == "__main__":
    asyncio.run(main())
