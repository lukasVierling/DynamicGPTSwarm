import asyncio
import os
import time

import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
from typing import Union, Literal, Optional
import argparse
import numpy as np

from swarm.graph.swarm import Swarm
from swarm.environment.operations.final_decision import MergingStrategy
from experiments.evaluator.evaluator import Evaluator
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download
from experiments.evaluator.datasets.cmmlu_dataset import CMMLUDataset
from experiments.evaluator.datasets.mixedmmlu_dataset import MixedMMLUDataset
from dataset.CMMLU.download import download as cmmlu_download


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='OptimizedSwarm',
                        choices=['DirectAnswer', 'FullConnectedSwarm', 'RandomSwarm', 'OptimizedSwarm'],
                        help="Mode of operation. Default is 'OptimizedSwarm'.")

    parser.add_argument('--num-truthful-agents', type=int, default=1,
                        help="Number of truthful agents. The total will be N truthful and M adversarial.")
    
    parser.add_argument('--num-adversarial-agents', type=int, default=1, 
                        help="Number of adversarial agents. The total will be N truthful and M adversarial.")

    parser.add_argument('--num-iterations', type=int, default=200,
                        help="Number of optimization iterations. Default 200.")

    parser.add_argument('--model_name', nargs='+', default=["google/gemma-7B-it"],
                    help="Model names, None runs the default ChatGPT4.") #Models: "vivo-ai/BlueLM-7B-Chat", "google/gemma-7B-it" , "google/gemma-2B-it"

    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="Set for a quick debug cycle")
    
    parser.add_argument('--edge_network_enable', action='store_true', default=False,
                        help="Enable edge network")
    

    parser.add_argument('--reproduce', action='store_true', default=False,
                        help="Set seed to 0 to have deterministic training data")
    
    parser.add_argument('--lr', type=float, default=0.0001, #0.1 for vector case
                        help="Learning rate for edge network optimization")
    
    parser.add_argument('--reduce_edges', action='store_true', default=False,)

    parser.add_argument('--delta', type=float, default=0.2, #0.1 for vector case
                        help="weight for edge reduction")
    
    parser.add_argument('--embedding_only', action='store_true', default=False,
                        help="Set for only embedding optimization")

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

    reproduce: bool = args.reproduce

    lr = args.lr

    reduce_edges = args.reduce_edges
    delta = args.delta

    embedding_only = args.embedding_only

    #Hardcoding herer TODO

    

    if mode == 'DirectAnswer':
        swarm_name = None
        swarm = None
    else:
        N = args.num_truthful_agents
        M = args.num_adversarial_agents
        agent_name_list = N * ["IO"] + M * ["AdversarialAgent"]

        #agent_name_list = ["COT"]

        swarm_name = f"{N}true_{M}adv_{'edge' if edge_network_enable else 'noedge'}_iter{args.num_iterations}_domain_{domain}"

        #swarm_name = "MMLUReflection_CoT_IO"


        swarm = Swarm(
            agent_name_list,
            domain,
            model_name=model_name,
            final_node_class="FinalDecision",
            final_node_kwargs=dict(strategy=strategy),
            edge_optimize=True,
            edge_network_enable=edge_network_enable,
            llm_backbone_name="google/gemma-2B",
            embedding_only=embedding_only
        )

    tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}_{time.time()}"

    if domain == 'mmlu':
        download()
        dataset_train = MMLUDataset('dev')
        dataset_val = MMLUDataset('val')

    elif domain == 'cmmlu':
        cmmlu_download()
        dataset_train = CMMLUDataset('dev')
        dataset_val = CMMLUDataset('test') #double check this if we should use dev and test
    elif domain == 'mixedmmlu':
        download()
        cmmlu_download()
        dataset_train = MixedMMLUDataset('dev')
        dataset_val = MixedMMLUDataset('test')
    else:
        raise Exception(f"Unsupported domain {domain}")

    evaluator = Evaluator(
        swarm,
        dataset_train,
        dataset_val,
        model_name=model_name,
        enable_tensorboard = mode=='OptimizedSwarm',
        enable_artifacts=True,
        tensorboard_tag=tag)

    limit_questions = 5 if debug else 1000

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
            #lr = 0.0001 #0.001
            print(f"Using lr {lr}")
        else: 
            lr = 0.1
            print("Using lr 0.1")
        if reproduce: # allow for reproducability between experiments
            np.random.seed(1) # Seed: 0
        edge_probs = await evaluator.optimize_swarm(num_iters=num_iters, lr=lr, edge_network_enable=edge_network_enable, reduce_edges=reduce_edges, delta=delta)
        mode = 'edge_network' if edge_network_enable else 'external_edge_probs_swarm'
        score = 0
        
        score = await evaluator.evaluate_swarm(
            mode=mode,
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            edge_network_enable=edge_network_enable
        )
        

        #store the swarm in result folder
        if domain == "mixedmmlu":
            print(dataset_train.counter)
        
    else:
        raise Exception(f"Unsupported mode {mode}")
    
    if not os.path.exists("result/mmlu"):
        os.makedirs("result/mmlu")
    with open(f"result/mmlu/{tag}.pkl", "wb") as f:
        pickle.dump(swarm, f) #store the swarm in result folder
        #also dump the score
        pickle.dump(score, f)

    print(f"Score: {score}")
    #print the final edge probs
    print(f"Edge Probs: {edge_probs}")

    #calculate mean and variance for edge probs over 100 samples of the test set
    edge_probs_samples = []
    



if __name__ == "__main__":
    asyncio.run(main())
