import asyncio
import os
import time

import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
import torch
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

from swarm.llm.custom_llm import CustomLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='OptimizedSwarm',
                        choices=['DirectAnswer', 'FullConnectedSwarm', 'RandomSwarm', 'OptimizedSwarm'],
                        help="Mode of operation. Default is 'OptimizedSwarm'.")
    
    parser.add_argument('--num-useless-agents', type=int, default=1,
                        help="Number of useless agents. The total will be N truthful and M adversarial.")

    parser.add_argument('--num-truthful-agents', type=int, default=1,
                        help="Number of truthful agents. The total will be N truthful and M adversarial.")

    parser.add_argument('--num-iterations', type=int, default=200,
                        help="Number of optimization iterations. Default 200.")

    parser.add_argument('--model_name', nargs='+', default=["google/gemma-7B-it"],
                    help="Model names, None runs the default ChatGPT4.") #Models: "vivo-ai/BlueLM-7B-Chat", "google/gemma-7B-it" , "google/gemma-2B-it", "meta-llama/Meta-Llama-3-8B-Instruct"

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

    parser.add_argument("--reduce_cost" , action='store_true', default=False,
                        help="Reduce cost for edge reduction")
    
    parser.add_argument("--epsilon" , type=float, default=1,
                        help="Epsilon for cost reduction")
    
    args = parser.parse_args()
    return args


async def main():
    def _print_conns(edge_probs: torch.Tensor, swarm):
        print(edge_probs)
        msgs = []
        for i_conn, (conn, prob) in enumerate(zip(
                swarm.connection_dist.potential_connections, edge_probs)):
            print("conn und prob: ",i_conn, conn, prob.item())
            src_id, dst_id = conn
            src_node = swarm.composite_graph.find_node(src_id)
            dst_node = swarm.composite_graph.find_node(dst_id)
            src_node_name = src_node.model_name if hasattr(src_node,"model_name") else src_node.node_name
            dst_node_name = dst_node.model_name if hasattr(dst_node,"model_name") else dst_node.node_name
            msg = (f"{i_conn}: src={src_node_name}({src_node.id}), "
                    f"dst={dst_node_name}({dst_node.id}), prob={prob.item():.3f}")
            msgs.append(msg+"\n")
            print(msg)

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

    reduce_edges = args.reduce_edges #additional loss for small graphs

    delta = args.delta #loss weight

    reduce_cost = args.reduce_cost

    epsilon = args.epsilon

    #Hardcoding herer TODO


    n = args.num_truthful_agents
    m = args.num_useless_agents
    
    #agent_name_list = n*["IO"] + m*["EmptyAgent"]

    agent_name_list = ["IO","IO"]
    price_list = {"google/gemma-7B-it": {"input":0.5, "output":1.5}, "google/gemma-2B-it": {"input":0.5, "output":1.5}, "vivo-ai/BlueLM-7B-Chat": {"input":0.5, "output":1.5}, "meta-llama/Meta-Llama-3-8B-Instruct": {"input":2.0, "output":6.0}}
    swarm = Swarm(
        agent_name_list,
        domain,
        model_name=model_name,
        final_node_class="FinalDecision",
        final_node_kwargs=dict(strategy=strategy),
        edge_optimize=True,
        edge_network_enable=edge_network_enable,
        llm_backbone_name="google/gemma-2B",
        price_list =price_list
    )

    swarm_name = f"COT_IO_save_compute_{delta}_{epsilon}_{'edge' if edge_network_enable else 'noedge'}_iter{args.num_iterations}_domain_{domain}"

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
        print(reduce_cost, reduce_edges)
        edge_probs = await evaluator.optimize_swarm(num_iters=num_iters, lr=lr, edge_network_enable=edge_network_enable, reduce_edges=reduce_edges, delta=delta, reduce_cost=reduce_cost, epsilon=epsilon)
        mode = 'edge_network' if edge_network_enable else 'external_edge_probs_swarm'
        score = 0
        
        #Evaluate edge probs
        if edge_network_enable:
            dataset_val = MMLUDataset('test')
            dataset_val._total_df = dataset_val._total_df[:100]
            all_probs = []
            swarm.connection_dist.model.eval()
            for record in dataset_val:
                #print(record)
                input_dict = dataset_val.record_to_swarm_input(record)
                edge_probs = swarm.connection_dist.get_edge_probs(swarm.composite_graph, inputs=input_dict)
                #log_probs = torch.tensor(log_probs)
                #print(log_probs.shape)
                #print(swarm.connection_dist.potential_connections.shape)
                #print(log_probs)
                #edge_probs = torch.sigmoid(torch.tensor(log_probs))
                #_print_conns(edge_probs,swarm)
                all_probs.append(edge_probs)
            all_probs = torch.stack(all_probs)
            print(all_probs[:][0])
            all_probs_variance = torch.std(all_probs, dim=0)
            print(all_probs_variance)
            all_probs_mean = torch.mean(all_probs, dim=0)
            print(all_probs_variance.shape)
            print(all_probs_mean.shape)
            print("Mean: ")
            _print_conns(all_probs_mean, swarm)
            print("Variance: ")
            _print_conns(all_probs_variance, swarm)
        else:
            _print_conns(torch.sigmoid(swarm.connection_dist.edge_logits), swarm)
        '''
        CustomLLM.start_counter()
        score = await evaluator.evaluate_swarm(
            mode=mode,
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            edge_network_enable=edge_network_enable
        )
        total_cost = CustomLLM.get_price(price_list)
        print("total cost: ", total_cost)

        #store the swarm in result folder
        if domain == "mixedmmlu":
            print(dataset_train.counter)
        '''
        
    else:
        raise Exception(f"Unsupported mode {mode}")
    
    if not os.path.exists("result/mmlu"):
        os.makedirs("result/mmlu")
    with open(f"result/mmlu/{tag}.pkl", "wb") as f:
        pickle.dump(swarm, f) #store the swarm in result folder
        #also dump the score
        pickle.dump(score, f)
        pickle.dump(total_cost, f)

    print(f"Score: {score}")
    #print the final edge probs
    print(f"Edge Probs: {edge_probs}")

    #calculate mean and variance for edge probs over 100 samples of the test set
    edge_probs_samples = []
    



if __name__ == "__main__":
    asyncio.run(main())
