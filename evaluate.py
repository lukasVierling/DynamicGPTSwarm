import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
from tqdm import tqdm
import asyncio
import numpy as np
from copy import deepcopy
import pickle
import torch
import sys
import random
import argparse

from swarm.environment.domain.crosswords.env import MiniCrosswordsEnv
from swarm.environment.agents.agent_registry import AgentRegistry
from swarm.graph.swarm import Swarm
from swarm.optimizer.edge_optimizer.optimization import optimize
from swarm.environment.domain.crosswords.evaluator import CrosswordsEvaluator

def my_load(swarm, path, edge_network_enable):
    if not(edge_network_enable):
        swarm.connection_dist.load_state_dict(torch.load(path))
    else:
        pretrained_dict = torch.load(path, map_location=swarm.connection_dist.model.device)
        model_dict = swarm.connection_dist.state_dict()
        print(model_dict.keys())
        print(pretrained_dict.keys())
        # 1. filter out unnecessary keys
        #pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in ['model.linear.weight','model.linear.bias']} relevant if we load full swarm
        # 2. overwrite entries in the existing state dict
        #save the old linear weights to compare if they actually got updated
        pretrained_dict = {"model.linear.weight":pretrained_dict["weight"],"model.linear.bias":pretrained_dict["bias"]}
        model_dict.update(pretrained_dict) 
        #print first 10 entries of both 
        # 3. load the new state dict
        swarm.connection_dist.load_state_dict(model_dict)


def batched_evaluator(evaluator, batch_size, graph, loop):
    tasks = []
    for _ in range(batch_size):
        tasks.append(evaluator.evaluate(deepcopy(graph)))
    return loop.run_until_complete(asyncio.gather(*tasks))

def batched_evaluator_withEdgeNetwork(evaluator, batch_size, swarm, loop):
    tasks = []
    for _ in range(batch_size):
        tasks.append(evaluator.evaluateWithEdgeNetwork(swarm))

    results = loop.run_until_complete(asyncio.gather(*tasks))
    scores = [result[0] for result in results]
    return scores

class DebugArgs:
    id = 9
    edge_network_enable = True
    pretrained_path = "result/crosswords/new_method_40_it_nostuck_5batch/experiment8_edge_logits_39.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--id" , type=int, default=0)
    parser.add_argument("--edge_network_enable", action="store_true", default=False)
    parser.add_argument("--pretrained_path", type=str, default=None)
    
    n_samples = 3

    args = parser.parse_args()
    args = DebugArgs()
    #hardcoding

    file_path = "dataset/crosswords/mini0505_0_100_5.json"
    with open(file_path, "r") as file:
        test_data = json.load(file)

    experiment_id = f"experiment{args.id}"
    init_connection_probability = .1
    epochs = 1
    batch_size = len(test_data)
    use_learned_order = False
    edge_network_enable = args.edge_network_enable
    llm_backbone_name = "google/gemma-2B"
    num_batches = 1
    evaluator = CrosswordsEvaluator(test_data, batch_size=batch_size, metric="words", window_size=num_batches)
    swarm = Swarm(["CrosswordsBruteForceOpt","CrosswordsReflection"], "crosswords", "meta-llama/Meta-Llama-3-8B-Instruct",#"google/gemma-7B-it",#,#"gpt-3.5-turbo-1106", #"gpt-4-1106-preview" ,  #"CrosswordsToT","CrosswordsBruteForceOpt","CrosswordsReflection"
                final_node_class="ReturnAll", 
                final_node_kwargs={},
                edge_optimize=True,
                init_connection_probability=init_connection_probability, 
                connect_output_nodes_to_final_node=True, 
                include_inner_agent_connections=True,
                edge_network_enable=edge_network_enable,
                llm_backbone_name=llm_backbone_name)
    #swarm.connection_dist.load_state_dict(torch.load(f"result/crosswords_Jan15/{experiment_id}_edge_logits_{int(epochs * len(test_data) / batch_size) - 1}.pkl"))
    my_load(swarm,args.pretrained_path, edge_network_enable)
    swarm.connection_dist.eval()
    loop = asyncio.get_event_loop()
    utilities = []
    evaluator.reset()
    num_edges = []

    average_adj_matrix = None
    for _ in range(100):
        graph = asyncio.run(evaluator.evaluateWithEdgeNetwork(swarm=swarm,return_moving_average = True, use_learned_order = False, evaluate_graph = False))
        if average_adj_matrix is None:
            average_adj_matrix = graph.adj_matrix
        else:
            average_adj_matrix += graph.adj_matrix
            
        num_edges.append(graph.num_edges)
    #wait for evaluate functions to terminate
    print(average_adj_matrix/100)

    '''
    if edge_network_enable:
        probs = []
        #let's print the average edge probabilites
        for i in range(len(test_data)):
            probs.append(evaluator.get_edge_probs(swarm))
        print("Mean: ",torch.mean(probs))
        print("Std: ",torch.std(probs))
    '''
    
    results = {
        'utilities': [],
        'mean_utilities_per_run': [],
        'overall_mean': None,
        'overall_std': None
    }

    for run in range(n_samples):
        graph = None
        current_util = []
        if not(edge_network_enable):
            graph = swarm.connection_dist.realize(swarm.composite_graph, use_learned_order=use_learned_order)[0] #TODO outside of the loop
        for k in range(num_batches):
            print("batch: ",k)
            if edge_network_enable:
                current_util += batched_evaluator_withEdgeNetwork(evaluator, batch_size, swarm, loop)
            else:
                current_util += batched_evaluator(evaluator, batch_size, graph, loop)
        results['utilities'] += current_util
        mean_utility = np.mean(current_util)
        results['mean_utilities_per_run'].append(mean_utility)
        print(f"avg. utility = {mean_utility:.3f}")

    results['overall_mean'] = np.mean(results['mean_utilities_per_run'])
    results['overall_std'] = np.std(results['mean_utilities_per_run'])

    print(f"Mean over all runs = {results['overall_mean']:.3f}")
    print(f"Std over all runs = {results['overall_std']:.3f}")          

    
    with open(f"result/crosswords/{experiment_id}_edge_network-{edge_network_enable}_final_utilities.pkl", "wb") as file:
        pickle.dump(results, file)

    
    #print the weights and bias of last linear layer of the EdgeNetwork in the connectiON-dis
    '''
    print("----- Model Weights ----")
    print(swarm.connection_dist.model.linear.bias)
    print("----- Model Bias -----")
    print(swarm.connection_dist.model.linear.weight)
    num_edges = int(np.array(num_edges).mean())
    print(f"Expected number of edges: {num_edges}")
  
        graphs = [
                swarm.connection_dist.random_sample_num_edges(swarm.composite_graph, num_edges),
                swarm.connection_dist.realize(swarm.composite_graph, threshold=init_connection_probability, use_learned_order=use_learned_order)[0],
                swarm.connection_dist.realize(swarm.composite_graph, use_learned_order=use_learned_order)[0],
                swarm.composite_graph,
                ]

    

    loop = asyncio.get_event_loop()
    for i, graph in tqdm(enumerate(graphs)):
        print(f"{graph.num_edges} edges")
        utilities = []
        evaluator.reset()
        for k in range(num_batches):
            utilities += batched_evaluator(evaluator, batch_size, graph, loop)
        print(f"avg. utility = {np.mean(utilities):.3f}")
        with open(f"result/crosswords/{experiment_id}_final_utilities_{i}.pkl", "wb") as file:
            pickle.dump(utilities, file)
    '''


    