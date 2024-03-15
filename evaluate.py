import json
from tqdm import tqdm
import asyncio
import numpy as np
from copy import deepcopy
import pickle
import torch
import sys
import random

from swarm.environment.domain.crosswords.env import MiniCrosswordsEnv
from swarm.environment.agents.agent_registry import AgentRegistry
from swarm.graph.swarm import Swarm
from swarm.optimizer.edge_optimizer.optimization import optimize
from swarm.environment.domain.crosswords.evaluator import CrosswordsEvaluator


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


if __name__ == "__main__":
    file_path = "datasets/crosswords/mini0505_0_100_5.json"
    with open(file_path, "r") as file:
        test_data = json.load(file)

    experiment_id = "experiment1"
    init_connection_probability = .1
    epochs = 1
    batch_size = 4
    use_learned_order = True
    edge_network_enable = True
    llm_backbone_name = "GPT2"
    num_batches = int(len(test_data) / batch_size)
    evaluator = CrosswordsEvaluator(test_data, batch_size=batch_size, metric="words", window_size=num_batches)

    swarm = Swarm(["CrosswordsReflection", "CrosswordsToT", "CrosswordsBruteForceOpt"], "crosswords", "gpt-3.5-turbo-1106", #"gpt-4-1106-preview"
                final_node_class="ReturnAll", 
                final_node_kwargs={},
                edge_optimize=True,
                init_connection_probability=init_connection_probability, 
                connect_output_nodes_to_final_node=True, 
                include_inner_agent_connections=True,
                edge_network_enable=edge_network_enable,
                llm_backbone_name=llm_backbone_name)
    #swarm.connection_dist.load_state_dict(torch.load(f"result/crosswords_Jan15/{experiment_id}_edge_logits_{int(epochs * len(test_data) / batch_size) - 1}.pkl"))
    swarm.connection_dist.load_state_dict(torch.load(f"result/crosswords/experiment_edge_logits_10.pt"))
    '''
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
    #print the weights and bias of last linear layer of the EdgeNetwork in the connectiON-dis
    
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
    loop = asyncio.get_event_loop()
    utilities = []
    evaluator.reset()
    for k in range(num_batches):
        print("batch: ",k)
        utilities += batched_evaluator_withEdgeNetwork(evaluator, batch_size, swarm, loop)
        print(f"avg. utility = {np.mean(utilities):.3f}")
    print(f"avg. utility = {np.mean(utilities):.3f}")
    with open(f"result/crosswords/{experiment_id}_final_utilities.pkl", "wb") as file:
        pickle.dump(utilities, file)

    