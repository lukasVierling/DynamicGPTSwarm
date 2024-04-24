import json
from tqdm import tqdm
import asyncio
import numpy as np
from copy import deepcopy
import pickle
import torch
import sys
import random
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

from swarm.environment.domain.crosswords.env import MiniCrosswordsEnv
from swarm.environment.agents.agent_registry import AgentRegistry
from swarm.graph.swarm import Swarm
from swarm.optimizer.edge_optimizer.optimization import optimize
from swarm.environment.domain.crosswords.evaluator import CrosswordsEvaluator

#from huggingface_hub import login


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--id" , type=int, default=0)    
    parser.add_argument('--edge_network_enable', action='store_true', default=False,
                        help="Enable edge network")
    parser.add_argument('--pretrained_path', type=str, default=None,)

    args = parser.parse_args()

    id = args.id
    experiment_id = f"experiment{id}"
    torch.manual_seed(id)
    np.random.seed(id)
    random.seed(id)
    
    print(experiment_id)

    file_path = "dataset/crosswords/mini0505_0_100_5.json"
    with open(file_path, "r") as file:
        test_data = json.load(file)

    init_connection_probability = .1
    batch_size = 5
    use_learned_order = False
    include_inner_agent_connections = True
    connect_output_nodes_to_final_node = True
    window_size = 10
    edge_network_enable = args.edge_network_enable
    llm_backbone_name = "google/gemma-2B"
    num_iter = 40
    pretrained_path = args.pretrained_path
    if edge_network_enable:
        lr = 0.0001
    else:
        lr = 0.4

    #clear the file result/crosswords/graphs.txt
    with open("result/crosswords/graphs.txt", "w") as file:
        file.write("")
    with open("result/crosswords/cpu_loop_check.txt", "w") as file:
        file.write("")

    evaluator = CrosswordsEvaluator(test_data, batch_size=batch_size, metric="words", window_size=window_size, init_socre=0.4, use_init_score=True)
    swarm = Swarm(["CrosswordsBruteForceOpt","CrosswordsReflection"], "crosswords", "meta-llama/Meta-Llama-3-8B-Instruct",#"google/gemma-7B-it",#,#"gpt-3.5-turbo-1106", #"gpt-4-1106-preview" ,  #"CrosswordsToT","CrosswordsBruteForceOpt","CrosswordsReflection"
                final_node_class="ReturnAll", 
                final_node_kwargs={},
                edge_optimize=True,
                init_connection_probability=init_connection_probability, 
                connect_output_nodes_to_final_node=connect_output_nodes_to_final_node, 
                include_inner_agent_connections=include_inner_agent_connections,
                edge_network_enable=edge_network_enable,
                llm_backbone_name=llm_backbone_name)
    optimizer = None
    if pretrained_path : 
        swarm.connection_dist.load_state_dict(torch.load(pretrained_path))
        optimizer = torch.optim.Adam(swarm.connection_dist.parameters(), lr=lr).load_state_dict(torch.load(pretrained_path.replace("edge_logits","optimizer")))
    optimize(swarm, evaluator, batch_size=batch_size, num_iter=num_iter, display_freq=1, record=True,
              experiment_id=experiment_id, lr=lr, use_learned_order=use_learned_order, edge_network_enable=edge_network_enable, optimizer=optimizer)

   #Start evaluating the final network
   #test_evaluator = CrosswordsEvaluator(test_data, batch_size=batch_size, metric="words", window_size=window_size)
 
'''

    #funktioniert nicht mit edge network omg
    file_path = "dataset/crosswords/mini0505_0_100_5.json"
    with open(file_path, "r") as file:
        test_data = json.load(file)
        
    epochs = 1
    batch_size = 1 #4
    use_learned_order = True
    edge_network_enable = True
    num_batches = int(len(test_data) / batch_size)
    evaluator = CrosswordsEvaluator(test_data, batch_size=batch_size, metric="words", window_size=num_batches)


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
    
    #new code for instant evaluation
    loop = asyncio.get_event_loop()
    utilities = []
    evaluator.reset()
    num_batches = int(len(test_data) / batch_size)
    for k in range(num_batches):
        print("batch: ",k)
        if edge_network_enable:
            utilities += batched_evaluator_withEdgeNetwork(evaluator, batch_size, swarm, loop)
        else:
            graph = swarm.connection_dist_realize(swarm.composite_graph, use_learned_order=use_learned_order)
            utilities += batched_evaluator(evaluator, batch_size, graph, loop)
        print(f"avg. utility = {np.mean(utilities):.3f}")
    print(f"avg. utility = {np.mean(utilities):.3f}")
    with open(f"result/crosswords/{experiment_id}_final_utilities.pkl", "wb") as file:
        pickle.dump(utilities, file)
'''