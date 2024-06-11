#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Tuple, List, Dict
import random

from swarm.graph.node import Node
from swarm.graph.graph import Graph
from swarm.graph.composite_graph import CompositeGraph

from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry


class ConnectDistribution(nn.Module):
    def __init__(self, potential_connections):
        super().__init__()
        self.potential_connections = potential_connections

    def realize(self, graph):
        raise NotImplemented


class MRFDist(ConnectDistribution):
    pass


class EdgeWiseDistribution(ConnectDistribution):
    def __init__(self,
                 potential_connections,
                 initial_probability: float = 0.5,
                 ):
        super().__init__(potential_connections)
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        init_tensor = torch.ones(
            len(potential_connections),
            requires_grad=True) * init_logit
        self.edge_logits = torch.nn.Parameter(init_tensor)
        node_ids = set([x for pair in potential_connections for x in pair])
        self.node_idx2id = {i: node_id for i, node_id in enumerate(node_ids)}
        self.node_id2idx = {node_id: i for i, node_id in enumerate(node_ids)}
        order_tensor = torch.randn(len(node_ids))
        self.order_params = torch.nn.Parameter(order_tensor)

    def random_sample_num_edges(self, graph: CompositeGraph, num_edges: int) -> CompositeGraph:
        _graph = deepcopy(graph)
        while True:
            if _graph.num_edges >= num_edges:
                break
            potential_connection = random.sample(self.potential_connections, 1)[0]
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph

    def realize_ranks(self, graph, use_max: bool = False):
        log_probs = []
        ranks = {}
        in_degrees = {node.id: len(node.predecessors) for node in graph.nodes.values()}
        for i in range(len(self.order_params)):
            avaliable_nodes = [node for node in graph.nodes if in_degrees[node] == 0]
            logits = []
            for node in avaliable_nodes:
                logits.append(self.order_params[self.node_id2idx[node]])
            logits = torch.stack(logits).reshape(-1)
            if use_max:
                idx = torch.argmax(logits)
            else:
                idx = torch.distributions.Categorical(logits=logits).sample()
            log_probs.append(torch.log_softmax(logits, dim=0)[idx])

            ranks[avaliable_nodes[idx]] = i
            in_degrees[avaliable_nodes[idx]] = -1
            for successor in graph.nodes[avaliable_nodes[idx]].successors:
                in_degrees[successor.id] -= 1
        return ranks, torch.sum(torch.stack(log_probs))


    def realize(self,
                graph: CompositeGraph,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = None,
                use_learned_order: bool = False,
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        if use_learned_order:
            ranks, log_prob = self.realize_ranks(graph, threshold is not None)
            log_probs = [log_prob]
        else:
            log_probs = [torch.tensor(0.0, requires_grad=True)]
        _graph = deepcopy(graph)
        for potential_connection, edge_logit in zip(
                self.potential_connections, self.edge_logits):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue
            
            addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])
            addable_if_not_used_learned_order = (not use_learned_order) and (not _graph.check_cycle(in_node, {out_node}, set()))
            if addable_if_not_used_learned_order or addable_if_use_learned_order:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node)
                    # in_node.add_predecessor(out_node)
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        log_prob = torch.sum(torch.stack(log_probs))
        return _graph, log_prob

    def realize_full(self, graph: CompositeGraph) -> CompositeGraph:
        _graph = deepcopy(graph)
        for i, potential_connection in enumerate(self.potential_connections):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph

    def realize_mask(self, graph: CompositeGraph, edge_mask: torch.Tensor) -> CompositeGraph:
        _graph = deepcopy(graph)
        for i, (potential_connection, is_edge) in enumerate(zip(self.potential_connections, edge_mask)):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                if is_edge:
                    out_node.add_successor(in_node)
                    in_node.add_predecessor(out_node)
        return _graph
    
class EdgeWiseDistributionByModel(ConnectDistribution):
    """
    EdgeWiseDistributionByModel is a class that computes edge probabilities for graph-based structures
    using a provided model and domain-specific prompts.

    Attributes:
        potential_connections (List[Tuple[int, int]]): List of potential connections (edges) in the graph.
        model (nn.Module): Model used to compute edge logits.
        domain (str): Domain specifying the type of prompt to use.
        node_idx2id (Dict[int, int]): Mapping from node indices to node IDs.
        node_id2idx (Dict[int, int]): Mapping from node IDs to node indices.
        order_params (nn.Parameter): Parameter tensor for ordering nodes.
        prompt_set (PromptSet): Prompt set for the given domain.
    """

    def __init__(self, potential_connections: List[Tuple[int, int]], model: nn.Module, domain: str):
        """
        Initializes the EdgeWiseDistributionByModel class.

        Args:
            potential_connections (List[Tuple[int, int]]): List of potential connections (edges) in the graph.
            model (nn.Module): Model used to compute edge logits.
            domain (str): Domain specifying the type of prompt to use.
        """
        super().__init__(potential_connections)
        self.model = model
        node_ids = set([x for pair in potential_connections for x in pair])
        self.node_idx2id = {i: node_id for i, node_id in enumerate(node_ids)}
        self.node_id2idx = {node_id: i for i, node_id in enumerate(node_ids)}
        order_tensor = torch.randn(len(node_ids))
        self.order_params = nn.Parameter(order_tensor)
        self.domain = domain
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def get_edge_probs(self, graph: CompositeGraph, inputs: Dict[str, any] = None) -> torch.Tensor:
        """
        Computes edge probabilities for the given graph.

        Args:
            graph (CompositeGraph): The graph to compute edge probabilities for.
            inputs (Dict[str, any], optional): Additional inputs required for prompt generation.

        Returns:
            torch.Tensor: Computed edge probabilities.
        """
        if self.domain == "crosswords":
            env = inputs["env"]
            prompt = [self.prompt_set.get_propose_prompt(env.render())]
        elif self.domain in ['mmlu', 'mixedmmlu', 'cmmlu']:
            prompt = [inputs["task"]]
        return torch.sigmoid(self.model(prompt).reshape(-1))

    def realize(
        self,
        graph: CompositeGraph,
        temperature: float = 1.0,
        threshold: float = None,
        use_learned_order: bool = False,
        inputs: Dict[str, any] = None,
        only_edge_probs: bool = False
    ) -> Tuple[CompositeGraph, torch.Tensor]:
        """
        Realizes the graph structure by adding edges based on computed edge probabilities.

        Args:
            graph (CompositeGraph): The graph to be realized.
            temperature (float, optional): Temperature for scaling edge logits.
            threshold (float, optional): Threshold for binarizing edge probabilities.
            use_learned_order (bool, optional): Whether to use learned order for adding edges.
            inputs (Dict[str, any], optional): Additional inputs required for prompt generation.
            only_edge_probs (bool, optional): If True, only return edge probabilities.

        Returns:
            Tuple[CompositeGraph, torch.Tensor]: Realized graph and log probabilities.
        """
        # Compute edge logits using the model
        if self.domain == "crosswords":    
            env = inputs["env"]
            prompt = [self.prompt_set.get_propose_prompt(env.render())]
        elif self.domain in ['mmlu', 'mixedmmlu', 'cmmlu']:
            prompt = [inputs["task"]]
        
        edge_logits = self.model(prompt).reshape(-1)

        if only_edge_probs:
            return torch.sigmoid(edge_logits)

        if use_learned_order:
            ranks, log_prob = self.realize_ranks(graph, threshold is not None)
            log_probs = [log_prob]
        else:
            log_probs = [torch.tensor(0.0, requires_grad=True)]
        
        _graph = deepcopy(graph)
        for potential_connection, edge_logit in zip(self.potential_connections, edge_logits):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue
            
            addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])
            addable_if_not_used_learned_order = not use_learned_order and not _graph.check_cycle(in_node, {out_node}, set())
            if addable_if_not_used_learned_order or addable_if_use_learned_order:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node)
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        log_prob = torch.sum(torch.stack(log_probs))
        return _graph, log_prob, torch.sigmoid(edge_logits)

    def random_sample_num_edges(self, graph: CompositeGraph, num_edges: int) -> CompositeGraph:
        """
        Randomly samples and adds a specified number of edges to the graph.

        Args:
            graph (CompositeGraph): The graph to add edges to.
            num_edges (int): The number of edges to sample and add.

        Returns:
            CompositeGraph: The graph with added edges.
        """
        _graph = deepcopy(graph)
        while True:
            if _graph.num_edges >= num_edges:
                break
            potential_connection = random.sample(self.potential_connections, 1)[0]
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph

    def realize_mask(self, graph: CompositeGraph, edge_mask: torch.Tensor) -> CompositeGraph:
        """
        Realizes the graph structure based on a given edge mask.

        Args:
            graph (CompositeGraph): The graph to be realized.
            edge_mask (torch.Tensor): Mask indicating which edges to add.

        Returns:
            CompositeGraph: The realized graph.
        """
        _graph = deepcopy(graph)
        for potential_connection, is_edge in zip(self.potential_connections, edge_mask):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()) and is_edge:
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph