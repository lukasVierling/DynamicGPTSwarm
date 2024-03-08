import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

from copy import deepcopy
from typing import Tuple
import random

from swarm.graph.node import Node
from swarm.graph.graph import Graph
from swarm.graph.composite_graph import CompositeGraph

class EdgeNetwork(nn.Module):
    def __init__(self, llm_backbone_name, num_edges):
        super(EdgeNetwork, self).__init__()
        self.llm_backbone = AutoModel.from_pretrained(llm_backbone_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_backbone_name)
        # check if tokenizer is the same as for the data preprocessing (eos_token, etc.)
        # Checked: Seems as if the input is basic text so the tokenizer can stay because text is not tokenized yet
        self.linear = nn.Linear(self.llm_backbone.config.hidden_size, num_edges)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_text):
        input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True, truncation=True, return_tensors='pt')['input_ids']
        llm_output = self.llm_backbone(input_ids)[0]
        edge_logits = self.linear(llm_output)
        edge_probs = self.sigmoid(edge_logits)
        return edge_probs
    