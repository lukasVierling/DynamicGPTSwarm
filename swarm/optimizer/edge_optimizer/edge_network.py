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
    def __init__(self, llm_backbone_name, num_edges, initial_probability=0.5):
        #if cude available set self.device to cuda
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        super(EdgeNetwork, self).__init__()
        self.llm_backbone = AutoModel.from_pretrained(llm_backbone_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_backbone_name)
        #add padding to stop hf from giving me warnings
        if llm_backbone_name.lower() == "gpt2":
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm_backbone.resize_token_embeddings(len(self.tokenizer))  # Resize token embeddings
            #freeze the backbone
        for param in self.llm_backbone.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(self.llm_backbone.config.hidden_size, num_edges)
        #nn.init.zeros_(self.linear.weight) TODO uncomment but let's see what happens with initialization of the weights
        #print(self.linear.weight)
        bias = torch.full((num_edges,), torch.log(torch.tensor(initial_probability / (1 - initial_probability))))
        with torch.no_grad():
            self.linear.bias.copy_(bias)
        # Move model to GPU
        self.to(self.device)

    def forward(self, input_text):
        # Move input to GPU
        input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True, return_tensors='pt')['input_ids'].to(self.device)
        llm_bare_output = self.llm_backbone(input_ids)
        llm_output = llm_bare_output.last_hidden_state
        llm_output = llm_output[0][-1]
        edge_logits = self.linear(llm_output)
        return edge_logits.cpu()
    