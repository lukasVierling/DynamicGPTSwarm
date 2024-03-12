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
        super(EdgeNetwork, self).__init__()
        self.llm_backbone = AutoModel.from_pretrained(llm_backbone_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_backbone_name)
        #add padding to stop hf from giving me warnings
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_backbone.resize_token_embeddings(len(self.tokenizer))  # Resize token embeddings
        #freeze the backbone
        for param in self.llm_backbone.parameters():
            param.requires_grad = False
        # check if tokenizer is the same as for the data preprocessing (eos_token, etc.)
        # Checked: Seems as if the input is basic text so the tokenizer can stay because text is not tokenized yet
        # TODO take care of initialization, should start with uniform dist, maybe use uniform + output and initialize with 0?
        self.linear = nn.Linear(self.llm_backbone.config.hidden_size, num_edges)
        nn.init.zeros_(self.linear.weight)
        # fill bias so that we get the initial probability after applying sigmoid
        # fill weights with zero
        bias = torch.full((num_edges,), torch.log(torch.tensor(initial_probability / (1 - initial_probability))))
        with torch.no_grad():
            self.linear.bias.copy_(bias)
        #self.sigmoid = nn.Sigmoid() #No Sigmoid because we do that later

    def forward(self, input_text):
        
        input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True, truncation=True, return_tensors='pt')['input_ids']
        # GEt the hidden states
        llm_bare_output = self.llm_backbone(input_ids)
        llm_output = llm_bare_output.last_hidden_state
        # get hidden state of last token
        llm_output = llm_output[0][-1]
        # get edge logtis
        edge_logits = self.linear(llm_output)
        #edge_probs = self.sigmoid(edge_logits)
        return edge_logits
    