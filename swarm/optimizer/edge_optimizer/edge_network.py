import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple
from swarm.utils.select_gpu import select_gpu


class EdgeNetwork(nn.Module):
    """
    EdgeNetwork is a neural network model designed to predict edge probabilities in a graph-based structure
    using a large language model (LLM) as the backbone.

    Attributes:
        device (str): Device to run the model on ('cuda' or 'cpu').
        llm_backbone (transformers.PreTrainedModel): Pretrained language model used as the backbone.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the LLM.
        embedding_only (bool): Flag to use only embeddings from the LLM.
        linear (nn.Linear): Linear layer to output edge probabilities.
    """
    
    def __init__(self, llm_backbone_name: str, num_edges: int, initial_probability: float = 0.5, embedding_only: bool = False):
        """
        Initializes the EdgeNetwork model.

        Args:
            llm_backbone_name (str): Name of the pretrained LLM.
            num_edges (int): Number of edges to predict probabilities for.
            initial_probability (float): Initial probability for edge logits.
            embedding_only (bool): Whether to use only embeddings from the LLM.
        """
        super(EdgeNetwork, self).__init__()
        
        self.device = f"cuda:{select_gpu()}" if torch.cuda.is_available() else "cpu"
        self.llm_backbone = AutoModel.from_pretrained(llm_backbone_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_backbone_name)
        self.embedding_only = embedding_only

        # Handle special token addition for certain models like GPT-2
        if llm_backbone_name.lower() == "gpt2":
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm_backbone.resize_token_embeddings(len(self.tokenizer))
        
        hidden_size = self.llm_backbone.config.hidden_size
        
        # Optionally use only input embeddings from the LLM
        if embedding_only:
            self.llm_backbone = self.llm_backbone.get_input_embeddings()

        # Freeze LLM backbone parameters
        for param in self.llm_backbone.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(hidden_size, num_edges)
        bias = torch.full((num_edges,), torch.log(torch.tensor(initial_probability / (1 - initial_probability))))
        with torch.no_grad():
            self.linear.bias.copy_(bias)
        
        self.to(self.device)

    def forward(self, input_text: str) -> torch.Tensor:
        """
        Forward pass of the EdgeNetwork model.

        Args:
            input_text (str): Input text for the model.

        Returns:
            torch.Tensor: Predicted edge logits.
        """
        # Tokenize input text and move to the appropriate device
        input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True, return_tensors='pt')['input_ids'].to(self.device)
        llm_bare_output = self.llm_backbone(input_ids)
        
        # If using only embeddings, return the processed output directly
        if self.embedding_only:
            return self.linear(torch.mean(llm_bare_output[-1], dim=0)).cpu()
        
        llm_output = llm_bare_output.last_hidden_state
        llm_output = llm_output[0][-1]
        edge_logits = self.linear(llm_output)
        
        return edge_logits.cpu()
