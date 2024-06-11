# DynamicGPTSwarm

## Introduction

Recent progress in the areas of Large Language Models (LLMs) and Language Agents has demonstrated significant promise for various future applications across multiple disciplines. Traditional approaches to language agents often rely on fixed, handcrafted designs. Our research aims to develop agents that are both learnable and dynamic, utilizing a graph framework to generate edges dynamically based on input.

In this framework, we learn a model that generates edges representing the flow of communication within the graph, adjusting the internal communication of a language agent. By fine-tuning a pretrained LLM with reinforcement learning on multiple datasets, we demonstrate that our approach surpasses static methods in accuracy and adaptability across various tasks. Specifically, our approach achieves nearly 6% higher accuracy on a combined dataset of MMLU and CMMLU, and over 10% higher with a sparsity-inducing loss.

## Features

- Dynamic edge generation based on input
- Training with reinforcement learning
- Supports multiple datasets simultaneously
- Superior performance on MMLU, CMMLU, and Mini Crossword Puzzles datasets

## Installation

To install and set up the repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/lukasVierling/DynamicGPTSwarm.git
   cd DynamicGPTSwarm

    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To use the code, follow these steps:
```bash
python train.py --config configs/train_config.json
```

## Results

We demonstrate that our approach surpasses the previous static approach by nearly 6% accuracy on a combined dataset of MMLU and CMMLU, and by more than 10% when trained with a sparsity-inducing loss. It also shows superior performance in additional experiments conducted with the MMLU and Mini Crossword Puzzles datasets.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments

This research builds upon the work by \citet{zhuge2024language}. Their original code base can be found here.

### Contact

For any questions or issues, please open an issue on this repository or contact us at [your_email@example.com].