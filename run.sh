#!/bin/bash
source activate gptswarm
python run_mmlu.py --num-truthful-agents 8 --edge_network_enable
python run_mmlu.py --num-truthful-agents 8