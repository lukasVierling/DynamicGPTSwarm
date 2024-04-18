#!/bin/bash
source activate gptswarm
python run_mmlu.py --num-truthful-agents 4 --num-adversarial-agents 4 --domain mmlu --edge_network_enable --num-iterations 200
