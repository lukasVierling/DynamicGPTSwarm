#!/bin/bash
#python run_mmlu.py --num-truthful-agents 8 --num-adversarial-agents 8 --model_name google/gemma-7B-it --edge_network_enable --domain mmlu --reproduce
python run_mmlu.py --num-truthful-agents 8 --num-adversarial-agents 8 --model_name google/gemma-7B-it --domain mmlu --edge_network_enable --reproduce