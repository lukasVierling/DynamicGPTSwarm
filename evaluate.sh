#!/bin/bash

#python run_mmlu2.py --model_name "google/gemma-2B-it" "meta-llama/Meta-Llama-3-8B-Instruct" --epsilon 0.5  --reduce_cost --num-iterations 250 --edge_network_enable
python run_mmlu.py --domain "mixedmmlu" --num-truthful-agents 8 --num-adversarial-agents 0 --model_name "google/gemma-7B-it" "vivo-ai/BlueLM-7B-Chat" --edge_network_enable --reproduce --reduce_edges --delta 0.05
python run_mmlu.py --domain "mixedmmlu" --num-truthful-agents 8 --num-adversarial-agents 0 --model_name "google/gemma-7B-it" "vivo-ai/BlueLM-7B-Chat"  --reproduce --reduce_edges --delta 0.05