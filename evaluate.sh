#!/bin/bash
source activate gptswarm
#python evaluate.py --id 9 --pretrained_path result/crosswords/old_method_10_it_nostuck_20batch/experiment8_edge_logits_9.pt
python evaluate.py --id 9 --pretrained_path  result/crosswords/new_method_40_it_nostuck_5batch/experiment8_edge_logits_39.pt --edge_network_enable