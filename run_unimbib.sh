#!/usr/bin/env bash

#########################################
# Author:       Gentry Atkinson         #
# Organization: Texas State University  #
# Date:         20 Feb, 2023            #
#########################################

python3 main.py --dataset=unimib --diffusion_style=probabilistic_conditional --dev_num=4 --run_name=probabilistic_conditional
python3 main.py --dataset=unimib --diffusion_style=soft_conditional --dev_num=4  --run_name=soft_conditional
python3 main.py --dataset=unimib --diffusion_style=conditional --dev_num=4 --run_name=conditional
python3 main.py --dataset=unimib --diffusion_style=unconditional --dev_num=4 --run_name=unconditional