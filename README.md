# Probabilistic_Conditional_Diffusion

## Concept

This work presents an adaptation of conditional DDPM that encorporates label smoothing for use with time series data with unreliable labels. This technique prevents the denoising model fromn developing overconfidence in the example data and improves the ability of DDPM to fit to the distribution of the example data.

## Using this work

The file main.py can be run to generate new data using one of the provided datasets and will run some experiments on FID and accuracy of a downstream classifier using the new data. The generated data will be saved as a Torch tensor. The following keyword arguments can be passed to main:

    --dataset': "The dataset to run experiments on."
    --mislab_rate': "Percentage of label noise to add."
    --diffusion_model': "A denoising model for reverse diffusion"
    --diffusion_style': "unconditional, conditional, soft_conditional, or probabilistic_conditional"
    --data_path': "Directory for storing datasets"
    --run_path': "Directory for storing training samples"
    --run_name': "folder to write losses for one training run"
    --results_path': "folder to write experimental results"
    --data_cardinality': "Dimensionality of data being processed"
    --batch_size': "Instances to train on per iteration"
    --lr': "Learning Rate"
    --epochs': "Number of epochs for training"
    --training_samples': "number of samples to generate for each training epoch"
    --test_split': "Portion of train data to hole out for test"
    --dev_num': "Device number for running experiments on GPU"
    --time_steps': "Time steps for noising/denoising."
    --smoothing_alpha': "Weight to give to T in computing training labels."

The diffuser.py file can be imported to insantiate our implementation of DDPM yourself. Pass in arguments as shown in main.py to pick the style of diffusion you'd like to use.

Several shell scripts have been provided which repeat the conditions used in our paper: "Conditional Diffusion with Label Smoothing for Data Synthesis from Examples with Noisy Labels". These are:

    run_mitbih.sh
    run_synthetic5.sh
    run_twristar.sh
    run_unimib.sh

Descriptions of the datasets used experimentally in this work can be found in the paper.

This work is under development.