#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1
#PBS -l walltime=24:0:0
#PBS -N DiT_origin_train_galore


cd $HOME/Low_Rank_Generative_Models

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate andy_diss


note="DiT_origin_train_galore"

cmd="python -u src/DiT_trainer.py --set num_epochs=1200"


echo $cmd

$cmd > "$HOME/Low_Rank_Generative_Models/logs/DiT_origin_train_galore_${PBS_JOBID}.log" 2>&1