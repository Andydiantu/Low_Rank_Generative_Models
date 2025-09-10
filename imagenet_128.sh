
### Running full rank baseline
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/DiT_trainer.py --set model=DiT-B/2 --set dataset=imagenet 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt


### Running low rank baseline with 50% rank
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/DiT_trainer.py --set model=DiT-B/2 --set dataset=imagenet --set low_rank_pretraining=True --set low_rank_rank=0.5 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt



### Running low rank with 50% rank and timestep conditioning
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/DiT_trainer.py --set model=DiT-B/2 --set dataset=imagenet --set low_rank_pretraining=True --set timestep_conditioning=True --set low_rank_rank=0.5  --set curriculum_learning=True --set curriculum_learning_start_from_low=False 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt


timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/DiT_trainer.py 
    --set model=DiT-B/2 # The model to use
    --set dataset=imagenet # The dataset to use
    --set low_rank_pretraining=True # Whether to use low rank pretraining
    --set timestep_conditioning=True # Whether to use adaptive timestep low rank parameterisation
    --set low_rank_rank=0.5 # The rank to use
    --set timestep_conditioning_match_type=activated # select timestep conditioning match type "activated" refers ISO inference compute, "total" refers to match total number of parameters.
    --set curriculum_learning=True # Whether to use curriculum learning
    --set curriculum_learning_start_from_low=False 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt
