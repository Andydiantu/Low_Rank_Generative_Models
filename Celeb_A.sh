
### Running full rank baseline
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/DiT_trainer.py --set model=DiT-S/2 --set dataset=celebA 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt


### Running low rank baseline with 50% rank
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/DiT_trainer.py --set model=DiT-S/2 --set dataset=celebA --set low_rank_pretraining=True --set low_rank_rank=0.5 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt



### Running low rank with 50% rank and timestep conditioning
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/DiT_trainer.py --set model=DiT-S/2 --set dataset=celebA --set low_rank_pretraining=True --set timestep_conditioning=True --set low_rank_rank=0.5 set curriculum_learning=True --set curriculum_learning_start_from_low=False 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt
