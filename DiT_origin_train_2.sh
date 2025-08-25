#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=aw624 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
source venv/bin/activate
source /vol/cuda/12.0.0/setup.sh
/usr/bin/nvidia-smi
uptime

timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/DiT_trainer.py --set curriculum_learning_gradual=False --set curriculum_learning_gradual_patience=5 --set curriculum_learning_gradual_start=45 --set curriculum_learning_gradual_step_size=50 --set curriculum_learning_gradual_start_alpha=0.05 --set curriculum_learning_gradual_end_alpha=0.10 --set gradual_curriculum_learning_num_epochs=120 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt


# timestamp=$(date +"%Y%m%d_%H%M%S")
# mkdir -p tmux_log/$(date +%m%d)
# python -u src/DiT_trainer.py --set curriculum_learning_gradual_patience=5 --set curriculum_learning_gradual_start=844 --set curriculum_learning_gradual_step_size=25 --set curriculum_learning_gradual_start_alpha=0.05 --set curriculum_learning_gradual_end_alpha=0.15 --set gradual_curriculum_learning_num_epochs=200 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt


# timestamp=$(date +"%Y%m%d_%H%M%S")
# mkdir -p tmux_log/$(date +%m%d)
# python -u src/DiT_trainer.py --set curriculum_learning_gradual_patience=5 --set curriculum_learning_gradual_start=844 --set curriculum_learning_gradual_step_size=13 --set curriculum_learning_gradual_start_alpha=0.05 --set curriculum_learning_gradual_end_alpha=0.15 --set gradual_curriculum_learning_num_epochs=400 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt