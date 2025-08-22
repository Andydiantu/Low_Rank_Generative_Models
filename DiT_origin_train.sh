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
python -u src/DiT_trainer.py 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt