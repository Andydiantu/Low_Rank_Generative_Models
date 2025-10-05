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
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0259.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt


timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0219.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt


timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0179.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt



timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0139.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt



timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0099.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt



timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0059.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt


timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0019.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt



timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0269.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt



timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0279.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt




timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p tmux_log/$(date +%m%d)
python -u src/FID_eval.py --evaluate_path DiT20250906_103859/EMA_model_0289.pt 2>&1 | tee tmux_log/$(date +%m%d)/output_log_${timestamp}.txt

