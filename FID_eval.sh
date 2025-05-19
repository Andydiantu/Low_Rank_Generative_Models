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

python src/FID_eval.py --evaluate_path DiT20250517_225927