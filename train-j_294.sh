#!/bin/bash
#SBATCH --job-name=SPOT-SKE
#SBATCH --account=ddt_acc23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=128gb
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=end          
#SBATCH --mail-user=21010294@st.phenikaa-uni.edu.vn
#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err
#SBATCH --nodelist=hpc21

module purge
module load cuda
module load python
source /home/21010294/VSR/VSREnv/bin/activate
module list
python -c "import sys; print(sys.path)"

which python
python --version
python /home/21010294/VSR/cudacheck.py
squeue --me
cd /home/21010294/ActionRecognition/visl-i3d


python src/main.py --config src/config/visl2_spoter.yaml



