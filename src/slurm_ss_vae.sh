#!/bin/sh

#SBATCH --workdir=/home/akujuou/node2vec-master/src  #working directory
#SBATCH --ntasks=5	 # Specify number of tasks, i.e. cores you need
#SBATCH -t 3-00:00	 # Specify Runtime in D-HH:MM
##SBATCH --gres=gpu:titan:1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -J VAE_SS
#SBATCH -o m_test/out/g2v_vae_ss-%A_%a.out
#SBATCH --error=m_test/error/g2v_vae_ss-%A_%a.err   # Standard  error log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uchenna.akujuobi@kaust.edu.sa

module purge
module load cuda/9.0.176
module load cudnn/7.0.3-cuda9.0.176

pwd; hostname; date


# python test_aux.py --Class $SLURM_ARRAY_TASK_ID --hlay 2 --is_test False --d_size big --model_dir model4_l2

python main.py --input ../graph/  --output ../emb/ --p 1 --q 1 --num-walks $n --walk-length $l --dataset $SLURM_ARRAY_TASK_ID --l_type vae --ss
date
