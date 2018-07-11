#!/bin/sh

rm m_test/out/*
rm m_test/error/*
rm outputs/*


# walk_limits=( 10, 20, 40, 80)
# num_walks=( 5, 10, 20)

for i in 5 10 20
do
	for j in 10 20 40 80
	do
		sbatch --exclude=dgpu502-13-l,dgpu502-01-r,dgpu703-21 --array=0-3 --export=l=$j,n=$i slurm_su_cluster.sh

		sbatch --exclude=dgpu502-13-l,dgpu502-01-r,dgpu703-21 --array=0-3 --export=l=$j,n=$i slurm_ss_cluster.sh

		sbatch --exclude=dgpu502-13-l,dgpu502-01-r,dgpu703-21 --array=0-3 --export=l=$j,n=$i slurm_su_vae.sh

		sbatch --exclude=dgpu502-13-l,dgpu502-01-r,dgpu703-21 --array=0-3 --export=l=$j,n=$i slurm_ss_vae.sh
	done
done