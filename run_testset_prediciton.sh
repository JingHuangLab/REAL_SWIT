#!/bin/bash
#SBATCH -J pred
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p A40,gpu
##SBATCH -x node38
#SBATCH --mem=30G
#SBATCH --gres=gpu:1
#SBATCH -o /home/zhangky/rein_joblog/%j.log
#SBATCH -e /home/zhangky/rein_joblog/%j.err
 
module load mathlib/cuda/10.1.168_418.67
source /home/zhangky/miniconda3/bin/activate /home/zhangky/miniconda3/envs/molpal_w_rein

task_name=$1
## train and test target-specific model
cd /home/zhangky/tool/swit_real/
## 1. Please change to your own ckpt path
## 2. Please also change the test set to the data path you want to use
## 3. Please change the task name used during training
#time python test_tss_model_40.py examples/${task_name}/lightning_logs/$2/checkpoints/$3 /home/zhangky/icm_files/training_sets/$4 --task_name ${task_name} --ncpu 6
time python test_tss_model_onlysmi.py examples/${task_name}/lightning_logs/$2/checkpoints/$3 $4 --task_name ${task_name} --ncpu 6

