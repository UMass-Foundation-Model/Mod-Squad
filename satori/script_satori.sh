cmd_file=$3
srun --gres=gpu:$2 --cpus-per-task=64 -N $1 --mem=300G --time 24:00:00 --qos=sched_level_2 --partition=sched_system_all_8 --pty bash $cmd_file $1 $2 $4

# ssh dcsfen01
# ssh dcsfen02
# ssh nplfen01
# srun --gres=gpu:4 --cpus-per-task=64 -N 1 --mem=300G --time 24:00:00 --qos=dcs-48hr --pty bash -l
# srun --gres=gpu:4 --cpus-per-task=64 -N 1 --mem=300G --time 6:00:00 --pty bash -l
# module load spectrum-mpi/10.4 cuda/11.2 gcc/8.4.1 cmake/3.20.0/1
# conda config --add channels 
# conda config --prepend channels https://opence.mit.edu
# conda install pytorch=1.10.2=cuda11.2_py38_1
# conda install jsonlines
# pip install sentencepiece
# pip install timm tqdm
# conda install tensorflow tensorboard tensorboardX
# bash script_slurm_jobs.sh 2 4 finetune_cls_m