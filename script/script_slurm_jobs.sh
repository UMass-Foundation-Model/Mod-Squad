cmd_file=$3
srun --gres=gpu:$2 --cpus-per-task=64 -N $1 --mem=300G --time 24:00:00 --qos=sched_level_2 --partition=sched_system_all_8 --pty bash $cmd_file $1 $2
# srun --gres=gpu:4 --cpus-per-task=64 -N 1 --mem=300G --time 24:00:00 --qos=sched_level_2 --partition=sched_system_all_8 --pty bash -sched_level_2