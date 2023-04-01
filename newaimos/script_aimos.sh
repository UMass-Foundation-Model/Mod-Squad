cmd_file=$3
srun --gres=gpu:$2 --cpus-per-task=48 -N $1 --mem=600G --time 6:00:00 --pty bash $cmd_file $1 $2 $4
