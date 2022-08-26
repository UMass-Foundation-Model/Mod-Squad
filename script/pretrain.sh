# #!/bin/bash
# #SBATCH --job-name=vl-mae
# #SBATCH --partition=sched_system_all_8
# #SBATCH --time=24:00:00
# #SBATCH --qos=sched_level_2
# #SBATCH --nodes=2
# #SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64gb
# #SBATCH -e /nobackup/users/zitian/slurm/%x-%j.err
# #SBATCH --output=/nobackup/users/zitian/slurm/%x-%j.out
P_PER_NODE=$2
# # this should be same as ntasks-per-node/gpu_num

# echo "START!!"

# echo $CUDA_VISIBLE_DEVICES

# ### init virtual environment if needed
source /nobackup/users/zitian/anaconda3/etc/profile.d/conda.sh
conda activate mycongeal

### the command to run
cd /nobackup/users/zitian/code/MoEMultiTask/mae/
RUN_FN=main_pretrain.py
NODE_RANK=${SLURM_PROCID}
ip2=node${SLURM_NODELIST:5:4}
NODE_LENGTH=${#SLURM_NODELIST}
if [[ ${NODE_LENGTH} == 8  ]]; then
    ip2=node${SLURM_NODELIST:4:4}
else
    ip2=node${SLURM_NODELIST:5:4}
fi

echo "ip2:"$ip2
echo "NODE_RANK"$NODE_RANK
echo "P_PER_NODE"$P_PER_NODE
echo "SLURM_JOB_NUM_NODEsS"$SLURM_JOB_NUM_NODES
# --use_env \

python -m torch.distributed.launch \
    --master_addr ${ip2} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${P_PER_NODE}  \
    --nnodes $SLURM_JOB_NUM_NODES    \
    ${RUN_FN} \
        --batch_size 64 \
        --epochs 800 \
        --blr 1.5e-4 --weight_decay 0.05 \
        --warmup_epochs 40 \
        --model mae_vit_base_patch16 \
        --norm_pix_loss \
        --exp-name original_mae_base

# python -m torch.distributed.launch \
#     --master_adds ${ip2} \
#     --node_rank ${NODE_RANK} \
#     --nproc_per_node ${P_PER_NODE}  \
#     --nnodes $SLURM_JOB_NUM_NODES    \
#     ${RUN_FN} \
#         --batch_size 64 \
#         --epochs 800 \
#         --wmlm 0.01 \
#         --blr 1.5e-4 --weight_decay 0.05 \
#         --model vit_large \
#         --norm_pix_loss \
#         --exp-name vlmae_large_0.01_wmlm


echo "FINISHED"
