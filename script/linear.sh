P_PER_NODE=$2

# ### init virtual environment if needed
source /nobackup/users/zitian/anaconda3/etc/profile.d/conda.sh
conda activate mycongeal

### the command to run
cd /nobackup/users/zitian/code/MoEMultiTask/mae
RUN_FN=main_linprobe.py
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
echo "SLURM_JOB_NUM_NODES"$SLURM_JOB_NUM_NODES
python -m torch.distributed.launch \
    --master_addr ${ip2} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${P_PER_NODE}  \
    --nnodes $SLURM_JOB_NUM_NODES    \
    ${RUN_FN} \
        --batch_size 512 \
        --epochs 90 \
        --blr 0.1 \
        --weight_decay 0.0 \
        --model vit_base_patch16 \
        --exp-name linear_original_mae_120 \
        --finetune /nobackup/users/zitian/work_dirs/vl_moe/original_mae_base/save-120.pth \
        # --exp-name linear_vlmae_base_vision_only_imagenet_158 \
        # --finetune /nobackup/users/zitian/work_dirs/vl_moe/vlmae_base_vision_only_imagenet/save-158.pth \

echo "FINISHED"
