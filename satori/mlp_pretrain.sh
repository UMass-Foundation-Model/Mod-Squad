P_PER_NODE=$2
# # this should be same as ntasks-per-node/gpu_num

# ### init virtual environment if needed
source /nobackup/users/zitian/anaconda3/etc/profile.d/conda.sh
conda activate mycongeal

echo "P_PER_NODE"$P_PER_NODE
echo "SLURM_JOB_NUM_NODES"$SLURM_JOB_NUM_NODES

### the command to run
cd /nobackup/users/zitian/code/mae/

NODE_RANK=${SLURM_PROCID}
ip2=node${SLURM_NODELIST:5:4}
NODE_LENGTH=${#SLURM_NODELIST}
if [[ ${NODE_LENGTH} == 8  ]]; then
    ip2=node${SLURM_NODELIST:4:4}
else
    ip2=node${SLURM_NODELIST:5:4}
fi

export MASTER_ADDR=${ip2}
export MASTER_PORT="8128"
export NODE_RANK=${NODE_RANK}


echo "MASTER_ADDR"${MASTER_ADDR}
echo "MASTER_PORT"${MASTER_PORT}
echo "NODE_RANK"${NODE_RANK}
echo "EXP"$3

# echo "ip2:"$ip2
# echo "NODE_RANK"$NODE_RANK
# echo "P_PER_NODE"$P_PER_NODE
# echo "SLURM_JOB_NUM_NODEsS"$SLURM_JOB_NUM_NODES
EXP=$3

python -m torch.distributed.launch \
    --master_addr ${ip2} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${P_PER_NODE}  \
    --nnodes $SLURM_JOB_NUM_NODES    \
    main_pretrain.py \
        --batch_size 48 \
        --epochs 300 \
        --input_size 224 \
        --blr 1.5e-4 --weight_decay 0.05 \
        --warmup_epochs 40 \
        --model mae_vit_moemlp_base_patch16 \
        --norm_pix_loss \
        --exp-name ${EXP} \

echo "FINISHED"
