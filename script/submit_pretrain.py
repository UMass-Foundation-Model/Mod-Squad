import os
import time
def run_main():
    # tar_fn = 'slurm.sh'
    # 3, 2 can work
    ngpus = 4          
    nodes = 4
    world_size = ngpus * nodes
    # cmd_str = 'sbatch slurm.sh'
    # \ ##' --world_size ' + str(world_size) + \
    # cmd_str = 'python submit_pretrain.py --ngpus ' + str(ngpus) + ' --nodes ' + str(nodes) + \
    #           ' --batch_size 64 --epochs 400 --blr 1.5e-4 --model vit_base --norm_pix_loss --exp-name vlmae_base_4*4'
    cmd_str = 'bash script_slurm_jobs.sh '+str(nodes) + ' ' + str(ngpus) +' pretrain.sh'
    run_num = 3
    for run_id in range(run_num):
        os.system(cmd_str)
        print('start waiting 60s')
        time.sleep(60)  
if __name__=='__main__':
    run_main()
