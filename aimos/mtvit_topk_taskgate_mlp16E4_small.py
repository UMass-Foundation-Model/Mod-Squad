import os
import time
from datetime import datetime
import shutil

def run_main():
    ngpus = 4    
    nodes = 13
    exp_name = 'mtvit_topk_taskgate_mlp16E4_small'
    world_size = ngpus * nodes
    # datetime object containing current date and time
    now = datetime.now()
     
    print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    src_file = '/gpfs/u/home/AICD/AICDzich/barn/code/MTMoe/aimos/mtvit_topk_taskgate_mlp16E4_small.sh'
    tgt_file = '/gpfs/u/home/AICD/AICDzich/barn/code/MTMoe/aimos/save/' + str(dt_string)+ str(exp_name)+'.sh'
    shutil.copyfile(src_file , tgt_file)

    print('save to ', tgt_file)

    cmd_str = 'bash script_aimos.sh '+str(nodes) + ' ' + str(ngpus) + ' save/' + str(dt_string)+ str(exp_name)+'.sh ' + str(exp_name)

    run_num = 4*6
    for run_id in range(run_num):
        os.system(cmd_str)
        print('done running ', tgt_file)
        print('start waiting 60s')
        time.sleep(60)

if __name__=='__main__':
    run_main()
