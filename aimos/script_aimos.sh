cmd_file=$3
srun --gres=gpu:$2 --cpus-per-task=48 -N $1 --mem=300G --time 6:00:00 --pty bash $cmd_file $1 $2 $4

# srun --gres=gpu:6 --cpus-per-task=64 --mem=300G --time 6:00:00 --pty bash -l

# rm /gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium/keypoints3d/taskonomy/pinesdale/point_1519_view_2_domain_keypoints3d.png
# rsync -avz -e 'ssh' taskonomy_medium/keypoints3d/taskonomy/pinesdale/point_1519_view_2_domain_keypoints3d.png AICDzich@blp02.ccni.rpi.edu:/gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium/keypoints3d/taskonomy/pinesdale

# /gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium/depth_euclidean/taskonomy/lakeville/point_1516_view_4_domain_depth_euclidean.png

# rm /gpfs/u/home/AICD/AICDzich/scratch/vl_data/AAA
# rsync -avz -e 'ssh' AAA AICDzich@blp02.ccni.rpi.edu:/gpfs/u/home/AICD/AICDzich/scratch/vl_data/AAA


# rm /gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium/depth_euclidean/taskonomy/lakeville/point_1516_view_4_domain_depth_euclidean.png
# rsync -avz -e 'ssh' taskonomy_medium/depth_euclidean/taskonomy/lakeville/point_1516_view_4_domain_depth_euclidean.png AICDzich@blp02.ccni.rpi.edu:/gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium/depth_euclidean/taskonomy/lakeville

# rm /gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium/rgb/taskonomy/newfields/point_1070_view_8_domain_rgb.png
# rsync -avz -e 'ssh' taskonomy_medium/rgb/taskonomy/newfields/point_1070_view_8_domain_rgb.png AICDzich@blp02.ccni.rpi.edu:/gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium/rgb/taskonomy/newfields
