import os

src_root = '/gpfs/u/home/AICD/AICDzich/scratch/vl_eval_data/ILSVRC2012'

tgt_root_0 = '/gpfs/u/home/AICD/AICDzich/scratch/vl_data/transfer/imgnet_0'
tgt_root_1 = '/gpfs/u/home/AICD/AICDzich/scratch/vl_data/transfer/imgnet_1'

for split in ['train', 'val']:
    categories = sorted(os.listdir(os.path.join(src_root, split)))
    half = len(categories)//2
    up = categories[:half]
    down = categories[half:]

    # for the_dir in up:
    #     target = os.path.join(src_root, split, the_dir)
    #     tmpLink = os.path.join(tgt_root_0, split, the_dir)
    #     os.symlink(target, tmpLink)

    for the_dir in down:
        target = os.path.join(src_root, split, the_dir)
        tmpLink = os.path.join(tgt_root_1, split, the_dir)
        os.symlink(target, tmpLink)

