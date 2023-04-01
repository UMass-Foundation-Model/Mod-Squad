import torch
import math
import skimage
import skimage.io
import scipy
import numpy as np
import os
# import scipy.misc
from PIL import Image

# AA = 0 
all_depth = []
def compute_depth_errors(pred, gt):
    # https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
    """Computation of error metrics between predicted and ground truth depths
    """

    before_gt = gt.clone()

    gt = torch.exp( gt * math.log( 2. ** 16.0 ) ) - 1.0
    gt = torch.log(gt) / 11.09
    gt = ( gt - 0.64 ) / 0.18
    gt = ( gt + 1. ) / 2
    gt = gt * 80.0 # map to [0,80]
    gt = torch.clamp(gt, min=1, max=80.0)

    pred = torch.clamp(pred, min=0.)
    pred = torch.exp(pred * math.log( 2. ** 16.0 ) ) - 1.0
    pred = torch.log(pred) / 11.09
    pred = ( pred - 0.64 ) / 0.18
    pred = ( pred + 1. ) / 2
    pred = pred * 80.0
    pred = torch.clamp(pred, min=1, max=80.0)

    mask = (gt>0.0) #& (gt < 127.9)
    if mask.sum() > 0:
        gt = gt[mask]
        pred = pred[mask]
    else:
        print(before_gt.median(), before_gt.mean(), before_gt.min(), before_gt.max())
        assert False

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return rmse

def depth_single_image( predicted, to_store_name):
    if predicted.dim() == 3:
        predicted = predicted.unsqueeze(3)
    predicted = predicted.numpy()
    predicted = np.exp(predicted * np.log( 2.0**16.0 )) - 1.0
    predicted = np.log(predicted) / 11.09
    predicted = ( predicted - 0.64 ) / 0.18
    predicted = ( predicted + 1. ) / 2

    predicted[:,0,0,:] = 0.
    predicted[:,1,0,:] = 1.
    predicted = np.clip(np.squeeze(predicted), 0., 1.0)
    predicted = np.uint8(predicted * 255)
    im = Image.fromarray((predicted)).convert("L")
    im.save(to_store_name)

def curvature_single_image(predicted, to_store_name):
    if predicted.shape[1] <= 3:
        predicted = predicted.permute(0,2,3,1)
    predicted = predicted.numpy()
    std = [31.922, 21.658]
    mean = [123.572, 120.1]
    predicted = (predicted * std) + mean
    predicted[:,0,0,:] = 0.
    predicted[:,1,0,:] = 1.
    predicted = np.squeeze(np.clip(predicted.astype(int) / 255., 0., 1. )[:,:,:,0])

    predicted = np.clip(np.squeeze(predicted), 0., 1.0)
    print(predicted.shape, np.mean(predicted), np.min(predicted), np.max(predicted), to_store_name)
    predicted = np.uint8(predicted * 255)
    im = Image.fromarray((predicted)).convert("L")
    im.save(to_store_name)

# just_rescale = ['autoencoder', 'denoise', 'edge2d', 
                    # 'edge3d', 'keypoint2d', 'keypoint3d',
                    # 'reshade', 'rgb2sfnorm' ]
def simple_rescale_img( predicted, to_store_name ):
    if predicted.dim() == 3:
        predicted = predicted.unsqueeze(3)
    if predicted.shape[1] <= 3:
        predicted = predicted.permute(0,2,3,1)
    predicted = predicted.numpy()

    predicted = (predicted + 1.) / 2.
    predicted = np.clip(predicted, 0., 1.)
    predicted[:,0,0,:] = 0.
    predicted[:,1,0,:] = 1.
    predicted = np.clip(np.squeeze(predicted), 0., 1.0)
    predicted = np.uint8(predicted * 255)
    im = Image.fromarray((predicted)).convert("RGB")
    im.save(to_store_name)
    # scipy.misc.toimage(np.squeeze(predicted), cmin=0.0, cmax=1.0).save(to_store_name)

# task == 'segment2d' or task == 'segment25d'
# https://github.com/StanfordVL/taskonomy/blob/master/taskbank/lib/models/encoder_decoder_segmentation.py#L84
def segmentation_pca( predicted, to_store_name ):
    
    if predicted.dim() == 3:
        predicted = predicted.unsqueeze(3)
    if predicted.shape[1] <= 3:
        predicted = predicted.permute(0,2,3,1)
    predicted = predicted.numpy()

    # print(predicted.shape, np.mean(predicted), np.min(predicted), np.max(predicted), to_store_name)
    predicted = np.squeeze(predicted)
    from sklearn.decomposition import PCA  
    x = np.zeros((224,224,3), dtype='float')
    k_embed = 8
    embedding_flattened = np.tile(predicted.reshape((-1,1)), (1,64))
    pca = PCA(n_components=3)
    pca.fit(np.vstack(embedding_flattened))
    lower_dim = pca.transform(embedding_flattened).reshape((224,224,-1))
    x = (lower_dim - lower_dim.min()) / (lower_dim.max() - lower_dim.min())
    predicted = np.clip(np.squeeze(predicted), 0., 1.0)
    predicted = np.uint8(predicted * 255)
    im = Image.fromarray((predicted)).convert("RGB")
    im.save(to_store_name)
    # scipy.misc.toimage(np.squeeze(x), cmin=0.0, cmax=1.0).save(to_store_name)

# task in ['class_object', 'class_places']
def classification(predicted, synset, to_store_name):
    predicted = predicted.numpy()
    predicted = predicted.squeeze()
    sorted_pred = np.argsort(predicted)[::-1]
    top_5_pred = [synset[sorted_pred[i]] for i in range(5)]
    to_print_pred = "Top 5 prediction: \n {}\n {}\n {}\n {} \n {}".format(*top_5_pred)
    img = Image.new('RGBA', (400, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('lib/data/DejaVuSerifCondensed.ttf', 25)
    d.text((20, 5), to_print_pred, fill=(255, 0, 0), font=fnt)
    img.save(to_store_name, 'PNG')

vis_bh = 0
def image_visualize(args, data, pred):
    global vis_bh
    vis_bh = vis_bh + 1
    if vis_bh > 10:
        assert False
    for the_type in args.img_types:
        os.makedirs('/gpfs/u/home/AICD/AICDzich/scratch/vis_img/'+str(args.exp_name), exist_ok=True)
        pred_name = '/gpfs/u/home/AICD/AICDzich/scratch/vis_img/'+args.exp_name+'/pred_'+the_type+'_'+str(vis_bh)+'.png'
        gt_name = '/gpfs/u/home/AICD/AICDzich/scratch/vis_img/'+args.exp_name+'/gt_'+the_type+'_'+str(vis_bh)+'.png'
        if 'rgb' in the_type:
            simple_rescale_img(data[the_type][0:1], gt_name)
        elif the_type in ['class_object', 'class_scene']:
            pass
        elif the_type == 'segment_unsup2d' or the_type == 'segment_unsup25d':
            # simple_rescale_img(data[the_type][0:1], gt_name)
            # simple_rescale_img(pred[the_type][0: 1], pred_name)
            segmentation_pca(data[the_type][0:1], gt_name)
            segmentation_pca(pred[the_type][0: 1], pred_name)
        elif 'depth' in the_type:
            depth_single_image(data[the_type][0:1], gt_name)
            depth_single_image(pred[the_type][0:1], pred_name)
        elif 'curvature' in the_type:
            print('curv data: ', data[the_type].shape)
            print('curv pred: ', pred[the_type].shape)
            curvature_single_image(data[the_type][0:1], gt_name)
            curvature_single_image(pred[the_type][0:1], pred_name)
        elif 'edge' in the_type or 'keypoint' in the_type or 'reshad' in the_type or 'normal' in the_type:
            print(the_type)
            print('data: ', data[the_type].shape)
            print('pred: ', pred[the_type].shape)
            simple_rescale_img(data[the_type][0:1], gt_name)
            simple_rescale_img(pred[the_type][0:1], pred_name)
        else:
            print(the_type)
            assert False
