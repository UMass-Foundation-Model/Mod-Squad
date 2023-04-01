import torch
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import random
import csv
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import PIL
import skimage  

all_tasks = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'keypoints2d', 'edge_occlusion', 'edge_texture', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
new_scale, current_scale, no_clip, preprocess, no_clip = {}, {}, {}, {}, {}

for task in all_tasks:
    new_scale[task], current_scale[task], no_clip[task] = [-1.,1.], None, None
    preprocess[task] = False
    no_clip[task] = False


current_scale['rgb'] = [0.0, 255.0]
# class_object', ' xentropy

# class_scene xentropy

# depth_euclidean l1_loss

# keypoints2d l1
current_scale['keypoints2d'] = [0.0, 0.005 * (2**16)]

# keypoints3d

current_scale['keypoints3d'] = [0.0, 1.0 * (2**16)] # 64000

# normal l1_loss

current_scale['normal'] = [0.0, 255.0]
# principal_curvature l2

# reshading l1
current_scale['reshading'] = [0.0, 255.0]
# segment_unsup2d metric_loss

# edge_texture l1
current_scale['edge_texture'] = [0.0, 0.08 * (2**16)]

# edge_occlusion l1

current_scale['edge_occlusion'] = [0.0, 0.00625* (2**16)]

no_clip['edge_occlusion'] = True

# segment_unsup2d
current_scale['segment_unsup2d'] = [0.0, 255.0]

# segment_unsup25d
current_scale['segment_unsup25d'] = [0.0, 255.0]

preprocess['principal_curvature'] = True

def curvature_preprocess(img, new_dims, interp_order=1):
    img = img[:,:,:2]
    img = img - [123.572, 120.1]
    img = img / [31.922, 21.658]
    return img

def rescale_image(im, new_scale=[-1.,1.], current_scale=None, no_clip=False):
    """
    Rescales an image pixel values to target_scale
    
    Args:
        img: A np.float_32 array, assumed between [0,1]
        new_scale: [min,max] 
        current_scale: If not supplied, it is assumed to be in:
            [0, 1]: if dtype=float
            [0, 2^16]: if dtype=uint
            [0, 255]: if dtype=ubyte
    Returns:
        rescaled_image
    """
    # im = skimage.img_as_float(im).astype(np.float32)
    im = np.array(im).astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            im = np.clip(im, min_val, max_val)
        im = im - min_val
        im /= (max_val - min_val)
    min_val, max_val = new_scale
    im *= (max_val - min_val)
    im += min_val

    return im

from scipy.ndimage.filters import gaussian_filter
def rescale_image_gaussian_blur(img, new_scale=[-1.,1.], interp_order=1, blur_strength=4, current_scale=None, no_clip=False):
    """
    Resize an image array with interpolation, and rescale to be 
      between 
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    new_scale : (min, max) tuple of new scale.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    # img = skimage.img_as_float( img ).astype(np.float32)
    # img = resize_image( img, new_dims, interp_order )
    img = rescale_image( img, new_scale, current_scale=current_scale, no_clip=True )
    blurred = gaussian_filter(img, sigma=blur_strength)
    if not no_clip:
        min_val, max_val = new_scale
        np.clip(blurred, min_val, max_val, out=blurred)
    return blurred

class TaskonomyDataset(data.Dataset):
    
    def __init__(self, img_types, data_dir='./data', 
        split='medium', partition='train', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        
        super(TaskonomyDataset, self).__init__()
        
        self.partition = partition
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.class_num = {'class_object': 1000, 'class_scene': 365, 'segment_semantic':18}

        def loadSplit(splitFile, full=False):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    scene = row[0]
                    if scene == 'woodbine': # missing from the dataset
                        continue
                    if scene == 'wiconisco': # missing 80 images for edge_texture
                        continue
                    no_list = {'brinnon', 'cauthron', 'cochranton', 'donaldson', 'german',
                        'castor', 'tokeland', 'andover', 'rogue', 'athens', 'broseley', 'tilghmanton', 'winooski', 'rosser', 'arkansaw', 'bonnie', 'willow', 'timberon', 'bohemia', 'micanopy', 'thrall', 'annona', 'byers', 'anaheim', 'duarte', 'wyldwood'
                    }
                    new_list = {'ballou', 'tansboro', 'cutlerville', 'macarthur', 'rough', 'darnestown', 'maryhill', 'bowlus', 'tomkins', 'herricks', 'mosquito', 'brinnon', 'gough'}
                    
                    if scene in new_list and full:
                        continue
                    if scene in no_list and (not full):
                        continue
                    is_train, is_val, is_test = row[1], row[2], row[3]
                    if is_train=='1' or is_val=='1':
                        label = 'train'
                    else:
                        label = 'test'

                    if label in dictLabels.keys():
                        dictLabels[label].append(scene)
                    else:
                        dictLabels[label] = [scene]
            return dictLabels

        self.split = split

        if split == 'medium':
            self.data = loadSplit(splitFile = os.path.join(data_dir, 'splits_taskonomy/train_val_test_medium.csv'))
        elif split == 'fullplus':
            self.data = loadSplit(splitFile = os.path.join(data_dir, 'splits_taskonomy/train_val_test_fullplus.csv'), full=True)
        else:
            assert False
        print('data_dir: ', data_dir)

        self.scene_list = self.data[partition]
        self.img_types = img_types

        self.data_list = {}
        for img_type in img_types:
            self.data_list[img_type] = []

        for scene in self.scene_list:
            length = {}
            _max = 0
            for img_type in img_types:
                image_dir = os.path.join(data_dir, img_type, 'taskonomy', scene)
                try:
                    images = sorted(os.listdir(image_dir))
                except:
                    print(scene)
                    continue

                length[img_type] = len(images)
                _max = max(_max, length[img_type])
                for image in images:
                    self.data_list[img_type].append(os.path.join(image_dir, image))
                    if 'class' in img_type:
                        continue

            for _key, value in length.items():
                if value < _max: # check which scene miss data
                    print(_key+'/taskonomy/'+scene)

        # assert False
        self.length = len(self.data_list[self.img_types[0]])
        # print(len(self.data_list[self.img_types[0]]), self.data_list[self.img_types[0]][self.length-1])
        self._max, self._min = {}, {}
        for img_type in self.img_types:
            self._max[img_type] = -1000000.0
            self._min[img_type] = 100000.0

    def __getitem__(self, index):
        # Load Image
        output = {}
        imgs = []
        no_use = False
        for img_type in self.img_types:
            if img_type == 'class_scene' or img_type == 'class_object':
                target = np.load(self.data_list[img_type][index])
                output[img_type] = torch.from_numpy(target).float()
            else:
                try:
                    img = Image.open(self.data_list[img_type][index])  
                except:
                    print(self.data_list[img_type][index])
                    img = Image.open(self.data_list[img_type][index-1])
                np_img = np.array(img)
                if isinstance(np_img.max(), PIL.PngImagePlugin.PngImageFile):
                    print('corrupt: ', self.data_list[img_type][index])
                    return self.__getitem__(index-1)

                imgs.append(img)

        # Transfor operation on image
        if self.resize_scale:
            imgs = [img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR) \
                for img in imgs]

        if self.crop_size:
            if self.partition == 'test':
                x = (self.resize_scale - self.crop_size + 1)//2
                y = (self.resize_scale - self.crop_size + 1)//2
                imgs = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in imgs]
            else:
                x = random.randint(0, self.resize_scale - self.crop_size + 1)
                y = random.randint(0, self.resize_scale - self.crop_size + 1)
                imgs = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in imgs]

        if self.fliplr:
            if random.random() < 0.5:
                # imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
                imgs = [img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for img in imgs]

        # imgs = [skimage.img_as_float(img).astype(np.float32) for img in imgs]

        # Value operation on Tensor
        pos = 0
        for img_type in self.img_types:
            if img_type == 'class_scene' or img_type == 'class_object':
                # Note: Seems that only part of class_object is used
                continue
            else:
                output[img_type] = imgs[pos]
                if 'depth' in img_type:
                    output[img_type] = np.array(output[img_type])
                    output[img_type] = np.log(1+output[img_type]) / ( np.log( 2. ** 16.0 ) )
                elif 'curvature' in img_type:
                    output[img_type] = np.array(output[img_type])
                    output[img_type] = curvature_preprocess(output[img_type], (256, 256))
                elif 'edge_occlusion' in img_type:
                    output[img_type] = rescale_image_gaussian_blur(output[img_type],current_scale=current_scale[img_type], no_clip=no_clip[img_type])
                else:
                    output[img_type] = rescale_image(output[img_type], new_scale[img_type], current_scale=current_scale[img_type], no_clip=no_clip[img_type])
 
                output[img_type] = torch.from_numpy(output[img_type]).float()
                if output[img_type].dim() == 3 and output[img_type].shape[2]>1:
                    output[img_type] = output[img_type].permute(2,0,1)
                pos = pos + 1

        return output

    def __len__(self):
        if self.partition == 'test':
            return self.length//10
        return self.length


class FewshotTaskonomy(TaskonomyDataset):

    def __init__(self, shots, *args, **kwargs):
        super(FewshotTaskonomy, self).__init__(*args, **kwargs)

        np.random.seed(20250901)
        self.choose = np.random.randint(self.length, size=shots)

        print(self.choose)

        self.length = shots

    def __getitem__(self, index):
        return super(FewshotTaskonomy, self).__getitem__(self.choose[index])


class PercentageTaskonomy(TaskonomyDataset):

    def __init__(self, perc, *args, **kwargs):
        super(PercentageTaskonomy, self).__init__(*args, **kwargs)

        np.random.seed(20250901)
        self.perc = perc
        self.choose = np.random.randint(self.length, size=int(self.length * self.perc))

        self.length = int(self.length * self.perc)

    def __getitem__(self, index):
        return super(PercentageTaskonomy, self).__getitem__(self.choose[index])



import random
if __name__ == '__main__':
    img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'normal', 'principal_curvature', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'reshading', 'rgb', 'segment_unsup2d', 'segment_unsup25d']
    
    train_set = TaskonomyDataset(img_types, split='fullplus', partition='train', resize_scale=256, crop_size=224, fliplr=True)
    print(len(train_set))
    A = train_set.__getitem__(len(train_set)-1)
    A = train_set.__getitem__(0)

    train_loader = DataLoader(train_set, batch_size=28*6, num_workers=48, shuffle=False, pin_memory=False)
    for itr, data in tqdm(enumerate(train_loader)):
        pass
