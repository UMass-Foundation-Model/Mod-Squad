import torch
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import random
import csv
from torchvision import transforms
from tqdm import tqdm
# from tqdm import *
from torch.utils.data import DataLoader
import PIL

# omnitools.download all --components taskonomy --subset medium \
#   --dest ./omnidata_starter_dataset/ \
#   --connections_total 40 --agree

# omnitools.download all --components taskonomy --subset medium   --dest ./omnidata_starter_dataset/   --connections_total 40 --agree --name Zitian --email tankche2@gmail.com
# omnitools.download all --components class_object class_scene depth_euclidean depth_zbuffer edge_occlusion edge_texture keypoints2d keypoints3d nonfixated_matches normal points principal_curvature reshading rgb segment_semantic segment_unsup2d segment_unsup25d --subset medium   --dest ./omnidata_starter_dataset/   --connections_total 40 --agree --name Zitian --email tankche2@gmail.com
max_v = {'rgb': 255, 'normal': 255, 'reshading':255, 'segment_unsup2d': 255, 'edge_texture': 11355, 'edge_occlusion': 11584, 'depth_euclidean': 11.1, 'depth_zbuffer': 11.1}
min_v = {'rgb': 0, 'normal': 0, 'reshading':0, 'segment_unsup2d': 0, 'edge_texture': 0, 'edge_occlusion': 0, 'depth_euclidean': 0, 'depth_zbuffer': 0}
# class_object class_scene depth_euclidean depth_zbuffer edge_occlusion edge_texture keypoints2d keypoints3d nonfixated_matches normal points principal_curvature reshading rgb segment_semantic segment_unsup2d segment_unsup25d 

# class_scene torch.Size([1])
# depth_euclidean torch.Size([512, 512])
# depth_zbuffer torch.Size([512, 512])
# edge_occlusion torch.Size([512, 512])
# edge_texture torch.Size([512, 512])
# keypoints2d torch.Size([512, 512])
# keypoints3d torch.Size([512, 512])
# normal torch.Size([512, 512, 3])
# principal_curvature torch.Size([512, 512, 3   ])
# reshading torch.Size([512, 512, 3])
# rgb torch.Size([512, 512, 3])
# segment_semantic torch.Size([256, 256])
# segment_unsup2d torch.Size([512, 512])
# segment_unsup25d torch.Size([512, 512])

# min:  {'class_object': 100000.0, 'class_scene': 100000.0, 'depth_euclidean': 208, 'depth_zbuffer': 166, 'edge_occlusion': 0, 'edge_texture': 0, 'keypoints2d': 0, 'keypoints3d': 0, 'normal': 0, 'principal_curvature': 0, '
# reshading': 0, 'rgb': 0, 'segment_semantic': 0, 'segment_unsup2d': 0, 'segment_unsup25d': 0}
# max:  {'class_object': -1000000.0, 'class_scene': -1000000.0, 'depth_euclidean': 65535, 'depth_zbuffer': 65535, 'edge_occlusion': 10128, 'edge_texture': 11041, 'keypoints2d': 3129, 'keypoints3d': 65368, 'normal': 255, 'p
# rincipal_curvature': 254, 'reshading': 255, 'rgb': 255, 'segment_semantic': 17, 'segment_unsup2d': 255, 'segment_unsup25d': 255}

# class_num = {'class_object': 1000, 'class_scene': 365, 'segment_semantic':18}

class TaskonomyDataset(data.Dataset):
    # /gpfs/u/home/AICD/AICDzich/scratch/vl_data/taskonomy_medium
    # /nobackup/users/zitian/vl_data/taskonomy_tiny
    # /data/zitianchen/taskonomy_medium
    
    def __init__(self, img_types, data_dir='/data/zitianchen/taskonomy_fullplus', 
        partition='train', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(TaskonomyDataset, self).__init__()

        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.class_num = {'class_object': 1000, 'class_scene': 365, 'segment_semantic':18}

        def loadSplit(splitFile):
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
                    # no_list = {'brinnon', 'cauthron', 'cochranton', 'donaldson', 'german',
                    #     'castor', 'tokeland', 'andover', 'rogue', 'athens', 'broseley', 'tilghmanton', 'winooski', 'rosser', 'arkansaw', 'bonnie', 'willow', 'timberon', 'bohemia', 'micanopy', 'thrall', 'annona', 'byers', 'anaheim', 'duarte', 'wyldwood'
                    # }
                    # if scene in no_list:
                    #     continue
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

        self.data = loadSplit(splitFile = os.path.join(data_dir, 'splits_taskonomy/train_val_test_fullplus.csv'))
        if partition == 'all':
            self.scene_list = self.data['train'] + self.data['test']
        else:
            self.scene_list = self.data[partition]
        self.img_types = img_types
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                                # transforms.Resize(256),
                                # transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        self.data_list = {}
        for img_type in img_types:
            self.data_list[img_type] = []

        missing_list = []
        for scene in self.scene_list:
            length = {}
            _max = 0
            for img_type in img_types:
                image_dir = os.path.join(data_dir, img_type, 'taskonomy', scene)
                try:
                    images = sorted(os.listdir(image_dir))
                except:
                    print('rm -r ', img_type+'/taskonomy/'+scene)
                    missing_list.append('rm -r ' + str(img_type) +'/taskonomy/'+str(scene))
                    continue
                # print(scene, img_type, len(images))
                length[img_type] = len(images)
                _max = max(_max, length[img_type])
                for image in images:
                    self.data_list[img_type].append(os.path.join(image_dir, image))
                    if 'class' in img_type:
                        continue
                    # try:
                    #     img = Image.open(self.data_list[img_type][-1])
                    #     np_img = np.array(img)
                    # except:
                    #     print(self.data_list[img_type][-1])
                    #     # assert False


                    # # if type(np_img.max()) == 'PngImageFile':
                    # if isinstance(np_img.max(), PIL.PngImagePlugin.PngImageFile):
                    #     print(type(np_img.max()), type(np_img), img_type, self.data_list[img_type][index])
                    #     # assert False
                    #     # return output

            for _key, value in length.items():
                if value < _max:
                    # print(_key+'/taskonomy/'+scene, value, _max)
                    print('rm -r ', _key+'/taskonomy/'+scene)
                    missing_list.append('rm -r ' + str(_key) +'/taskonomy/'+str(scene))

        with open('/data/zitianchen/taskonomy_fullplus/remove.sh', 'w') as f:
            for line in missing_list:
                f.write("%s\n" % line)
        # assert False
        self.length = len(self.data_list[self.img_types[0]])
        print(len(self.data_list[self.img_types[0]]), self.data_list[self.img_types[0]][self.length-1       ])
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
                # target = np.argmax(target)
                # print(target)
                output[img_type] = torch.from_numpy(target).float()
            else:
                # img = Image.open(self.data_list[img_type][index])
                # np_img = np.array(img)

                try:
                    img = Image.open(self.data_list[img_type][index])
                    np_img = np.array(img)
                except:
                    print(self.data_list[img_type][index])
                    img = Image.open(self.data_list[img_type][index-1])
                    np_img = np.array(img)
                    # return self.__getitem__(0)

                # if type(np_img.max()) == 'PngImageFile':
                if isinstance(np_img.max(), PIL.PngImagePlugin.PngImageFile):
                    # print(type(np_img.max()), type(np_img), img_type, self.data_list[img_type][index])
                    # assert False
                    print('corrupt: ', self.data_list[img_type][index])
                    # np_img = np_img * 0
                    # assert False
                    no_use = True
                    # return output
                # print(np_img.max(), max(self._max[img_type], np_img.max()))
                else:
                    self._max[img_type] = max(self._max[img_type], np_img.max())
                    self._min[img_type] = min(self._min[img_type], np_img.min())

                    imgs.append(img)

        if no_use:
            return self.__getitem__(0)
        # Transfor operation on image
        if self.resize_scale:
            imgs = [img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR) \
                for img in imgs]

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            imgs = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in imgs]

        if self.fliplr:
            if random.random() < 0.5:
                # imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
                imgs = [img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for img in imgs]
                

        # Value operation on Tensor
        pos = 0
        for img_type in self.img_types:
            if img_type == 'class_scene' or img_type == 'class_object':
                continue
            else:
                output[img_type] = torch.from_numpy(np.array(imgs[pos]))
                if 'segment_semantic' in img_type:
                    continue
                output[img_type] = output[img_type].float()
                if 'depth' in img_type:
                    output[img_type] = torch.log(1+output[img_type]) / ( np.log( 2. ** 16.0 ) )
                if 'edge' in img_type:
                    output[img_type] = output[img_type] / 12000.0 # I am not quite understand

                if 'keypoints2d' in img_type:
                    output[img_type] = output[img_type] / 4096.0 # I am not quite understand
                if 'keypoints3d' in img_type:
                    output[img_type] = output[img_type] / 65536.0

                # if img_type in max_v:
                #     output[img_type] = (output[img_type] - min_v[img_type]) * 1.0 / (max_v[img_type] - min_v[img_type])
                
                if 'segment_unsup2d' in img_type or 'segment_unsup25d' in img_type:
                    output[img_type] = output[img_type]/255.0

                if 'rgb' in img_type or 'normal' in img_type or 'principal_curvature' in img_type or 'reshading' in img_type:
                    output[img_type] = self.transform(output[img_type].permute(2,0,1)/255.0)

                pos = pos + 1

        return output

    def __len__(self):
        # return 1000
        return self.length

def line_to_path_fn(x, data_dir):
    path = x.decode('utf-8').strip('\n')
    return os.path.join(data_dir, path)


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, data_file, img_types, transform=None,
                 resize_scale=None, crop_size=None, fliplr=False, is_cls=False):
        super(DatasetFromFolder, self).__init__()
        with open(data_file, 'rb') as f:
            data_list = f.readlines()
        self.data_list = [line_to_path_fn(line, data_dir) for line in data_list]
        self.img_types = img_types
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.is_cls = is_cls

    def __getitem__(self, index):
        # Load Image
        domain_path = self.data_list[index]
        if self.is_cls:
            img_types = self.img_types[:-1]
            cls_target = self.img_types[-1]
        else:
            img_types = self.img_types
        img_paths = [domain_path.replace('{domain}', img_type) for img_type in img_types]
        imgs = [Image.open(img_path) for img_path in img_paths]

        for l in range(len(imgs)):
            img = np.array(imgs[l])
            img_type = img_types[l]
            update = False
            if len(img.shape) == 2:
                img = img[:,:, np.newaxis]
                img = np.concatenate([img] * 3, 2)
                update = True
            if 'depth' in img_type:
                img = np.log(1 + img)
                update = True
            if img_type in max_v:
                img = (img - min_v[img_type]) * 255.0 / (max_v[img_type] - min_v[img_type])
                update = True
            if update:
                imgs[l] = Image.fromarray(img.astype('uint8'))
        
        if self.resize_scale:
            imgs = [img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR) \
                for img in imgs]
        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            imgs = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in imgs]
        if self.fliplr:
            if random.random() < 0.5:
                imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        if self.is_cls:
            inputs = imgs
            target = np.load(domain_path.replace('{domain}', cls_target).\
                replace('png', 'npy').replace('scene.npy', 'places.npy'))
            target = np.argmax(target)
        else:
            inputs, target = imgs[:-1], imgs[-1]
            
        return inputs, target

    def __len__(self):
        return len(self.data_list)

import random
if __name__ == '__main__':
    # img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'nonfixated_matches', 'normal', 'point_info', 'principal_curvature', 'reshading', 'rgb', 'segment_semantic', 'segment_unsup2d', 'segment_unsup25d']
    # img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d', 'normal', 'principal_curvature', 'reshading', 'rgb', 'segment_semantic', 'segment_unsup2d', 'segment_unsup25d']
    # img_types = ['rgb', 'class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'normal']
    # img_types = ['principal_curvature', 'reshading', 'segment_unsup2d', 'segment_unsup25d' ] 
    img_types = ['edge_occlusion', 'edge_texture', 'keypoints2d', 'keypoints3d']
    # img_types = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'normal', 'rgb']
    # rgb class_object class_scene depth_euclidean depth_zbuffer normal 
    # img_types = ['class_scene', 'class_object', 'rgb', 'normal', 'reshading', 'depth_euclidean', 'segment_unsup2d']
    # img_types = ['rgb', 'class_object', 'normal', 'depth_euclidean', 'keypoints2d']
    # A = TaskonomyDataset(img_types, resize_scale=256)
    # # print('done')
    # # A_test = TaskonomyDataset(img_types, resize_scale=256, partition='test')
    # # assert False
    # # print('len: ', len(A))
    # # B = A.__getitem__(32)
    # # # assert False
    # for i in tqdm(range(len(A))):
    #     t = random.randint(0,len(A)-1)
    #     # print(t)
    #     B = A.__getitem__(t)
    # #     # print(B['edge_texture'][200:230, 200:230]/15000)
    # #     # print(B['edge_texture'].min(), B['edge_texture'].max(), B['edge_texture'].mean())
    # #     if i <5:
    # #         for img_type in B.keys():
    # #             print(img_type, B[img_type].min(), B[img_type].max())
    #     if i > 50:
    #         break
    # #     # print('i: ', i)
    # #     # if i==5:
    # #     #     for img_type in img_types:
    # #     #         print(img_type, B[img_type].min(), B[img_type].max(), B[img_type].shape)
    # # assert False
    # # train_loader = DataLoader(A, batch_size=32, num_workers=12, shuffle=False, pin_memory=True)
    # # for itr, data in tqdm(enumerate(train_loader)):
    # #     pass
    # #     if itr>400:
    # #         break

    # print('min: ', A._min)
    # print('max: ', A._max)

    # for i in range(10):
    #     B = A.__getitem__(i)
    #     # if i==5:
    #     #     for img_type in img_types:
    #     #         print(img_type, B[img_type].min(), B[img_type].max(), B[img_type].shape)
    # print('min: ', A._min)
    # print('max: ', A._max)
    # img_types = ['rgb', 'class_object']

    # assert False
    
    train_set = TaskonomyDataset(img_types, partition='all', resize_scale=298, crop_size=256, fliplr=True)
    print(len(train_set))
    A = train_set.__getitem__(len(train_set)-10)
    
    # B = train_set.__getitem__(10)
    # for img_type in img_types:
    #     print(B[img_type].shape)
    # assert False
    # # for i in range(1146 * 64, len(train_set)):
    # #     print('i: ', i)
    # #     B = train_set.__getitem__(i)

    # train_loader = DataLoader(train_set, batch_size=32, num_workers=30, shuffle=False, pin_memory=True)
    # for itr, data in tqdm(enumerate(train_loader)):
    #     pass
