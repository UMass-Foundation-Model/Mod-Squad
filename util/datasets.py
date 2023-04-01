import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision.datasets.vision import VisionDataset
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image
import json
from PIL import Image
from util.dataset_taskonomy import TaskonomyDataset
# from util.taskonomy import TaskonomyDataset

class SUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.
    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.
    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_train = True, 
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = Path(self.root) #/ "SUN397"

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._image_files = []
        the_path = Path(self.root) / "Partitions"
        if is_train:
            the_list = list(the_path.rglob("Training_*.txt"))
        else:
            the_list = list(the_path.rglob("Testing_*.txt"))
        for the_file in the_list:
            with open(the_file) as f:
                self._image_files.extend([Path(self.root + c[:-1]) for c in f])

        # self._image_files = list(self._data_dir.rglob("sun_*.jpg"))

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
        ]
        # print(self._labels)

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

class INaturalist(VisionDataset):
    def __init__(
        self,
        root: str,
        version: str = "2019",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        is_train=True,
    ) -> None:

        ori_root = root
        root = os.path.join(root, 'train_val2019')
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.version = version

        self.all_categories: List[str] = []

        # map: category type -> name of category -> index
        self.categories_index: Dict[str, Dict[str, int]] = {}

        # list indexed by category id, containing mapping from category type -> index
        self.categories_map: List[Dict[str, int]] = []

        self._init_pre2021()

        # index of all files: (full category id, filename)
        self.index: List[Tuple[int, str]] = []

        path_json = os.path.join(ori_root, f'{"train" if is_train else "val"}{version}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        self.classes = self.all_categories
        self.cat_to_id = {}
        for index, cat in enumerate(self.all_categories):
            self.cat_to_id[cat] = index

        for elem in data['images']:
            fname = os.path.join(ori_root, elem['file_name'])
            cut = elem['file_name'].split('/')
            cat = cut[1] + '/' + cut[2]
            cat_index = self.cat_to_id[cat]
            self.index.append((cat_index, fname))


    def _init_pre2021(self) -> None:
        """Initialize based on 2017-2019 layout"""

        # map: category type -> name of category -> index
        self.categories_index = {"super": {}}

        cat_index = 0
        print('root: ', self.root)
        super_categories = sorted(os.listdir(self.root))
        # print('super: ', super_categories)
        for sindex, scat in enumerate(super_categories):
            self.categories_index["super"][scat] = sindex
            subcategories = sorted(os.listdir(os.path.join(self.root, scat)))
            for subcat in subcategories:
                if self.version == "2017":
                    # this version does not use ids as directory names
                    subcat_i = cat_index
                    cat_index += 1
                else:
                    try:
                        subcat_i = int(subcat)
                    except ValueError:
                        raise RuntimeError(f"Unexpected non-numeric dir name: {subcat}")
                if subcat_i >= len(self.categories_map):
                    old_len = len(self.categories_map)
                    self.categories_map.extend([{}] * (subcat_i - old_len + 1))
                    self.all_categories.extend([""] * (subcat_i - old_len + 1))
                if self.categories_map[subcat_i]:
                    raise RuntimeError(f"Duplicate category {subcat}")
                self.categories_map[subcat_i] = {"super": sindex}
                self.all_categories[subcat_i] = os.path.join(scat, subcat)

        # validate the dictionary
        for cindex, c in enumerate(self.categories_map):
            if not c:
                raise RuntimeError(f"Missing category {cindex}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """

        cat_id, fname = self.index[index]
        img = Image.open(fname).convert('RGB')

        target = cat_id

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.index)

    def category_name(self, category_type: str, category_id: int) -> str:
        """
        Args:
            category_type(str): one of "full", "kingdom", "phylum", "class", "order", "family", "genus" or "super"
            category_id(int): an index (class id) from this category

        Returns:
            the name of the category
        """
        if category_type == "full":
            return self.all_categories[category_id]
        else:
            if category_type not in self.categories_index:
                raise ValueError(f"Invalid category type '{category_type}'")
            else:
                for name, id in self.categories_index[category_type].items():
                    if id == category_id:
                        return name
                raise ValueError(f"Invalid category id {category_id} for {category_type}")

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    if args.data_path[-6:] == 'SUN397':
        dataset = SUN397(args.data_path, transform=transform, is_train=is_train)
    elif args.data_path[-11:] == 'INaturalist':
        dataset = INaturalist(args.data_path, transform=transform, is_train=is_train)
    else:
        if is_train == False and args.eval_all:
            dataset = datasets.ImageFolder('/gpfs/u/home/AICD/AICDzich/scratch/vl_eval_data/ILSVRC2012/val', transform=transform) 
        else:
            dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def build_taskonomy(is_train, args):
    transform = transforms.Compose([
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225))])

    # def __init__(self, img_types, data_dir='/data/zitianchen/taskonomy_medium', 
    #     partition='train', transform=None, resize_scale=None, crop_size=None, fliplr=False):

    if is_train:
        dataset = TaskonomyDataset(args.img_types, partition='train', transform=transform, resize_scale=256, crop_size=224, fliplr=True)
    else:
        dataset = TaskonomyDataset(args.img_types, partition='test', transform=transform, resize_scale=256, crop_size=224)
    return dataset

