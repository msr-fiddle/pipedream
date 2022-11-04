# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import warnings
from contextlib import contextmanager
import hashlib
import os
import shutil
import tempfile
import torch
import gzip
import tarfile
import zipfile
from torchvision.datasets.folder import ImageFolder

ARCHIVE_META = {
    'train': ('ILSVRC2012_img_train.tar', '1d675b47d978889d74fa0da5fadfb00e'),
    'val': ('ILSVRC2012_img_val.tar', '29b22e2961454d5413ddabcf34fc5622'),
    'devkit': ('ILSVRC2012_devkit_t12.tar.gz', 'fa75699e90414af021442c21a62c3abf')
}

META_FILE = "meta.bin"


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if _is_tar(from_path):
        with tarfile.open(from_path, 'r') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
    elif _is_targz(from_path):
        with tarfile.open(from_path, 'r:gz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
    elif _is_tarxz(from_path) and PY3:
        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, 'r:xz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=to_path)
    elif _is_gzip(from_path):
        to_path = os.path.join(to_path, os.path.splitext(os.path.basename(from_path))[0])
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        with zipfile.ZipFile(from_path, 'r') as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def verify_str_arg(value, arg=None, valid_values=None, custom_msg=None):
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ("Unknown value '{value}' for argument {arg}. "
                   "Valid values are {{{valid_values}}}.")
            msg = msg.format(value=value, arg=arg,
                             valid_values=iterable_to_str(valid_values))
        raise ValueError(msg)

    return value


class ImageNet(ImageFolder):
    """`ImageNet <http://image-net.org/>`_ 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        imgs (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, split='train', download=None, **kwargs):
        if download is True:
            msg = ("The dataset is no longer publicly accessible. You need to "
                   "download the archives externally and place them in the root "
                   "directory.")
            raise RuntimeError(msg)
        elif download is False:
            msg = ("The use of the download flag is deprecated, since the dataset "
                   "is no longer publicly accessible.")
            warnings.warn(msg, RuntimeWarning)
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]

        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def parse_archives(self):
        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not os.path.isdir(self.split_folder):
            if self.split == 'train':
                parse_train_archive(self.root)
            elif self.split == 'val':
                parse_val_archive(self.root)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def load_meta_file(root, file=None):
    if file is None:
        file = META_FILE
    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file)
    else:
        msg = ("The meta file {} is not present in the root directory or is corrupted. "
               "This file is automatically created by the ImageNet dataset.")
        raise RuntimeError(msg.format(file, root))


def _verify_archive(root, file, md5):
    if not check_integrity(os.path.join(root, file), md5):
        msg = ("The archive {} is not present in the root directory or is corrupted. "
               "You need to download it externally and place it in {}.")
        raise RuntimeError(msg.format(file, root))


def parse_devkit_archive(root, file=None):
    """Parse the devkit archive of the ImageNet2012 classification dataset and save
    the meta information in a binary file.

    Args:
        root (str): Root directory containing the devkit archive
        file (str, optional): Name of devkit archive. Defaults to
            'ILSVRC2012_devkit_t12.tar.gz'
    """
    import scipy.io as sio

    def parse_meta_mat(devkit_root):
        metafile = os.path.join(devkit_root, "data", "meta.mat")
        meta = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children)
                if num_children == 0]
        idcs, wnids, classes = list(zip(*meta))[:3]
        classes = [tuple(clss.split(', ')) for clss in classes]
        idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
        return idx_to_wnid, wnid_to_classes

    def parse_val_groundtruth_txt(devkit_root):
        file = os.path.join(devkit_root, "data",
                            "ILSVRC2012_validation_ground_truth.txt")
        with open(file, 'r') as txtfh:
            val_idcs = txtfh.readlines()
        return [int(val_idx) for val_idx in val_idcs]

    @contextmanager
    def get_tmp_dir():
        tmp_dir = tempfile.mkdtemp()
        try:
            yield tmp_dir
        finally:
            shutil.rmtree(tmp_dir)

    archive_meta = ARCHIVE_META["devkit"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    with get_tmp_dir() as tmp_dir:
        extract_archive(os.path.join(root, file), tmp_dir)

        devkit_root = os.path.join(tmp_dir, "ILSVRC2012_devkit_t12")
        idx_to_wnid, wnid_to_classes = parse_meta_mat(devkit_root)
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))


def parse_train_archive(root, file=None, folder="train"):
    """Parse the train images archive of the ImageNet2012 classification dataset and
    prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the train images archive
        file (str, optional): Name of train images archive. Defaults to
            'ILSVRC2012_img_train.tar'
        folder (str, optional): Optional name for train images folder. Defaults to
            'train'
    """
    archive_meta = ARCHIVE_META["train"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]

    _verify_archive(root, file, md5)

    train_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), train_root)

    archives = [os.path.join(train_root, archive) for archive in os.listdir(train_root)]
    for archive in archives:
        extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)


def parse_val_archive(root, file=None, wnids=None, folder="val"):
    """Parse the validation images archive of the ImageNet2012 classification dataset
    and prepare it for usage with the ImageNet dataset.

    Args:
        root (str): Root directory containing the validation images archive
        file (str, optional): Name of validation images archive. Defaults to
            'ILSVRC2012_img_val.tar'
        wnids (list, optional): List of WordNet IDs of the validation images. If None
            is given, the IDs are loaded from the meta file in the root directory
        folder (str, optional): Optional name for validation images folder. Defaults to
            'val'
    """
    archive_meta = ARCHIVE_META["val"]
    if file is None:
        file = archive_meta[0]
    md5 = archive_meta[1]
    if wnids is None:
        wnids = load_meta_file(root)[1]

    _verify_archive(root, file, md5)

    val_root = os.path.join(root, folder)
    extract_archive(os.path.join(root, file), val_root)

    images = sorted([os.path.join(val_root, image) for image in os.listdir(val_root)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(val_root, wnid))

    for wnid, img_file in zip(wnids, images):
        shutil.move(img_file, os.path.join(val_root, wnid, os.path.basename(img_file)))
