import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .msmt17_v2 import MSMT17_V2
from .cuhk03_np import CUHK03_NP
from .viper import VIPeR
from .ilids import iLIDS
from .grid import GRID
from .prid import PRID
from .multi_source_dg import MultiSourceDG, MultiSourceDG_CS, ClassicalMultiSourceDG
from .preprocessing import RandomErasing
from .dataset import ImageDataset
from .sampler import RandomIdentitySampler

FACTORY = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17_V2,
    'cuhk03np': CUHK03_NP,
    'viper': VIPeR,
    'ilids': iLIDS,
    'grid': GRID,
    'prid': PRID,
    'msdg': MultiSourceDG,
    'msdg_cs': MultiSourceDG_CS,
    'classical_msdg': ClassicalMultiSourceDG
}

def collate_fn(batch):
    """
    # The input to collate_fn is a list, where the length of the list is the batch size,
    # and each element in the list is the result from __getitem__
    """
    path, imgs, pids, camids, viewids = zip(*batch)
    
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    return path, imgs, pids, camids, viewids

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
    print('Disable random crop & erase preprocessing for ReID training.')

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = FACTORY[cfg.DATASETS.NAMES[0]](root=cfg.DATASETS.ROOT_DIR) # single-source
    elif cfg.DATASETS.USE_NEW_MSDG_PROTOCOL:
        assert len(cfg.DATASETS.NAMES) > 1, 'Should contain more than one dataset under new MSDG protocol!'
        dataset = FACTORY['msdg_cs'](root=cfg.DATASETS.ROOT_DIR, cuhk_protocol='detected',
                                train_datasets=cfg.DATASETS.NAMES,
                                test_dataset=cfg.DATASETS.EVAL_DATASET) # multi-source
    else:
        dataset = FACTORY['classical_msdg'](root=cfg.DATASETS.ROOT_DIR,
                                            all_for_train=True,
                                            test_dataset=cfg.DATASETS.EVAL_DATASET)
    
    train_set = ImageDataset(dataset.train, train_transforms, return_path=True)
    train_set_normal = ImageDataset(dataset.train, val_transforms, return_path=True)
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms, return_path=True)

    num_classes = dataset.num_train_pids
    num_cams = dataset.num_train_cams
    num_views = dataset.num_train_vids

    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )

    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader_normal, train_loader, val_loader, len(dataset.query), num_classes, num_cams, num_views

def make_val_dataloader(cfg):
    """Only return a dataloader for test split."""
    
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = FACTORY[cfg.DATASETS.EVAL_DATASET](root=cfg.DATASETS.ROOT_DIR)
    val_set = ImageDataset(dataset.query+dataset.gallery, val_transforms, return_path=True)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )
    return val_loader, len(dataset.query)
