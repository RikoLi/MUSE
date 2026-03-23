import torch
from torch.utils.data import DataLoader
from engine.datasets.rerank_dataset import (
    PairwiseRerankingDatasetInterface,
    PairwiseRerankingDatasetTestInterface,
    PairwiseRerankTrainingDataset,
    PairwiseRerankTestingDataset
)

FACTORY = {
    "pairwise_reranking": PairwiseRerankingDatasetInterface,
}

def rerank_collator(batch):
    n_batch_items = len(batch[0])
    assert n_batch_items == 2, "batch should be a tuple of (input_dict, gt_answer)"

    # stack each item of input_dict into a batched tensor
    input_dict, gt_answer = zip(*batch)
    input_dict = {k: torch.stack([inp[k] for inp in input_dict]) for k in input_dict[0]}
    return input_dict, gt_answer

def make_rerank_dataloader(cfg, tokenizer, processor):
    dataset = FACTORY[cfg.DATASETS.NAME](cfg)
    trainset = PairwiseRerankTrainingDataset(dataset.train, tokenizer, processor, cfg.INPUT.IMAGE_SIZE, cfg.DATASETS.PROMPT_TEMPLATE)
    valset = PairwiseRerankTestingDataset(dataset.val, tokenizer, processor, cfg.INPUT.IMAGE_SIZE, cfg.DATASETS.PROMPT_TEMPLATE)

    train_loader = DataLoader(trainset,
                              batch_size=cfg.SOLVER.BATCHSIZE,
                              shuffle=True,
                              num_workers=cfg.DATALOADER.NUM_WORKERS,
                              pin_memory=True,
                              collate_fn=rerank_collator)
    val_loader = DataLoader(valset,
                            batch_size=cfg.TEST.BATCHSIZE,
                            shuffle=False,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            pin_memory=True,
                            collate_fn=rerank_collator)
    
    return train_loader, val_loader

def make_rerank_test_dataloader(cfg, tokenizer, processor):
    dataset = PairwiseRerankingDatasetTestInterface(cfg)
    testset = PairwiseRerankTestingDataset(dataset.test, tokenizer, processor, cfg.INPUT.IMAGE_SIZE, cfg.DATASETS.PROMPT_TEMPLATE)

    test_loader = DataLoader(testset,
                            batch_size=cfg.TEST.BATCHSIZE,
                            shuffle=False,
                            num_workers=cfg.DATALOADER.NUM_WORKERS,
                            collate_fn=rerank_collator)
    
    return test_loader