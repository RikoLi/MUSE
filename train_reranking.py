import os
os.environ["HF_DATASETS_CACHE"] = "./hf_caches"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from utils.tools import set_seed, save_running_config, set_partial_layer_trainable
from utils.logger import setup_logger
import torch
import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
from accelerate import Accelerator, InitProcessGroupKwargs
import torch
import argparse
from datetime import timedelta
from configs import hf_cfg as cfg
from engine.datasets.rerank_dataloader import make_rerank_dataloader
from engine.trainers.rerank_trainer import RerankTrainer
from engine.solvers.optimizers import make_optimizer
from engine.solvers.schedulers import create_cosine_scheduler

MODEL_DICT = {
    "Qwen2-VL-2B-Instruct": Qwen2VLForConditionalGeneration,
    "Qwen2.5-VL-7B-Instruct": Qwen2_5_VLForConditionalGeneration
}

def main(args):
    # set random seed
    set_seed(cfg.SOLVER.SEED)
    
    # set accelerator
    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=cfg.MODEL.DDP_TIMEOUT_HOURS))])
    
    # load config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # set other logger
    logger = setup_logger("logger", cfg.OUTPUT_DIR, if_train=True)
    if accelerator.is_main_process:
        if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # save config to output dir
        save_running_config(cfg, cfg.OUTPUT_DIR)
        


    # load pretrained model
    model_path = cfg.MODEL.PRETRAINED_MODEL_DIR
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = MODEL_DICT[os.path.basename(model_path)].from_pretrained(model_path, torch_dtype=torch.bfloat16)
    
    # create dataloaders
    train_loader, val_loader = make_rerank_dataloader(cfg, tokenizer, processor)

    # LoRA configs
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=cfg.LORA.TARGET_MODULES,
        r=cfg.LORA.RANK,
        lora_alpha=cfg.LORA.ALPHA,
        lora_dropout=cfg.LORA.DROPOUT,
        bias="none",
    )
    # equipped with LoRA
    peft_model = get_peft_model(model, lora_config)
    set_partial_layer_trainable(peft_model, cfg.MODEL.EXTRA_TRAINABLE_LAYERS)
    if accelerator.is_main_process:
        peft_model.print_trainable_parameters()

    # initialize optimizer
    optimizer = make_optimizer(cfg, peft_model, accelerator)
    scheduler = create_cosine_scheduler(optimizer,
                                        cfg.SOLVER.MAX_EPOCHS,
                                        cfg.SOLVER.LR_MIN,
                                        cfg.SOLVER.WARMUP_LR_INIT,
                                        cfg.SOLVER.WARMUP_T,
                                        noise_range=None)
    # trainer
    trainer = RerankTrainer(cfg,
                        logger=logger,
                        model=peft_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        evaluator=None,
                        processor=processor,
                        tokenizer=tokenizer,
                        accelerator=accelerator)
    
    trainer.train(train_loader, val_loader)
    
if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    main(args)
