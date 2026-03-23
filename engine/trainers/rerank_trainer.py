import os
import time
from datetime import timedelta
import torch
import torch.nn.functional as F
import tqdm
import tensorboardX as tbx
from typing import *
from .base_trainer import AccelerateBaseTrainer
from utils.meter import AverageMeter
from utils.metrics import compute_accuracy

class RerankTrainer(AccelerateBaseTrainer):
    """Trainer for Huggingface model."""
    def __init__(self, cfg, logger, model, optimizer, scheduler, evaluator, processor, tokenizer, accelerator):
        super().__init__(cfg, logger, model, optimizer, scheduler, evaluator, accelerator)
        self.processor = processor
        self.tokenizer = tokenizer

    def train(self, train_loader, val_loader, start_epoch=1):
        # enable tensorboardx and timer
        if self.accelerator.is_main_process:
            tbx_writer = tbx.SummaryWriter(log_dir=os.path.join(self.cfg.OUTPUT_DIR))
            start_time = time.monotonic()
        
        # evaluation metrics
        meters = {k: AverageMeter() for k in ["total"]}

        # enable accelerator for dataloaders
        self.model, self.optimizer, self.scheduler, train_loader, val_loader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, train_loader, val_loader)
        if self.accelerator.is_main_process:
            self.logger.info(f"Train loader length: {len(train_loader)}")
            self.logger.info(f"Val loader length: {len(val_loader)}")
        
        # training loop
        self.accelerator.wait_for_everyone()
        for epoch in range(start_epoch, self.cfg.SOLVER.MAX_EPOCHS + 1):
            # reset meters
            for v in meters.values():
                v.reset()

            # schedule learning rate
            if self.accelerator.is_main_process:
                if self.scheduler is not None:
                    self.scheduler.step(epoch)
                    self.logger.info(f"Learning rate is changed to {self.scheduler._get_lr(epoch)[0]:.2e}")
                else:
                    self.logger.info(f"Learning rate: {self.cfg.SOLVER.BASE_LR:.2e}")
                
            # forwarding
            self.model.train()
            tqdm_loader = tqdm.tqdm(train_loader, total=len(train_loader))
            truncated_iters = self.cfg.DATALOADER.MAX_ITER if self.cfg.DATALOADER.MAX_ITER > 0 else len(tqdm_loader)
            if self.accelerator.is_main_process:
                self.logger.info(f"Truncated iterations: {truncated_iters}")
            for i, (batch_input, gt_answer) in enumerate(tqdm_loader):
                # check if reach max iterations
                if i >= truncated_iters:
                    break
                
                
                # prepare inputs
                batch_input["image_grid_thw"] = batch_input["image_grid_thw"].reshape(-1, 3) # NOTE: flatten first two dimensions as B*n_images for Qwen2VL RoPE
                batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
                loss_dict = {}
                
                # feed into model
                output = self.model(**batch_input)
                loss_dict["total"] = output.loss
                
                # loss computation
                loss = sum(loss_dict.values())
                
                # backwarding
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # update meters
                meters["total"].update(loss_dict["total"].item(), n=batch_input["labels"].shape[0])
                
                # log to tensorboard
                all_loss = self.accelerator.gather_for_metrics([loss.detach().cpu().item()])
                if self.cfg.SOLVER.TBX_LOG_PERIOD > 0 and i % self.cfg.SOLVER.TBX_LOG_PERIOD == 0:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        tbx_writer.add_scalar("train/loss", sum(all_loss)/len(all_loss), epoch * len(tqdm_loader) + i)
                
                # update tqdm
                desc = f"Epoch [{epoch}/{self.cfg.SOLVER.MAX_EPOCHS}]: Loss={meters['total'].avg:.4e}, lr={self.scheduler._get_lr(epoch)[0]:.2e}"
                tqdm_loader.set_description(desc=desc)
            
            
            if self.accelerator.is_main_process:
                self.logger.info(desc)

            # evaluation
            if len(val_loader) > 0 and (epoch % self.cfg.SOLVER.EVAL_PERIOD == 0):
                self.eval(epoch, val_loader)

            # checkpoint
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process and (epoch % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0 or epoch == self.cfg.SOLVER.MAX_EPOCHS):
                self.save_checkpoint(epoch)

        # close tensorboard and timer
        if self.accelerator.is_main_process:
            tbx_writer.close()
            end_time = time.monotonic()
            elapsed_time = timedelta(seconds=end_time - start_time)
            self.logger.info(f"Training finished in {elapsed_time}")

    def eval(self, epoch, val_loader):
        torch.cuda.empty_cache()
        self.model.eval()
        
        # get forwarding entrance
        model_forward = self.model.module if hasattr(self.model, "module") else self.model

        # start timer
        if self.accelerator.is_main_process:
            start_time = time.monotonic()
        
        # prepare "yes" and "no" token ids
        yes_token_id = self.tokenizer("Yes").input_ids[0]
        no_token_id = self.tokenizer("No").input_ids[0]
        
        # evaluation loop
        all_logits = []
        all_gt_answers = []
        tqdm_loader = tqdm.tqdm(val_loader, total=len(val_loader), desc="Validation")
        with torch.no_grad():
            for i, (batch_input, gt_answer) in enumerate(tqdm_loader):
                batch_input = {k: v.to(self.device) for k, v in batch_input.items()}
                batch_input["image_grid_thw"] = batch_input["image_grid_thw"].reshape(-1, 3) # NOTE: flatten first two dimensions as B*n_images for Qwen2VL RoPE
                output = model_forward.generate(**batch_input,
                                                    max_new_tokens=2, # we expect the model to generate "Yes" or "No" + <|im_end|>, total 2 new tokens generated
                                                    output_scores=True,
                                                    return_dict_in_generate=True,
                                                    do_sample=False, # greedy decoding, keep stable outputs
                                                    temperature=None,
                                                    top_p=None,
                                                    top_k=None)
                
                # model.generate() returns a dict with keys: "sequences", "scores", "past_key_values"
                # "sequences": torch.Tensor, generated token ids
                # "scores": Tuple[torch.Tensor], where each tensor is the logit of each generated token
                # "past_key_values": cached key-values for fast decoding
                logits = output.scores[0] # [B, vocab_size], first generated token should be "Yes" or "No"

                # gather "yes" and "no" probabilities from logits
                yes_logits = logits[:, yes_token_id]
                no_logits = logits[:, no_token_id]
                yes_and_no_logits = torch.stack([no_logits, yes_logits], dim=1) # [B, 2]

                # convert GT answer to value in {0, 1}
                gt_answer = torch.tensor([1 if ans == "Yes" else 0 for ans in gt_answer], device=self.device)

                # gather from subprocesses
                yes_and_no_logits = self.accelerator.gather_for_metrics(yes_and_no_logits)
                gt_answer = self.accelerator.gather_for_metrics(gt_answer)

                # collect this batch
                all_logits.append(yes_and_no_logits)
                all_gt_answers.append(gt_answer)
                
        # gather all results
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            all_logits = torch.cat(all_logits, dim=0) # [N, 2]
            all_gt_answers = torch.cat(all_gt_answers, dim=0) # [N]
            
            # compute evaluation accuracy & loss
            loss, acc = compute_accuracy(all_logits, all_gt_answers)
            self.logger.info(f"Validation after epoch {epoch}: Avg. loss={loss:.4e}, acc={acc:.1%}")

            # time elapsed
            end_time = time.monotonic()
            elapsed_time = timedelta(seconds=end_time - start_time)
            self.logger.info(f"Validation finished in {elapsed_time}")

    def save_checkpoint(self, epoch):
        model_to_save = self.accelerator.unwrap_model(self.model)
        saved_state_dict = {k: v.cpu() for k, v in model_to_save.named_parameters() if v.requires_grad}
        checkpoint_path = os.path.join(self.cfg.OUTPUT_DIR, f"trainable_params_epoch_{epoch}.pth")
        self.accelerator.save(saved_state_dict, checkpoint_path)
        self.logger.info(f"LoRA checkpoint saved at {checkpoint_path}")