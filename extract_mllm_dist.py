import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import argparse
import numpy as np
import torch
import time
from datetime import timedelta
from utils.tools import set_seed
from utils.logger import timestamp_logger
from utils.metrics import eval_func
from engine.inference.inference_engine import InferenceEngine
from engine.inference.inference_dataset import InferenceDataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import LoraConfig, TaskType, get_peft_model
import accelerate
from accelerate import Accelerator, InitProcessGroupKwargs

MODEL_DICT = {
    "Qwen2-VL-2B-Instruct": Qwen2VLForConditionalGeneration,
    "Qwen2.5-VL-7B-Instruct": Qwen2_5_VLForConditionalGeneration
}

def load_checkpoint(model, checkpoint_path):
    # Load LoRA weights
    param_dict = torch.load(checkpoint_path, map_location="cpu")
    cnt = 0
    loaded_keys = []
    for k, v in param_dict.items():
        model.state_dict()[k].copy_(v)
        loaded_keys.append(k)
        cnt += 1
    return model, loaded_keys, cnt

def get_query_candidates(distmat, paths, topk, preview_num=0):
    results = []
    gallery_indices = []
    num_queries = distmat.shape[0]

    for i in range(num_queries):
        query_path = paths[i]
        g_indices = np.argsort(distmat[i])[:topk]
        g_paths = [paths[num_queries+idx] for idx in g_indices]
        results.append({
            'query': query_path,
            'candidates': g_paths,
            'resized_height': args.height,
            'resized_width': args.width
        })
        gallery_indices.append(g_indices)

    gallery_indices = torch.stack(gallery_indices, dim=0) # [num_queries, topk]

    if preview_num > 0:
        print(f'query-gallery preview: {results[:preview_num]}, total {len(results)} samples')
        print(f'gallery_indices preview: {gallery_indices[:preview_num]}, shape: {gallery_indices.shape}')

    return results, gallery_indices

def main(args):
    set_seed(1234)

    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=args.timeout))]) # set timeout
    device = accelerator.device

    if accelerator.is_main_process:
        logger = timestamp_logger(osp.basename(__file__), save_dir=args.logdir)
        logger.info(f'Test arguments:')
        logger.info('-' * 50)
        for k, v in sorted(vars(args).items()):
            logger.info(f'{k} = {v}')
        logger.info('-' * 50)

    # load dist mat
    data = np.load(args.distmat_path)
    distmat = torch.from_numpy(data['distmat'])
    num_queries = distmat.shape[0]
    paths = data['paths']
    pids = data['pids']
    camids = data['camids']
    
    if accelerator.is_main_process:
        logger.info(f'Loaded distmat from {args.distmat_path}')
        logger.info(f'distmat shape: {distmat.shape}')
        logger.info(f'num_queries: {num_queries}')

    # prepare input data
    ds, gallery_indices = get_query_candidates(distmat, paths, args.topk, preview_num=args.preview_num) # gallery_indices: [num_queries, topk]

    # prepare model
    model = MODEL_DICT[osp.basename(args.model_path)].from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    if args.zeroshot:
        if accelerator.is_main_process:
            logger.info('Zero-shot mode enabled, no checkpoint is loaded.')
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]+args.extra_lora_layers,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        # Load LoRA/other trainable parameter weights
        model, loaded_keys, cnt = load_checkpoint(model, args.checkpoint)
        model = model.merge_and_unload() # Merge LoRA weights

        if accelerator.is_main_process:
            logger.info(f'Loaded keys: {loaded_keys}')
            logger.info(f'Loaded {cnt} keys from {args.checkpoint}')
    
    
    # prepare model
    model = accelerator.prepare(model)


    # inference locally
    model.eval()
    model = model.module if hasattr(model, 'module') else model # unwrap DDP
    InferenceEngine.init(
        prompt_template=args.prompt_template,
        device=device,
        model=model.to(device),
        tokenizer=tokenizer,
        processor=processor,
        batchsize=args.batchsize,
        num_queries=len(ds),
        yes_token_id=tokenizer('Yes').input_ids[0],
        no_token_id=tokenizer('No').input_ids[0],
        tau=args.tau,
        max_new_tokens = args.max_new_tokens,
        is_swap=args.swap_query_candidate,
        return_scores=args.return_scores,
    )
    if accelerator.is_main_process and args.swap_query_candidate:
        logger.info(f'Swap query and candidate images in each inquiry prompt.')

    # prepare test dataloader
    messages = InferenceEngine.generate_messages(ds) # length: num_queries * topk
    
    # inference
    if accelerator.is_main_process:
        logger.info(f'Start inference...')
        if args.return_scores:
            logger.info(f'Inference will return scores instead of probs.')
        start_time = time.monotonic()

    # split the whole dataset across processes
    # each process will only process its own subset of data
    accelerator.wait_for_everyone()
    with accelerator.split_between_processes(messages) as msgs_per_proc:
        print(f'Process {accelerator.process_index} is processing {len(msgs_per_proc)} samples')
        test_set_per_proc = InferenceDataset(msgs_per_proc)
        test_loader_per_proc = torch.utils.data.DataLoader(test_set_per_proc,
                                                          batch_size=args.batchsize,
                                                          num_workers=8,
                                                          shuffle=False,
                                                          pin_memory=True,
                                                          collate_fn=InferenceEngine.collate_fn)
        probs_per_proc = InferenceEngine.generate_per_process(test_loader_per_proc) # [num_queries*num_candidates/num_proc, 2]
        probs_per_proc = [probs_per_proc.cpu().tolist()] # wrapping for communication among processes
    probs = accelerate.utils.gather_object(probs_per_proc)

    # combine results from all processes
    if accelerator.is_main_process:
        probs = [torch.tensor(p) for p in probs]
        probs = torch.cat(probs, dim=0) # [num_queries*num_candidates, 2]
        probs = probs.unsqueeze(0).reshape(InferenceEngine.num_queries, -1, 2) # [num_queries, num_candidates, 2]
        probs = InferenceEngine.post_process(probs).cpu() # [num_queries, num_candidates]
        elapsed_time = timedelta(seconds=time.monotonic() - start_time)
    
        # reranking
        logger.info(f'{probs.shape} probabilities predicted in {elapsed_time}')

        # build new distmat
        output_path = args.new_distmat_output_path
        if output_path != '':
            np.savez(output_path, probs=probs.numpy(), gallery_indices=gallery_indices.numpy())
            logger.info(f'MLLM-enhanced probs saved to {output_path}')

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument('--timeout', type=int, default=3, help='timeout for DDP in hours')
    parser.add_argument('--prompt-template', type=str, default='', help='Prompt template file')
    parser.add_argument('--model-path', type=str, default='pretrained/Qwen/Qwen2-VL-2B-Instruct', help='Model path')
    parser.add_argument('--checkpoint', type=str, default='', help='LoRA checkpoint path')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=float, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--extra-lora-layers', type=str, nargs='*', default=[], help='Extra LoRA layers')
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size')
    parser.add_argument('--height', type=int, default=280, help='height of input image')
    parser.add_argument('--width', type=int, default=140, help='width of input image')
    parser.add_argument('--tau', type=float, default=1.0, help='temperature for softmax')
    parser.add_argument('--max-new-tokens', type=int, default=1, help='max new tokens for generation')
    parser.add_argument('--return-scores', action='store_true', help='return scores instead of probs')
    parser.add_argument('--zeroshot', action='store_true', help='Enable zero-shot mode')
    parser.add_argument('--swap-query-candidate', action='store_true', help='Swap query and candidate images in each inquiry prompt')
    
    # reranking arguments
    parser.add_argument('--distmat-path', type=str, default='', help='path to distmat')
    parser.add_argument('--preview-num', type=int, default=0, help='number of samples to preview')
    parser.add_argument('--topk', type=int, default=10, help='topk for reranking')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha for fusion, weight for new distmat')
    parser.add_argument('--logdir', type=str, default='logs/test_reranking', help='log directory')
    parser.add_argument('--new-distmat-output-path', type=str, default='', help='output directory for new distmat')
    args = parser.parse_args()
    main(args)