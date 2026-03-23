import torch
import tqdm
import json
from qwen_vl_utils import process_vision_info

class InferenceEngine:
    prompt_template = None # prompt template placeholder
    device = None # device placeholder
    model = None # model placeholder
    tokenizer = None # tokenizer placeholder
    processor = None # processor placeholder
    batchsize = None # batchsize placeholder
    num_queries = None # num_queries placeholder
    yes_token_id = None # Yes token id placeholder
    no_token_id = None # No token id placeholder
    tau = None # temperature placeholder
    max_new_tokens = None # max_new_tokens placeholder
    is_swap = False # whether to swap the query and candidate images
    return_scores = False # whether to return the scores
    
    @staticmethod
    def init(prompt_template, device, model, tokenizer, processor, batchsize, num_queries,
             yes_token_id, no_token_id, tau, max_new_tokens, is_swap=False, return_scores=False):
        InferenceEngine.prompt_template = InferenceEngine._parse_prompt_template(prompt_template)
        InferenceEngine.device = device
        InferenceEngine.model = model
        InferenceEngine.tokenizer = tokenizer
        InferenceEngine.processor = processor
        InferenceEngine.batchsize = batchsize
        InferenceEngine.num_queries = num_queries
        InferenceEngine.yes_token_id = yes_token_id
        InferenceEngine.no_token_id = no_token_id
        InferenceEngine.tau = tau
        InferenceEngine.max_new_tokens = max_new_tokens
        InferenceEngine.is_swap = is_swap # whether to swap the query and candidate images
        InferenceEngine.return_scores = return_scores # whether to return the scores
        
    @staticmethod
    def _parse_prompt_template(prompt_template):
        if prompt_template == "":
            print('Using void template.')
            return None
        
        with open(prompt_template, 'r') as f:
            template = json.load(f)
        print('Prompt template loaded from:', prompt_template)
        return template

    @staticmethod
    def _generate_message(data):
        query = data['query']
        candidates = data['candidates']
        resized_height = data['resized_height']
        resized_width = data['resized_width']
        messages = []
        for cand in candidates:
            if InferenceEngine.prompt_template is not None:
                task_desc = InferenceEngine.prompt_template['task_description']
                rules = '\n'.join(InferenceEngine.prompt_template['rules'] + [InferenceEngine.prompt_template['response_format']])
                msg = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": task_desc
                                },
                                {
                                    "type": "text",
                                    "text": "Query:"
                                },
                                {
                                    "type": "image",
                                    "image": f"{query}" if not InferenceEngine.is_swap else f"{cand}",
                                    "resized_height": resized_height,
                                    "resized_width": resized_width,
                                },
                                {
                                    "type": "text",
                                    "text": " Candidate:"
                                },
                                {
                                    "type": "image",
                                    "image": f"{cand}" if not InferenceEngine.is_swap else f"{query}",
                                    "resized_height": resized_height,
                                    "resized_width": resized_width,
                                },
                                {
                                    "type": "text",
                                    "text": rules
                                }
                            ],
                        }, # NOTE: no assistant response in testing
                    ]
            else:
                msg = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{query}",
                            "resized_height": resized_height,
                            "resized_width": resized_width,
                        },
                        {
                            "type": "image",
                            "image": f"{cand}",
                            "resized_height": resized_height,
                            "resized_width": resized_width,
                        },
                    ],
                } # NOTE: no assistant response in testing
            ]
            messages.append(msg)
        return messages

    @staticmethod
    def _convert_to_standard_input_format(messages):
        InferenceEngine.tokenizer.padding_side = "right"
        texts = InferenceEngine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = InferenceEngine.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt")
        
        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
        else:
            attention_mask = None
        
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
        else:
            pixel_values = None

        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw']
        else:
            image_grid_thw = None

        input_dict = {
            "input_ids": inputs['input_ids'],
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        return input_dict

    @staticmethod
    def _run_generate(input_dict):
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                input_dict = {k: v.to('cuda') for k, v in input_dict.items()}
                output = InferenceEngine.model.generate(
                    **input_dict,
                    max_new_tokens=2,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )
                logits = output.scores[0]
                yes_logits = logits[:, InferenceEngine.yes_token_id]
                no_logits = logits[:, InferenceEngine.no_token_id]
                yes_and_no_logits = torch.stack([no_logits, yes_logits], dim=1)
                probs = yes_and_no_logits.softmax(dim=1)
        return probs
    
    @staticmethod
    def post_process(probs):
        if InferenceEngine.return_scores:
            # return the scores of "Yes" and "No"
            return probs # [num_queries, num_candidates, 2]
        
        # only return the probability of "No"
        probs = probs[:, :, 0] # [num_queries, num_candidates]
        return probs
    
    @staticmethod
    def generate_messages(data):
        messages = []
        for d in tqdm.tqdm(data, desc='Generating input messages'):
            messages.extend(InferenceEngine._generate_message(d))
        return messages
    
    @staticmethod
    def collate_fn(batch):
        input_dict = InferenceEngine._convert_to_standard_input_format(batch)
        return input_dict
    
    @staticmethod
    def generate_per_process(test_loader):
        outputs = []
        with torch.no_grad():
            for input_dict in tqdm.tqdm(test_loader, desc='Generating responses'):
                input_dict = {k: v.to(InferenceEngine.device) for k, v in input_dict.items()}
                output = InferenceEngine.model.generate(
                    **input_dict,
                    max_new_tokens=InferenceEngine.max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )
                logits = output.scores[0]
                yes_logits = logits[:, InferenceEngine.yes_token_id]
                no_logits = logits[:, InferenceEngine.no_token_id]
                yes_and_no_logits = torch.stack([no_logits, yes_logits], dim=1)
                if InferenceEngine.return_scores:
                    probs = yes_and_no_logits
                else:
                    probs = yes_and_no_logits.div(InferenceEngine.tau).softmax(dim=1)
                outputs.append(probs)
        outputs = torch.cat(outputs, dim=0) # [num_queries*num_candidates/num_proc, 2]
        return outputs
