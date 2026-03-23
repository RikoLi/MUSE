import json
import random
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info

class PairwiseRerankingDatasetInterface:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trainset_path = cfg.DATASETS.PAIRWISE_TRAINSET_PATH
        self.valset_path = cfg.DATASETS.PAIRWISE_VALSET_PATH
        self.truncate_ratio = cfg.DATASETS.PAIRWISE_TRAINSET_TRUNCATE_RATIO # truncate ratio for training set samples
        self._parse_dataset()
        self._summarize()

    def _parse_dataset(self):
        with open(self.trainset_path, "r") as f:
            trainset = json.load(f) # dataset is a list of dict
        random.shuffle(trainset)

        # load validation set if provided
        if self.valset_path == '':
            valset = []
        else:
            with open(self.valset_path, "r") as f:
                valset = json.load(f)
        random.shuffle(valset)

        if self.truncate_ratio < 1.0:
            self.train = trainset[:int(self.truncate_ratio * len(trainset))] # truncate training set
            print(f"Truncated training set to {len(self.train)} ({self.truncate_ratio:.2%}) samples")
        else:
            self.train = trainset
            print(f"Using full training set with {len(self.train)} samples")
        self.val = valset
    
    def _summarize(self):
        self.num_pos_pairs_train = sum([1 for item in self.train if item["match"] == "Yes"])
        self.num_neg_pairs_train = sum([1 for item in self.train if item["match"] == "No"])
        self.num_pos_pairs_val = sum([1 for item in self.val if item["match"] == "Yes"])
        self.num_neg_pairs_val = sum([1 for item in self.val if item["match"] == "No"])
        print('----------------------------------------')
        print(f"* Dataset summary *")
        print(f"Train set samples: {len(self.train)}, positive pairs: {self.num_pos_pairs_train} ({self.num_pos_pairs_train/len(self.train):.2%}), negative pairs: {self.num_neg_pairs_train} ({self.num_neg_pairs_train/len(self.train):.2%})")
        if len(self.val) != 0:
            print(f"Val set samples: {len(self.val)}, positive pairs: {self.num_pos_pairs_val} ({self.num_pos_pairs_val/len(self.val):.2%}), negative pairs: {self.num_neg_pairs_val} ({self.num_neg_pairs_val/len(self.val):.2%})")
        else:
            print('No validation set provided.')
        print('----------------------------------------')
        
class PairwiseRerankingDatasetTestInterface:
    def __init__(self, cfg):
        self.cfg = cfg
        self.testset_path = cfg.DATASETS.PAIRWISE_TESTSET_PATH
        self._parse_dataset()
        self._summarize()

    def _parse_dataset(self):
        with open(self.testset_path, "r") as f:
            testset = json.load(f) # dataset is a list of dict
        random.shuffle(testset)
        self.test = testset
    
    def _summarize(self):
        self.num_pos_pairs_test = sum([1 for item in self.test if item["match"] == "Yes"])
        self.num_neg_pairs_test = sum([1 for item in self.test if item["match"] == "No"])
        print('----------------------------------------')
        print(f"* Dataset summary *")
        print(f"Test set samples: {len(self.test)}, positive pairs: {self.num_pos_pairs_test} ({self.num_pos_pairs_test/len(self.test):.2%}), negative pairs: {self.num_neg_pairs_test} ({self.num_neg_pairs_test/len(self.test):.2%})")
        print('----------------------------------------')

class PairwiseRerankTrainingDataset(Dataset):
    """Dataset for pair-wise reranking training."""
    def __init__(self, dataset, tokenizer, processor, image_size, prompt_template=''):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_size = image_size
        self.prompt_template = self._load_template(prompt_template)
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id
        self.IM_START_TOKEN_ID = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.IM_END_TOKEN_ID = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.IGNORE_TOKEN_ID = -100
        
    def _load_template(self, prompt_template):
        if prompt_template == '':
            print('Using void template.')
            return None

        with open(prompt_template, "r") as f:
            template = json.load(f)
            
        return template

    def __len__(self):
        return len(self.dataset)
    
    def _generate_message(self, query_path, candidate_path, answer):
        if self.prompt_template is not None:
            task_desc = self.prompt_template['task_description']
            rules = '\n'.join(self.prompt_template['rules'] + [self.prompt_template['response_format']])     
            message = [
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
                            "image": f"{query_path}",
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                        {
                            "type": "text",
                            "text": " Candidate:"
                        },
                        {
                            "type": "image",
                            "image": f"{candidate_path}",
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                        {
                            "type": "text",
                            "text": rules
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{answer}"},
                    ]
                }
            ]
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{query_path}",
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                        {
                            "type": "image",
                            "image": f"{candidate_path}",
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{answer}"},
                    ]
                }
            ]
        return message
    
    def _extract_inputs(self, inputs):
        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
            attention_mask = attention_mask[0] # remove batch dimension
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

        input_ids = inputs['input_ids'][0] # remove batch dimension
        labels = input_ids.clone()
        labels[labels == self.PAD_TOKEN_ID] = self.IGNORE_TOKEN_ID
        
        # mask out non-response tokens for SFT
        # no label shifting, since it will be handled in the model
        start_ids = (labels == self.IM_START_TOKEN_ID).nonzero(as_tuple=True)[0]
        last_start_id = start_ids[-1] # start of response contents
        labels[:last_start_id] = self.IGNORE_TOKEN_ID # mask out all tokens before response

        return input_ids, labels, attention_mask, pixel_values, image_grid_thw 
    
    def _convert_to_standard_input_format(self, message):
        self.processor.tokenizer.padding_side = 'right'
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        return inputs
    
    def __getitem__(self, index):
        query_path = self.dataset[index]["query"]
        candidate_path = self.dataset[index]["candidate"]
        gt_answer = self.dataset[index]["match"]
        message = self._generate_message(query_path, candidate_path, gt_answer)
        
        inputs = self._convert_to_standard_input_format(message)
        input_ids, labels, attention_mask, pixel_values, image_grid_thw = self._extract_inputs(inputs)
        
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }
        
        return input_dict, gt_answer
    
class PairwiseRerankTestingDataset(Dataset):
    """Dataset for pair-wise reranking testing."""
    def __init__(self, dataset, tokenizer, processor, image_size, prompt_template=''):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_size = image_size
        self.prompt_template = self._load_template(prompt_template)
        self.PAD_TOKEN_ID = self.tokenizer.pad_token_id
        self.IGNORE_TOKEN_ID = -100
        
    def _load_template(self, prompt_template):
        if prompt_template == '':
            print('Using void template.')
            return None

        with open(prompt_template, "r") as f:
            template = json.load(f)
            
        return template

    def __len__(self):
        return len(self.dataset)
    
    def _generate_message(self, query_path, candidate_path):
        if self.prompt_template is not None:
            task_desc = self.prompt_template['task_description']
            rules = '\n'.join(self.prompt_template['rules'] + [self.prompt_template['response_format']])     
            message = [
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
                            "image": f"{query_path}",
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                        {
                            "type": "text",
                            "text": " Candidate:"
                        },
                        {
                            "type": "image",
                            "image": f"{candidate_path}",
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                        {
                            "type": "text",
                            "text": rules
                        }
                    ],
                }, # NOTE: no assistant response in testing
            ]
        else:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{query_path}",
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                        {
                            "type": "image",
                            "image": f"{candidate_path}",
                            "resized_height": self.image_size[0],
                            "resized_width": self.image_size[1],
                        },
                    ],
                } # NOTE: no assistant response in testing
            ]
        return message
    
    def _extract_inputs(self, inputs):
        if 'attention_mask' in inputs:
            attention_mask = inputs['attention_mask']
            attention_mask = attention_mask[0] # remove batch dimension
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

        input_ids = inputs['input_ids'][0] # remove batch dimension
        
        return input_ids, attention_mask, pixel_values, image_grid_thw 
    
    def _convert_to_standard_input_format(self, message):
        self.processor.tokenizer.padding_side = 'right'
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) # NOTE: add generation prompt for testing
        image_inputs, video_inputs = process_vision_info(message)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        return inputs
    
    def __getitem__(self, index):
        query_path = self.dataset[index]["query"]
        candidate_path = self.dataset[index]["candidate"]
        gt_answer = self.dataset[index]["match"]
        message = self._generate_message(query_path, candidate_path)
        
        inputs = self._convert_to_standard_input_format(message)
        input_ids, attention_mask, pixel_values, image_grid_thw = self._extract_inputs(inputs)
        
        input_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }
        
        return input_dict, gt_answer