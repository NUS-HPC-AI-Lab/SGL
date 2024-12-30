import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torchvision.ops.boxes import box_area
from tqdm import tqdm
from transformers import AutoTokenizer
import misc
import math
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
import time

ds_collections = {
    'refcoco_val': 'data/refcoco/refcoco_val.jsonl',
    'refcoco_testA': 'data/refcoco/refcoco_testA.jsonl',
    'refcoco_testB': 'data/refcoco/refcoco_testB.jsonl',
    'refcoco+_val': 'data/refcoco/refcoco+_val.jsonl',
    'refcoco+_testA': 'data/refcoco/refcoco+_testA.jsonl',
    'refcoco+_testB': 'data/refcoco/refcoco+_testB.jsonl',
    'refcocog_val': 'data/refcoco/refcocog_val.jsonl',
    'refcocog_test': 'data/refcoco/refcocog_test.jsonl',
}


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    texts = [_['text'] for _ in batches]
    bboxes = [_['bbox'] for _ in batches]
    hws = [_['hw'] for _ in batches]
    return pixel_values, texts, bboxes, hws


class RefCOCODataset(torch.utils.data.Dataset):

    def __init__(self, test, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.datas = open(test).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx].strip())
        image = data['image']
        text = data['sent']
        bbox = data['bbox']

        w, h = data['width'], data['height']

        image = Image.open(image).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'text': self.prompt.format(text),
            'pixel_values': pixel_values,
            'bbox': bbox,
            'hw': (h, w),
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = misc.get_rank()
        self._world_size = misc.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    print('prompt:', prompt)
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        dataset = RefCOCODataset(
            test=ds_collections[ds_name],
            prompt=prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=InferenceSampler(len(dataset)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn, tokenizer=small_model_tokenizer),
        )

        outputs = []
        for _, (pixel_values, questions, bboxes, hws) in enumerate(tqdm(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=100,
                min_new_tokens=1,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
            )

            generation_config["return_dict_in_generate"] = True
            generation_config["output_scores"] = True
            generation_config["output_attentions"] = True

            generation_config["consistency_config"] = generation_config.copy()
            generation_config["consistency_config"]["return_dict_in_generate"] = False
            generation_config["consistency_config"]["output_scores"] = False
            generation_config["consistency_config"]["output_attentions"] = False
            generation_config["consistency_config"]["large_model_prune_layer"] = 0.0
            generation_config["consistency_config"]["large_model_prune_ratio"] = args.consistency_token_ratio

            torch.cuda.synchronize()
            start = time.time()
            pred, scores, consistency_score, visual_token_importance  = small_model.chat(
                tokenizer=small_model_tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config,
                large_model=False
            )
            small_answers = [pred]


            scores = torch.concatenate(scores, dim=0)
            scores, _ = scores.softmax(dim=-1).max(dim=-1)
            original_confidence = math.pow(torch.prod(scores).item(), 1 / len(scores))
            original_confidences = [original_confidence]
            consistency_scores = [consistency_score.item()]

            torch.cuda.synchronize()
            end = time.time()
            small_model_times = [end - start]



            del generation_config['consistency_config'] 
            generation_config["return_dict_in_generate"] = False
            generation_config["output_scores"] = False
            generation_config["output_attentions"] = False
            generation_config["large_model_prune_layer"] = args.large_model_prune_layer
            generation_config["large_model_prune_ratio"] = args.large_model_prune_ratio
            generation_config['visual_token_importance'] = visual_token_importance

            torch.cuda.synchronize()
            start = time.time()
            pred = large_model.chat(
                tokenizer=large_model_tokenizer,
                pixel_values=pixel_values,
                question=questions[0],
                generation_config=generation_config,
                large_model=True
            )
            torch.cuda.synchronize()
            end = time.time()
            large_model_times = [end - start]

            large_answers = [pred]

            answers = large_answers

            print("a")






            for bbox, hw, answer, large_answer, small_answer, original_confidence,  consistency_score, small_model_time, large_model_time  in zip(bboxes, hws, answers, large_answers, small_answers, original_confidences, consistency_scores, small_model_times, large_model_times):
                outputs.append({
                    'answer': answer,
                    'gt_bbox': bbox,
                    'hw': hw,
                    'large_answer': large_answer,
                    'large_model_time': large_model_time,
                    'small_answer': small_answer,
                    'small_model_time':small_model_time,
                    'original_confidence': original_confidence,
                    'consistency_score': consistency_score
                })

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))

            correct = total_cnt = 0
            for i, output in enumerate(merged_outputs):
                predict_bbox = re.findall(PATTERN, output['answer'])
                try:
                    predict_bbox = (float(predict_bbox[0][0]), float(predict_bbox[0][1]), float(predict_bbox[0][2]),
                                    float(predict_bbox[0][3]))
                except:
                    predict_bbox = (0., 0., 0., 0.)
                target_bbox = torch.tensor(output['gt_bbox'],
                                           dtype=torch.float32).view(-1, 4)
                predict_bbox = torch.tensor(predict_bbox,
                                            dtype=torch.float32).view(-1, 4)
                if predict_bbox.sum() >= 4:
                    predict_bbox = predict_bbox / 1000
                predict_bbox[:, 0::2] *= output['hw'][1]
                predict_bbox[:, 1::2] *= output['hw'][0]
                iou, _ = box_iou(predict_bbox, target_bbox)
                iou = iou.item()
                total_cnt += 1
                if iou >= 0.5:
                    correct += 1

            print(f'Evaluating {ds_name} ...')
            print(f'Precision @ 1: {correct / total_cnt} \n')
            summaries.append([args.large_checkpoint, ds_name, f'Precision @ 1: {correct / total_cnt} \n'])

        torch.distributed.barrier()

    out_path = '_'.join(args.large_checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--small_checkpoint', type=str, default='')
    parser.add_argument('--large_checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='refcoco_val,refcoco_testA,refcoco_testB,'
                                                        'refcoco+_val,refcoco+_testA,refcoco+_testB,'
                                                        'refcocog_val,refcocog_test')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--large_model_prune_layer', type=float, default=0.3)
    parser.add_argument('--large_model_prune_ratio', type=float, default=0.3)
    parser.add_argument('--consistency_token_ratio', type=float, default=0.05)

    args = parser.parse_args()

    args.out_dir = os.path.join(args.out_dir, f"PruneLayer_{args.large_model_prune_layer}_PruneRatio_{args.large_model_prune_ratio}")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    misc.init_distributed_mode(args)
    
    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     world_size=int(os.getenv('WORLD_SIZE', '1')),
    #     rank=int(os.getenv('RANK', '0')),
    # )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    kwargs = {'device_map': 'auto'} if args.auto else {}
    PATTERN = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')
    # smalll model 
    small_model_tokenizer = AutoTokenizer.from_pretrained(args.small_checkpoint, trust_remote_code=True, use_fast=False)
    small_config = InternVLChatConfig.from_json_file(f"{args.small_checkpoint}/config.json")
    small_model_size = args.small_checkpoint.split("-")[-1]
    if small_model_size in ['1B','40B']:
        small_config.llm_config._attn_implementation = 'eager' 
    else:
        small_config.llm_config.attn_implementation = 'eager'
    small_config.vision_config.use_flash_attn = True

    small_model = InternVLChatModel.from_pretrained(
        args.small_checkpoint, config=small_config, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        small_model = small_model.cuda()

    
    # large model
    large_model_tokenizer = AutoTokenizer.from_pretrained(args.large_checkpoint, trust_remote_code=True, use_fast=False)
    large_config = InternVLChatConfig.from_json_file(f"{args.large_checkpoint}/config.json")
    large_model_size = args.large_checkpoint.split("-")[-1]
    if large_model_size in ['1B','40B']:
        large_config.llm_config._attn_implementation = 'eager' 
    else:
        large_config.llm_config.attn_implementation = 'eager' 
    # our method also supports inference with flashattn by setting attn_implementation to 'flash_attention_2'
    large_config.vision_config.use_flash_attn = True

    large_model = InternVLChatModel.from_pretrained(
        args.large_checkpoint, config=large_config, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, **kwargs).eval()
    if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
        large_model = large_model.cuda()
        
    image_size = large_model.config.force_image_size or large_model.config.vision_config.image_size
    use_thumbnail = large_model.config.use_thumbnail
    prompt = 'Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>'

    total_params = sum(p.numel() for p in small_model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {small_model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    evaluate_chat_model()
