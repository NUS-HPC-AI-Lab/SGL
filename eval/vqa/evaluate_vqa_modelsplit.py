import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional

import torch
from internvl.model.internvl_chat import InternVLChatModel
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from textvqa_eval import TextVQAAccuracyEvaluator
from tqdm import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import misc
import math
from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
# import warnings
# warnings.filterwarnings('ignore', category=DeprecationWarning)

ds_collections = {
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val_ocr': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val_llava.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docvqa_val': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/test_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'gqa_testdev_llava': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/llava_gqa_testdev_balanced_qwen_format.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test_vlmevalkit.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'infographicsvqa_val': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/val.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_val_v1.0_withQT.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'infographicsvqa_test': {
        'train': 'data/infographicsvqa/train.jsonl',
        'test': 'data/infographicsvqa/test.jsonl',
        'annotation': 'data/infographicsvqa/infographicsVQA_test_v1.0.json',
        'metric': None,
        'max_new_tokens': 100,
    }
}


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]

    return pixel_values, questions, question_ids, annotations






class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, few_shot, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.test = open(test).readlines()
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.few_shot = few_shot
        self.max_num = max_num
        if few_shot > 0:
            self.train = open(train).readlines()
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = json.loads(self.test[idx].strip())
        image, question, question_id, annotation = data['image'], data[
            'question'], data['question_id'], data.get('answer', None)

        few_shot_prompt = ''
        if self.few_shot > 0:
            few_shot_samples = random.sample(self.train, self.few_shot)
            for sample in few_shot_samples:
                sample = json.loads(sample.strip())
                few_shot_prompt += self.prompt.format(
                    sample['image'],
                    sample['question']) + f" {sample['answer']}"

        image = Image.open(image).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        if len(self.prompt) != 0:
            question = question + ' ' + self.prompt
        return {
            'question_id': question_id,
            'question': question,
            'pixel_values': pixel_values,
            'annotation': annotation
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


def post_process(response):
    response = response.strip().split('.')[0].split(
        ',')[0].split('!')[0].lower()
    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]
    response = response.strip()
    return response


def evaluate_chat_model():
    base_prompt = 'Answer the question using a single word or phrase.'
    vizwiz_prompt = "When the provided information is insufficient, respond with 'Unanswerable'. "
    # infovqa_prompt = 'Answer the question directly.'
    infovqa_prompt = 'Answer the question using a single word or phrase.'
    ai2d_prompt = ''
    random.seed(args.seed)
    summaries = []

    out_path =  f"{args.small_checkpoint.split('-')[-1]}_{args.large_checkpoint.split('-')[-1]}"
    fold_path = os.path.join(args.out_dir, f'PruneLayer_{args.large_model_prune_layer}_PruneRatio_{args.large_model_prune_ratio}')
    os.makedirs(fold_path, exist_ok=True)



    for ds_name in args.datasets:
        if 'vizwiz' in ds_name:
            input_prompt = vizwiz_prompt + base_prompt
        elif 'ai2d' in ds_name:
            input_prompt = ai2d_prompt
        elif 'infographicsvqa' in ds_name:
            input_prompt = infovqa_prompt
        else:
            input_prompt = base_prompt

        dataset = VQADataset(
            train=ds_collections[ds_name]['train'],
            test=ds_collections[ds_name]['test'],
            prompt=input_prompt,
            few_shot=args.few_shot,
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
            collate_fn=partial(collate_fn),
        )

        outputs = []
        evoke_large_model_num = torch.tensor(0.0).cuda()

        total_time = 0

        for _, (pixel_values, questions, question_ids, annotations) in tqdm(enumerate(dataloader)):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            generation_config = dict(
                num_beams=args.num_beams,
                max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
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
            # generation_config["consistency_config"]["consistency_strategy"] = args.consistency_strategy

            torch.cuda.synchronize()
            start = time.time()
            pred, scores, consistency_score, visual_token_importance = small_model.chat(
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

            # if args.consistency_strategy == 'product_sum':
            #     consistency_score = torch.prod(consistency_score)
            # elif args.consistency_strategy == 'product_average':
            #     consistency_score = torch.pow(torch.prod(consistency_score), 1 / consistency_score.shape[1])



            # if args.exit_strategy == 'product_average':


            # elif args.exit_strategy == 'product_sum':
            #     scores = torch.concatenate(scores, dim=0)
            #     scores, _ = scores.softmax(dim=-1).max(dim=-1)
            #     confidence = torch.prod(scores).item()

            # elif args.exit_strategy == 'product_quantile_0.25':
            #     scores = torch.concatenate(scores, dim=0)
            #     scores, _ = scores.softmax(dim=-1).max(dim=-1)
            #     confidence = torch.quantile(scores, dim=-1, q=0.25, interpolation='lower').item()

            # elif args.exit_strategy == 'product_quantile_0.50':
            #     scores = torch.concatenate(scores, dim=0)
            #     scores, _ = scores.softmax(dim=-1).max(dim=-1)
            #     confidence = torch.quantile(scores, dim=-1, q=0.50, interpolation='lower').item()

            # elif args.exit_strategy == 'product_quantile_0.75':
            #     scores = torch.concatenate(scores, dim=0)
            #     scores, _ = scores.softmax(dim=-1).max(dim=-1)
            #     confidence = torch.quantile(scores, dim=-1, q=0.75, interpolation='lower').item()

            # elif args.exit_strategy == 'entropy':
            #     scores = torch.concatenate(scores, dim=0)
            #     scores = scores.softmax(dim=-1)
            #     max_entropy = math.log(scores.shape[1])
            #     ratio = -torch.sum(scores * torch.log(scores + 1e-12), dim=-1) / max_entropy 
            #     confidence = torch.clip((1 - ratio.mean()), min=0.0, max=1.0).item()


            # if args.fuse_strategy == 'product':
            #     confidence = confidence * consistency_score.item()
            # elif args.fuse_strategy == 'average':
            #     confidence = (confidence + consistency_score.item()) / 2
            

            torch.cuda.synchronize()
            end = time.time()
            small_model_times = [end - start]
            



            # confidences = [confidence]
            evoke_large_model_num += 1
           
            del generation_config['consistency_config'] 
            generation_config["return_dict_in_generate"] = False
            generation_config["output_scores"] = False
            generation_config["output_attentions"] = False
            generation_config["large_model_prune_layer"] = args.large_model_prune_layer
            generation_config["large_model_prune_ratio"] = args.large_model_prune_ratio
            generation_config['visual_token_importance'] = visual_token_importance

            torch.cuda.reset_peak_memory_stats(device=0)
            torch.cuda.reset_peak_memory_stats(device=1)
            torch.cuda.synchronize()
            start = time.time()
            pred = large_model.chat(
                    tokenizer=large_model_tokenizer,
                    pixel_values=pixel_values,
                    question=questions[0],
                    generation_config=generation_config,
                    large_model=True
                )
            print(f"{float(torch.cuda.max_memory_allocated(0) + torch.cuda.max_memory_allocated(1))  / float(1024**3)} GiB")
            torch.cuda.synchronize()
            end = time.time()
            large_model_times = [end - start]


            
            large_answers = [pred]

            for question, question_id, small_answer, large_answer, annotation, original_confidence,  consistency_score, small_model_time, large_model_time  in zip(questions, question_ids, small_answers, large_answers, annotations, original_confidences, consistency_scores, small_model_times, large_model_times):
                if ds_name in ['vqav2_val', 'vqav2_testdev', 'okvqa_val', 'textvqa_val',
                               'vizwiz_val', 'textvqa_val_ocr']:
                    outputs.append({
                        'question': question,
                        'question_id': question_id,
                        'answer': large_answer,
                        'large_answer': large_answer,
                        'large_model_time': large_model_time,
                        'small_answer': small_answer,
                        'small_model_time':small_model_time,
                        'original_confidence': original_confidence,
                        'consistency_score': consistency_score
                    })
                elif ds_name in ['docvqa_val', 'infographicsvqa_val', 'gqa_testdev', 'ocrvqa_val',
                                 'ocrvqa_test', 'gqa_testdev_llava', 'infographicsvqa_test',]:
                    outputs.append({
                        'question': question,
                        'questionId': question_id,
                        'answer': large_answer,
                        'annotation': annotation,
                        'large_answer': large_answer,
                        'small_answer': small_answer,
                        'original_confidence': original_confidence,
                        'consistency_score': consistency_score
                    })
                elif ds_name in ['ai2diagram_test']:
                    outputs.append({
                        'question': question,
                        'image': question_id,
                        'answer': answer,
                        'annotation': annotation,
                    })
                elif ds_name in ['chartqa_test_human', 'chartqa_test_augmented']:
                    outputs.append({
                        'question': question,
                        'answer': large_answer,
                        'annotation': annotation,
                        'large_answer': large_answer,
                        'small_answer': small_answer,
                        'original_confidence': original_confidence,
                        'consistency_score': consistency_score
                    })
                elif ds_name in ['docvqa_test']:
                    outputs.append({
                        'questionId': question_id,
                        'answer': answer,
                    })
                elif ds_name in ['vizwiz_test']:
                    outputs.append({
                        'image': question_id.replace('data/vizwiz/test/', ''),
                        'answer': answer,
                    })
                else:
                    raise NotImplementedError
        

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))
        
        
        torch.distributed.all_reduce(evoke_large_model_num, op=torch.distributed.ReduceOp.SUM)
        print(f"use large model:{evoke_large_model_num.item()}", )

        merged_outputs = [json.loads(_) for _ in merged_outputs]
        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())





            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(fold_path, results_file)
            json.dump(merged_outputs, open(results_file, 'w'))
            print('Results saved to {}'.format(results_file))

            if ds_collections[ds_name]['metric'] == 'vqa_score':
                evaluator = TextVQAAccuracyEvaluator()
                annotation = json.load(open(ds_collections[ds_name]['annotation'], 'r'))['annotations']
                question_id2answers = {}
                for item in annotation:
                    question_id = item['question_id']
                    answers = [answer['answer'] for answer in item['answers']]
                    question_id2answers[question_id] = answers
                for item in merged_outputs:
                    item['pred_answer'] = item['answer']
                    item['gt_answers'] = question_id2answers[item['question_id']]
                accuracy = evaluator.eval_pred_list(merged_outputs)
                print(ds_name, accuracy)
                summaries.append([args.small_checkpoint, ds_name, accuracy, \
                    f"use large model:{evoke_large_model_num.item()}"])

            elif ds_collections[ds_name]['metric'] == 'anls':
                json.dump(merged_outputs,
                          open(results_file, 'w'),
                          ensure_ascii=False)
                print('python eval/vqa/infographicsvqa_eval.py -g ' +
                      ds_collections[ds_name]['annotation'] + ' -s ' +
                      results_file)
                os.system('python eval/vqa/infographicsvqa_eval.py -g ' +
                          ds_collections[ds_name]['annotation'] + ' -s ' +
                          results_file)
            elif ds_collections[ds_name]['metric'] == 'relaxed_accuracy':
                relaxed_accuracy = evaluate_relaxed_accuracy(merged_outputs)
                print(ds_name, {'relaxed_accuracy': relaxed_accuracy})
                summaries.append([ds_name, {'relaxed_accuracy': relaxed_accuracy}, \
                    f"use large model:{evoke_large_model_num.item()}"])
            elif ds_collections[ds_name]['metric'] == 'accuracy':
                if 'gqa' in ds_name:
                    dst_file = './data/gqa/testdev_balanced_predictions.json'
                    print('python eval/vqa/convert_gqa_for_eval.py --src ' +
                          results_file + ' --dst ' + dst_file)
                    python_path = 'python'
                    os.system(python_path + ' eval/vqa/convert_gqa_for_eval.py --src ' +
                              results_file + ' --dst ' + dst_file)
                    command = f'cd ./data/gqa/ && {python_path} eval.py --tier testdev_balanced && cd ../../'
                    print(command)
                    accuracy = subprocess.check_output(command, shell=True, universal_newlines=True)
                else:
                    accuracy = {'accuracy': f'{evaluate_exact_match_accuracy(merged_outputs):.4f}'}
                print(ds_name, accuracy)
                summaries.append([args.small_checkpoint, ds_name, accuracy, \
                    f"use large model:{evoke_large_model_num.item()}"])

        torch.distributed.barrier()


    writer = open(os.path.join(fold_path, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")

    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()

def split_model(model_name, gpus_per_model):
    device_map = {}
    world_size = gpus_per_model
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 1

    return device_map




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--small_checkpoint', type=str, default='')
    parser.add_argument('--large_checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str,
                        default='okvqa_val,textvqa_val,vizwiz_val,ai2diagram_test,gqa_testdev_llava')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--large_model_prune_layer', type=float, default=0.3)
    parser.add_argument('--large_model_prune_ratio', type=float, default=0.3)
    parser.add_argument('--consistency_token_ratio', type=float, default=0.05)
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--gpus_per_model', type=int, default=2)

    args = parser.parse_args()

    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir, exist_ok=True)

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
    if large_model_size in ['1B', '40B', '76B']:
        large_config.llm_config._attn_implementation = 'eager'
    else:
        large_config.llm_config.attn_implementation = 'eager'
    large_config.vision_config.use_flash_attn = True
    # our method also supports inference with flashattn by setting attn_implementation to 'flash_attention_2'

    # assert args.split, "args.split must be True"
    # assert args.gpus_per_model is not None, "args.gpus_per_model must be inputed"

    gpus_per_model = args.gpus_per_model
    device_map = split_model(args.large_checkpoint.split("--")[-1], gpus_per_model)
    device_map = {k: int(os.getenv('LOCAL_RANK', 0)) + v * int(os.getenv('WORLD_SIZE', 1))  for k, v in device_map.items()}

    large_model = InternVLChatModel.from_pretrained(
        args.large_checkpoint, config=large_config, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit, device_map=device_map, **kwargs).eval()
    # if not args.load_in_8bit and not args.load_in_4bit and not args.auto:
    #     large_model = large_model.cuda()
        
        
        
    image_size = large_model.config.force_image_size or large_model.config.vision_config.image_size
    use_thumbnail = large_model.config.use_thumbnail
    
    assert large_model.config.force_image_size or large_model.config.vision_config.image_size == \
        small_model.config.force_image_size or small_model.config.vision_config.image_size
    assert small_model.config.use_thumbnail == large_model.config.use_thumbnail



    total_params = sum(p.numel() for p in small_model.parameters()) / 1e9
    args.num_beams = 1
    print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {small_model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')

    evaluate_chat_model()
