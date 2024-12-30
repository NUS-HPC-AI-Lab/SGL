set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
GPUS=${GPUS:-8}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/vqa/evaluate_vqa.py \
    --small_checkpoint "/root/model_ckpts/models--OpenGVLab--InternVL2-2B" \
    --large_checkpoint "/root/model_ckpts/models--OpenGVLab--InternVL2-26B" \
    --datasets "textvqa_val" --dynamic \
    --out-dir "results-2B-26B_textvqa_val" \
    --large_model_prune_layer 0.4 \
    --large_model_prune_ratio 0.4

