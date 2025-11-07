export CUDA_VISIBLE_DEVICES=0

export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python3 uni_eval.py \
    --base_model /home/Chushi.Chen@amd.com/dev/checkpoints/ERM-ccs-1105 \
    --chat_template_name default \
    --system_prompt_name disabled \
    --output_dir results \
    --tensor_parallel_size 1 \
    --data_id zwhe99/aime90 \
    --bf16 True \
    --split 2024 \
    --max_model_len 32768 \
    --temperature 0.0 \
    --top_p 1.0 \
    --n 16
