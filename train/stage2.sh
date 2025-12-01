machine_rank=$1
main_process_ip=$2
export WANDB_DISABLED=True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1


accelerate launch \
    --num_processes 32 \
    --num_machines 4 \
    --machine_rank $machine_rank \
    --main_process_ip $main_process_ip \
    --main_process_port 8848 \
    --config_file recipes/accelerate_configs/zero1.yaml \
    src/open_r1/sft.py \
    --config config_stage2.yaml
