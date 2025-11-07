import os
import time
import yaml
import click



base_cmd = '''
export VLLM_ROCM_USE_AITER=1
export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export TORCHINDUCTOR_CACHE_DIR=/tmp/vllm_cache_v1_{port}
export VLLM_TORCH_COMPILE_CACHE_DIR=/tmp/vllm_cache_v2_{port}

export CUDA_VISIBLE_DEVICES={device}
python -m vllm.entrypoints.openai.api_server \
    --compilation-config '{{"full_cuda_graph": true}}' \
    --gpu-memory-utilization {gpu} \
    --model {model_path} \
    --max_model_len 32768 \
    --port {port} \
    --seed {port} \
    -tp {tp} \
    --no-enable-prefix-caching \
    &
'''

class vllm_server():
    def __init__(self, config_path='config/oss.yaml'):
        self.config = yaml.safe_load(open(config_path))
        self.run()

    
    def run(self):
        gpu_index_part = [self.config['gpu_index'][i:i+self.config['tp']] for i in range(0, len(self.config['gpu_index']), self.config['tp'])]
        for c, tmp_gpu_index in enumerate(gpu_index_part):
            tmp_config = {
                    'port': 8040 + tmp_gpu_index[0],
                    'model_path': self.config['model_path'],
                    'device': ','.join([str(i) for i in tmp_gpu_index]),
                    'tp': self.config['tp'],
                    'gpu': self.config.get('gpu', 0.95),
                    }
            tmp_cmd = base_cmd.format(**tmp_config)
            print(tmp_cmd)
            os.system(tmp_cmd)
            if c == 0:
                time.sleep(30)

@click.command()
@click.option('-c', '--config-path', default='config/oss.yaml', help='Path to the configuration YAML file.')
def main(config_path):
    vllm_server(config_path)

if __name__ == '__main__':
    main()
