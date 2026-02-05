import argparse
import os
import subprocess

from model_info import *

support_datasets = ['MaCBench']

datasets = support_datasets
models = openllm_models[:1]
work_dir = '/mnt/shared-storage-user/ailab-llmai4s/sunyu2/eval_output/vlmevalkit/materials'

rjob_cfg = dict(
    env=[
        'LMUData=/mnt/shared-storage-user/auto-eval-pipeline/vlmeval/LMUData',
        'TORCH_HOME=/mnt/shared-storage-user/auto-eval-pipeline/opencompass/llmeval/torch_cache',
        'HF_DATASETS_CACHE=/mnt/shared-storage-user/ailab-llmai4s/sunyu2/cache/opencompass/llmeval/hf_cache',
        'HF_HUB_CACHE=/mnt/shared-storage-user/ailab-llmai4s/sunyu2/cache/opencompass/llmeval/hf_cache',
        'HUGGINGFACE_HUB_CACHE=/mnt/shared-storage-user/ailab-llmai4s/sunyu2/cache/opencompass/hf_hub',
        'HF_DATASETS_OFFLINE=1',
        'HF_EVALUATE_OFFLINE=1',
        'HF_HUB_OFFLINE=1',
    ],
    bashrc_path="/mnt/shared-storage-user/ailab-llmai4s/sunyu2/.bashrc",
    conda_env_name="vlmeval_env",
    priority=5,
    charged_group='llmai4s_gpu',
    namespace='ailab-llmai4s',
    # charged_group='puyullm_gpu',
    # namespace='ailab-puyullmgpu',
    mount=[
        "gpfs://gpfs1/ailab-llmai4s:/mnt/shared-storage-user/ailab-llmai4s",
        "gpfs://gpfs1/songdemin:/mnt/shared-storage-user/songdemin",
        "gpfs://gpfs1/video-shared:/mnt/shared-storage-user/video-shared",
        "gpfs://gpfs1/auto-eval-pipeline:/mnt/shared-storage-user/auto-eval-pipeline",
        "gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights",
        "gpfs://gpfs1/puyudelivery:/mnt/shared-storage-user/puyudelivery",
    ],
    image='registry.h.pjlab.org.cn/ailab/pytorch:22.04-pjlab-py3.10-torch2.2.0-cu12.1',
    host_network=True,
    # idle=True,
)


def rjob_command(task_name, shell_cmd):
    # conda环境激活
    if rjob_cfg['bashrc_path'] and rjob_cfg['conda_env_name']:
        env_cmd = f'source {rjob_cfg["bashrc_path"]}; conda activate {rjob_cfg["conda_env_name"]}; '
        shell_cmd = env_cmd + shell_cmd
    code_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = code_dir.split('scripts/')[0]
    shell_cmd = f'cd {code_dir}; ' + shell_cmd

    # rjob命令模板
    tmpl = (
        "rjob submit -e DISTRIBUTED_JOB=true"
        " --host-network=true --gang-start=true"
        " --custom-resources rdma/mlnx_shared=8"
        " --custom-resources mellanox.com/mlnx_rdma=1 "
        " --private-machine='group'"
        f' --image {rjob_cfg["image"]}'
        f" --name {task_name[:512]}"
        f" -P 1"
        f" --cpu 32"
        f" --gpu 0"
        f" --memory 200000"
        f" --charged-group {rjob_cfg['charged_group']}"
        f" --namespace={rjob_cfg['namespace']}"
        f" --priority={rjob_cfg['priority']}"
    )
    for mount in rjob_cfg['mount']:
        tmpl += f" --mount={mount}"
    for env_var in rjob_cfg['env']:
        tmpl += f" --env={env_var}"
    if rjob_cfg.get('idle', False):
        tmpl += " --task-type=idle --restart-policy='restartjobonfailure' --backoff_limit=10"
    tmpl += f' -- bash -c "{shell_cmd}"'

    print(f"执行命令: {tmpl}")

    try:
        # 方法1: 直接执行字符串命令
        result = subprocess.run(tmpl, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print("命令执行成功!")
            print("输出:", result.stdout)
        else:
            print("命令执行失败!")
            print("错误:", result)

        return result.returncode == 0

    except Exception as e:
        print(f"执行命令时出错: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 缩写是-r
    parser.add_argument('--reuse', '-r', action='store_true', help='Reuse previous results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    for m in models:
        dataset_str = ' '.join(datasets)
        cmd = (f'python run.py '
               f'--data {dataset_str} '
               f'--model {m["model_name"]} '
               f'--base-url {m["base_url"]} '
               f'--key {m["api_key"]} '
               f'--api-nproc {m["num_workers"]} '
               f'--temperature {m.get("temperature", 1e-6)} '
               f'--top-p {m.get("top_p", 1)} '
               f'--top-k {m.get("top_k", 1)} '
               f'--work-dir {os.path.join(work_dir, m["model_abbr"])} '
               f'--judge-key sk-HUA9TyNTddM1MHLvkOlhEsWIJojsXJcX8k33mg47ZInwDbYH --judge-base-url http://10.140.52.166:8080/aoe/v1 --judge-api-nproc 64')
        if args.reuse:
            cmd += ' --reuse'
        if args.verbose:
            cmd += ' --verbose'
        if m.get('think', False):
            cmd += ' --use-enable-thinking --enable-thinking --thinker --custom-prompt interns1_1_think'

        rjob_command(
            task_name=f"vlmeval-{m['model_abbr']}-[{','.join(datasets)}]",
            shell_cmd=cmd
        )
