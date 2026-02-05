closellm_models = {
    "GPT-4o": {
        "api_key": "sk-Z60Fu6jLvEhLsVwCQuBYzovu2Mgj7LVgw6c2NeR9QkWubEiT",
        "base_url": "http://35.220.164.252:3888/v1",
        "model_name": "gpt-4o",
        "max_out_len": 16384,
        "num_workers": 16
    },
    "GPT-5": {
        "api_key": "sk-Z60Fu6jLvEhLsVwCQuBYzovu2Mgj7LVgw6c2NeR9QkWubEiT",
        "base_url": "http://35.220.164.252:3888/v1",
        "model_name": "gpt-5",
        "max_out_len": 16384,
        "num_workers": 16
    },
    "GLM-4.5V": {
        "api_key": "sk-Z60Fu6jLvEhLsVwCQuBYzovu2Mgj7LVgw6c2NeR9QkWubEiT",
        "base_url": "http://35.220.164.252:3888/v1",
        "model_name": "z-ai/glm-4.5v",
        "max_out_len": 16384,
        "num_workers": 16
    },
}

openllm_models = [
    {
        "model_abbr": "Intern-S1-Pro",
        "api_key": "sk-admin",
        "base_url": "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1",
        "model_name": "interns1pro-candidate-1",
        "max_out_len": 32768,
        "num_workers": 4,
        "temperature": 1e-6,
        "top_p": 0.95,
        "top_k": 1,
        "think": True
    },
    {
        "model_abbr": "Qwen3-VL-30B-A3B-Thinking",
        "api_key": "sk-admin",
        "base_url": "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1",
        "model_name": "Qwen3-VL-30B-A3B-Thinking",
        "max_out_len": 32768,
        "num_workers": 96,
        "temperature": 1e-6,
        "top_p": 0.95,
        "top_k": 1,
    },
]

train_models = {
    "Intern-S1-cpt": {
        "api_key": "EMPTY",
        "base_url": "http://10.102.241.53:5001/v1",
        "model_name": "interns1-fp8-cpt",
        "max_out_len": 16384,
        "num_workers": 16
    },
    "InternS1_8B_base-spectrum-8k_32bsz-3epoch": {
        "api_key": "EMPTY",
        "base_url": "http://10.102.215.25:5001/v1",
        "model_name": "spectrum-8k-32bsz-3epoch",
        "max_out_len": 16384,
        "num_workers": 16
    },
}

materials_models = [
    {
        "model_abbr": "Qwen3_VL_30B_A3B_Thinking-macbench_test-16k_32bsz-3epoch",
        "api_key": "sk-admin",
        "base_url": "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1",
        "model_name": "Qwen3_VL_30B_A3B_Thinking-macbench_test-16k_32bsz-3epoch",
        "max_out_len": 32768,
        "num_workers": 96,
        "temperature": 1e-6,
        "top_p": 0.95,
        "top_k": 1,
    },
]

sfe_models = [
    {
        "model_abbr": "Qwen3_VL_30B_A3B_Thinking-sfe_s1_pro_cot-16k_32bsz-3epoch",
        "api_key": "sk-admin",
        "base_url": "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1",
        "model_name": "Qwen3_VL_30B_A3B_Thinking-sfe_s1_pro_cot-16k_32bsz-3epoch",
        "max_out_len": 32768,
        "num_workers": 96,
        "temperature": 1e-6,
        "top_p": 0.95,
        "top_k": 1,
    },
]

interns1_pro_models = [
    {
        "model_abbr": "base02_20260109a",
        "api_key": "sk-admin",
        "base_url": "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1",
        "model_name": "interns1_1_base02_20260109a_lr3e5_512gpu_fp8-sunyu",
        "max_out_len": 32768,
        "num_workers": 4,
        "temperature": 1e-6,
        "top_p": 1,
        "top_k": 1,
        "think": True
    },
    {
        "model_abbr": "base02_20260110a",
        "api_key": "sk-admin",
        "base_url": "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1",
        "model_name": "interns1_1_base02_20260110a_lr3e5_512gpu_fp8-sunyu",
        "max_out_len": 32768,
        "num_workers": 64,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "think": True
    },
    {
        "model_abbr": "base02_20260113a",
        "api_key": "sk-admin",
        "base_url": "http://s-20260104203038-22bhb-decode.ailab-evalservice.svc:4000/v1",
        "model_name": "interns1_1_base02_20260113a_lr3e5_512gpu_fp8-sunyu_demo",
        "max_out_len": 32768,
        "num_workers": 64,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 40,
        "think": True
    },
]
