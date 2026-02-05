export LMUData=/mnt/shared-storage-user/intern7shared/internvl_a4s/data/LMUData
export OPENAI_API_KEY=sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL
export OPENAI_API_BASE=http://35.220.164.252:3888/v1/chat/completions
model_name=${1:-"internvl-30b-a3b-cpt-tiny"}
api_nproc=${2:-64}

python run.py \
    --model ${model_name} \
    --data MicroVQA MSEarthMCQ XLRS-Bench-lite MMBench_V11 MMSci_DEV_MCQ \
    --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
    --key sk-admin \
    --api-nproc ${api_nproc} \
    --work-dir s1_outputs \
    --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
    --custom-prompt internvl3 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95 \
    --thinker \
    --reuse --verbose

# python run.py \
#     --model ${model_name} \
#     --data MMMU_DEV_VAL MathVista_MINI SFE Physics MathVision \
#     --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
#     --key sk-admin \
#     --api-nproc ${api_nproc} \
#     --work-dir s1_outputs \
#     --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
#     --custom-prompt internvl3 \
#     --temperature 0.8 \
#     --top-k 50 \
#     --top-p 0.95 \
#     --thinker \
#     --reuse --verbose