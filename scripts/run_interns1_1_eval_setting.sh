export LMUData=/mnt/shared-storage-user/intern7shared/internvl_a4s/data/LMUData
export OPENAI_API_KEY=sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL
export OPENAI_API_BASE=http://35.220.164.252:3888/v1/chat/completions
model_name=${1:-"internvl-30b-a3b-cpt-tiny"}
api_nproc=${2:-64}

python run.py \
    --model ${model_name} \
    --data MMMU_DEV_VAL MathVista_MINI MMStar RealWorldQA AI2D_TEST SEEDBench2_Plus ChartQA_TEST InfoVQA_VAL DocVQA_VAL \
    --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
    --key sk-admin \
    --api-nproc ${api_nproc} \
    --work-dir outputs \
    --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
    --custom-prompt interns1_1 \
    --temperature 0.0 \
    --top-k 1 \
    --reuse --verbose