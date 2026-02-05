export LMUData=/mnt/shared-storage-user/auto-eval-pipeline/vlmeval/LMUData
export OPENAI_API_KEY=sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL
export OPENAI_API_BASE=http://35.220.164.252:3888/v1/chat/completions
model_name=${1:-"internvl-30b-a3b-cpt-tiny"}
api_nproc=${2:-64}

# CCOCR MaCBench MM-IFEval OCRBench_v2_MINI SArena_MINI ChartMimic_v2_direct
datasets=(MMMU_Pro_10c OlympiadBench IPhO_2025 BLINK ChartQAPro RefCOCO ScreenSpot_v2)

# 2-1
run_experiment_2_1() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_2_1 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1 \
        --use-enable-thinking \
        --temperature 0.0 \
        --top-k 1 \
        --reuse --verbose
}

# 2-2
run_experiment_2_2() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_2_2 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1_no_custom_prompt \
        --use-enable-thinking \
        --temperature 0.0 \
        --top-k 1 \
        --reuse --verbose
}

# 2-3
run_experiment_2_3() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_2_3 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1 \
        --use-enable-thinking \
        --temperature 1.0 \
        --top-p 0.95 \
        --top-k 20 \
        --repetition-penalty 1.0 \
        --reuse --verbose
}

# 2-4
run_experiment_2_4() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_2_4 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1_no_custom_prompt \
        --use-enable-thinking \
        --temperature 1.0 \
        --top-p 0.95 \
        --top-k 20 \
        --repetition-penalty 1.0 \
        --reuse --verbose
}

run_experiment_2_1
run_experiment_2_2
run_experiment_2_3
run_experiment_2_4