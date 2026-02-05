export LMUData=/mnt/shared-storage-user/auto-eval-pipeline/vlmeval/LMUData
export OPENAI_API_KEY=sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL
export OPENAI_API_BASE=http://35.220.164.252:3888/v1/chat/completions
model_name=${1:-"internvl-30b-a3b-cpt-tiny"}
api_nproc=${2:-64}

# datasets=(MMMU_Pro_10c OlympiadBench IPhO_2025 BLINK CCOCR ChartMimic_v2_direct ChartQAPro MaCBench MM-IFEval OCRBench_v2_MINI RefCOCO SArena_MINI ScreenSpot_v2)
datasets=(MMMU_Pro_10c OlympiadBench IPhO_2025 BLINK ChartQAPro RefCOCO ScreenSpot_v2)

# 3-1
run_experiment_3_1() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_3_1 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1 \
        --use-enable-thinking \
        --enable-thinking \
        --thinker \
        --temperature 0.0 \
        --top-k 1 \
        --reuse --verbose
}

# 3-2
run_experiment_3_2() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_3_2 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1_no_custom_prompt \
        --use-enable-thinking \
        --enable-thinking \
        --thinker \
        --temperature 0.0 \
        --top-k 1 \
        --reuse --verbose
}

# 3-3
run_experiment_3_3() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_3_3 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1 \
        --use-enable-thinking \
        --enable-thinking \
        --thinker \
        --temperature 1.0 \
        --top-p 0.95 \
        --top-k 20 \
        --repetition-penalty 1.0 \
        --reuse --verbose
}

# 3-4
run_experiment_3_4() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_3_4 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1_no_custom_prompt \
        --use-enable-thinking \
        --enable-thinking \
        --thinker \
        --temperature 1.0 \
        --top-p 0.95 \
        --top-k 20 \
        --repetition-penalty 1.0 \
        --reuse --verbose
}

# 3-5
run_experiment_3_5() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_3_5 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1 \
        --use-enable-thinking \
        --enable-thinking \
        --thinker \
        --temperature 0.8 \
        --top-p 0.95 \
        --top-k 50 \
        --reuse --verbose
}

# 3-6
run_experiment_3_6() {
    python run.py \
        --model ${model_name} \
        --data ${datasets[@]} \
        --base-url http://nginx.xujun.ailab-intern7.svc.pjlab.local:4000/v1 \
        --key sk-admin \
        --api-nproc ${api_nproc} \
        --work-dir outputs/experiment_3_6 \
        --judge-key sk-TyI2Zk5wsjSaiHIiOmdwctOy91A400jxw31gnhfQTFWGckpL \
        --custom-prompt interns1_1_no_custom_prompt \
        --use-enable-thinking \
        --enable-thinking \
        --thinker \
        --temperature 0.8 \
        --top-p 0.95 \
        --top-k 50 \
        --reuse --verbose
}

run_experiment_3_6
# run_experiment_3_5
run_experiment_3_4
# run_experiment_3_3
run_experiment_3_2
# run_experiment_3_1






