python run_v.py \
    --instruction_path agent/prompts/jsons/cot_dom_gpt_4v.json \
    --observation_type image_text \
    --test_start_idx 0 \
    --test_end_idx 812 \
    --model gpt-4o \
    --result_dir ./outputs/gpt-4o-dom-aci/result \
    --provider openai \