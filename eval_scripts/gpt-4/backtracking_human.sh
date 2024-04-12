python run_bt_human.py \
    --instruction_path agent/prompts/jsons/backtracking_predictor.json \
    --observation_type accessibility_tree \
    --test_start_idx 0 \
    --test_end_idx 25 \
    --model gpt-4-turbo-preview \
    --result_dir ./outputs/gpt-4-bt-debug/result \
    --record_dir ./outputs/gpt-4-bt-debug/record \
    --provider openai \
    --selected_files 43