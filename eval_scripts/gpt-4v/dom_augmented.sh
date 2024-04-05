python run_v.py \
    --instruction_path agent/prompts/jsons/cot_dom_gpt_4v.json \
    --observation_type image_text \
    --test_start_idx 0 \
    --test_end_idx 25 \
    --model gpt-4-vision-preview \
    --result_dir ./outputs/gpt-4v-dom/result \
    --provider openai \
    --selected_files 24 37 43 52 65 91 105 111 138 142 149 191 219 222 250 253 258 260 272 284 301 334 344 349 379 392 399 410 420 449 490 496 502 531 534 543 545 598 605 614 617 628 632 646 650 663 710 754 762 806