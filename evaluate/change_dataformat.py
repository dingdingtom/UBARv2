import json

if __name__ == "__main__":
    ubar_res_dict, standard_res_dict = None, {} 
    
    ubar_res_path = "experiments/baseline_Ding/epoch60_inform-92.59%_success-80.38%_bleu-15.32_score-101.81/inference-test.json"
    standard_res_path = "experiments/baseline_Ding/epoch60_inform-92.59%_success-80.38%_bleu-15.32_score-101.81/inference-test-standard-gen_state.json"

    with open(ubar_res_path, 'r') as f:
        ubar_res_dict = json.load(f)
    # print(ubar_res_dict)
    # 转换为标准格式
    dial_id, cur_standard_dial_list, cur_standard_turn_dict = None, None, None
    for dial in ubar_res_dict:
        cur_standard_dial_list = []
        dial_id = dial[0]['dial_id']

        for turn in dial[1:]:
            cur_standard_turn_dict = {}
            cur_standard_turn_dict['response'] = turn['resp_gen']
            cur_standard_dial_list.append(cur_standard_turn_dict)

        standard_res_dict[dial_id] = cur_standard_dial_list
    # print(standard_res_dict)

    # 保存标准格式文件
    json.dump(standard_res_dict, open(standard_res_path, 'w'), indent=2)




    
