# 读取baseline模型和当前最优模型的 MATCH_SUCCESS.json 文件
import json

if __name__ == "__main__":
    #---------------------------------
    # MATCH_SUCCESS.json 文件路径
    baseline_path = "Ding/main/1-epoch75_inform-92.79%_success-80.18%_bleu-15.30_score-101.79/gen_db/gen_state/MATCH_SUCCESS.json"
    optimal_path = "Ding/epoch1_inform-94.69%_success-84.88%_bleu-15.59_score-105.38/gen_db/gen_state/MATCH_SUCCESS.json"

    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    with open(optimal_path, 'r') as f:
        optimal = json.load(f)
    #---------------------------------
    
    baseline_better = {}
    optimal_better = {}
    for dial_id, baseline_match_success in baseline.items():
        optimal_match_success = optimal[dial_id]
        # optimal结果更好
        # 1. optimal['match'] > baseline['match']
        # 2. optimal['success'] > baseline['success']
        if (optimal_match_success['match'] > baseline_match_success['match']) or \
            optimal_match_success['success'] > baseline_match_success['success'] :
            optimal_better[dial_id] = {'optimal_match_success' : optimal_match_success, 'baseline_match_success' : baseline_match_success}
        # if (baseline_match_success['match'] > optimal_match_success['match']) or \
        #     baseline_match_success['success'] > optimal_match_success['success'] :
        #     baseline_better[dial_id] = {'optimal_match_success' : optimal_match_success, 'baseline_match_success' : baseline_match_success}

        #----------
        # optimal同时提升了match和success
        # if (optimal_match_success['match'] > baseline_match_success['match']) and \
        #     optimal_match_success['success'] > baseline_match_success['success'] :
        #     optimal_better[dial_id] = {'optimal_match_success' : optimal_match_success, 'baseline_match_success' : baseline_match_success}

        #----------
        # optimal match=1、success=1, baseline match=0 
        # if (optimal_match_success['match'] == 1 and optimal_match_success['success'] == 1 and baseline_match_success['match']== 0):
        #     optimal_better[dial_id] = {'optimal_match_success' : optimal_match_success, 'baseline_match_success' : baseline_match_success}
    
    # 记录对话数量
    baseline_better['Length'] = {'Length' : len(baseline_better)}
    optimal_better['Length'] = {'Length' : len(optimal_better)}
    # 保存json文件
    # json.dump(baseline_better, open("baseline_better.json", "w"), indent=2)
    json.dump(optimal_better, open("optimal_better.json", "w"), indent=2)



