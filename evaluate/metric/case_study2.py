import json
from pyexpat import model
from textwrap import indent

from torch import gt

if __name__ == "__main__":
    #--------------
    # 需要修改的参数
    optimal_better_path = "101.79-gen_vs_105.38/optimal_better.json"
    with open(optimal_better_path, "r") as f:
        optimal_better = json.load(f)
    optimal_better_dial_ids = list(optimal_better.keys())[:-1]

    # 需要比较的两个模型，model1和model2
    # 注意两个模型的结果文件是原始格式的，而不是新指标所需的标准格式
    model1_file_path = "../20211213_myUBAR/main/1-epoch75_inform-92.79%_success-80.18%_bleu-15.30_score-101.79/inference-test.json" # model1 结果json文件的路径
    model1_name = "baseline"
    model2_file_path = "baseline _vs_105.38/epoch1_inform-94.69%_success-84.88%_bleu-15.59_score-105.38.json" # model2 结果json文件的路径
    model2_name = "105.38"
    gt_data_path = "baseline _vs_105.38/data_for_damd.json" # gt_data json文件路径
    

    #----------------
    with open(model1_file_path, "r") as f:
        model1_file = json.load(f)
    model1_dial_dict = {}
    for model1_dial_list in model1_file:
        dial_id = model1_dial_list[0]['dial_id']
        model1_dial_dict[dial_id] = model1_dial_list[1:]

    with open(model2_file_path, "r") as f:
        model2_file = json.load(f)
    model2_dial_dict = {}
    for model2_dial_list in model2_file:
        dial_id = model2_dial_list[0]['dial_id']
        model2_dial_dict[dial_id] = model2_dial_list[1:]

    with open(gt_data_path, "r") as f:
        gt_data = json.load(f)
    
    optimal_better_cases = {}
    for dial_id in optimal_better_dial_ids:
        cur_dial = []

        cur_dial.append({'goal' : gt_data[dial_id]['goal']})
        for model1_turn, model2_turn in zip(model1_dial_dict[dial_id], model2_dial_dict[dial_id]):
            # user utterance
            user_utterance = model1_turn['user']

            # system reponse
            gt_response = model1_turn['resp']
            model1_response = model1_turn['resp_gen']
            model2_response = model2_turn['resp_gen']

            # system belief state
            gt_state = model1_turn['bspn']
            model1_state = model1_turn['bspn_gen']
            model2_state = model2_turn['bspn_gen']

            cur_dial.append({
                'user' : user_utterance,
                'response' : {
                    'gt_response' : gt_response,
                    '{}_response'.format(model1_name) : model1_response,
                    '{}_repsonse'.format(model2_name) : model2_response,
                },
                'state' : {
                    'gt_state' : gt_state,
                    "{}_state".format(model1_name) : model1_state,
                    '{}_state'.format(model2_name) : model2_state
                }
            })

        optimal_better_cases[dial_id] = cur_dial

    json.dump(optimal_better_cases, open("optimal_better_case.json", "w"), indent=2)
            
                

