from cmath import inf
import json
from logging import root
import os
import pandas as pd

if __name__ == "__main__":
    #---------------
    root_path = "Ding/ubar_attraction_test/ubar_attraction"
    output_puth = "ubar_attraction_test.xls"
    #---------------

    # df_res = pd.DataFrame()
    # for dir in os.listdir(root_path):
    #     output_path = os.path.join(root_path, dir, 'gen_db/gen_state/OUTPUT.json')
    #     try:
    #         with open(output_path) as f:
    #             res_dict = json.load(f)
    #     except FileNotFoundError:
    #         print("No such file or directory: {}".format(output_path))
    #         continue
    #     model_name = dir
    #     bleu = res_dict['bleu']['mwz22']
    #     inform = res_dict['success']['inform']['total']
    #     success = res_dict['success']['success']['total']
    #     combine_score = bleu + (inform + success) / 2.
    #     df_res = df_res.append({"Model name" : model_name, "Bleu" : bleu, 'Inform' : inform, 'Success' : success, 'Combine score' : combine_score}, ignore_index=True)
    # df_res.to_excel(output_puth)

    df_res = pd.DataFrame()
    output_path = os.path.join(root_path, 'gen_db/gen_state/OUTPUT.json')
    try:
        with open(output_path) as f:
            res_dict = json.load(f)
    except FileNotFoundError:
        print("No such file or directory: {}".format(output_path))
    model_name = dir
    bleu = res_dict['bleu']['mwz22']
    inform = res_dict['success']['inform']['total']
    success = res_dict['success']['success']['total']
    combine_score = bleu + (inform + success) / 2.
    df_res = df_res.append({"Bleu" : bleu, 'Inform' : inform, 'Success' : success, 'Combine score' : combine_score}, ignore_index=True)
    df_res.to_excel(output_puth)
