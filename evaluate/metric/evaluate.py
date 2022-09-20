# !/usr/bin/env python3
# conding=utf-8

import sys
import json
import os
from textwrap import indent 

from mwzeval.metrics import Evaluator


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bleu", dest='bleu', action="store_true", default=False, help="If set, BLEU is evaluated.")
    parser.add_argument("-s", "--success", dest='success', action="store_true", default=False, help="If set, inform and success rates are evaluated.")
    parser.add_argument("-r", "--richness", dest='richness', action="store_true", default=False, help="If set, various lexical richness metrics are evaluated.")
    parser.add_argument("--input_path", type=str, required=True, help="Input JSON file path.")
    parser.add_argument("--input_file_name", type=str, required=True, help="Input JSON file name")
    parser.add_argument("--output_father_folder_path", type=str, required=True, help="Output path")
    parser.add_argument("--output_file_name", type=str, default="evaluation_results.json", help="Output file name, here will be the final report.")
    args = parser.parse_args()

    if not args.bleu and not args.success and not args.richness:
        sys.stderr.write('error: Missing argument, at least one of -b, -s, and -r must be used!\n')
        parser.print_help()
        sys.exit(1)

    # 读取模型输出结果的json文件，作为该测试脚本的输入
    with open(os.path.join(args.input_path, args.input_file_name), 'r') as f:
        input_data = json.load(f)

    
    # output_path = output_father_folder + output_file_path
    output_path = os.path.join(os.path.join(args.output_father_folder_path, args.input_path.split('/')[-1]), 'gen_db/gen_state') # inference时，使用生成的db和生成的bspan
    # 同时，将模型输出结果的json文件保存到该文件夹
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, "INPUT.json"), 'w') as f:
        json.dump(input_data, f, indent=2)

    e = Evaluator(args.bleu, args.success, args.richness, output_path)
    results = e.evaluate(input_data)

    for metric, values in results.items():
        if values is not None:
            print(f"====== {metric.upper()} ======")
            for k, v in values.items():
                print(f"{k.ljust(15)}{v}")
            print("")

    with open(os.path.join(output_path, args.output_file_name), 'w+') as f:
        json.dump(results, f)
