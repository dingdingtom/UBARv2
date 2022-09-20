# 导入库
from init import *

import ontology
import clean_dataset

#----------------------------------------
# BLEU (corpus level)
def getMetricBleu(zip_pair, n_max=4):
    
    # zip_pair: zip_object(List[List[Str]], List[List[Str]]) | use `for` to unzip
    
    tot_len_candidate = 0
    tot_len_reference_closest = 0
    dic_n2tot_gramn_candidate = {n: 0 for n in range(1, n_max + 1)}
    dic_n2tot_gramn_clip_candidate = {n: 0 for n in range(1, n_max + 1)}
    coef_penalty = 1
    metric_bleu = 0
    
    # 各对平行语料
    for i_pair, (lst_candidate, lst_reference) in enumerate(zip_pair):
        # lst_candidate: List[Str] | ex. ['a b', 'a c']
        f_str2lst = lambda x: x.split()
        lst_candidate = list(map(f_str2lst, lst_candidate))
        lst_reference = list(map(f_str2lst, lst_reference))
        # lst_candidate: List[List[Str]] | ex. [['a', 'b'], ['a', 'c']]

        # 某对语料的多个参考句子先统计，再把 counter 保存到字典，减少重复计算
        dic_id2counter_reference = {}
        for i_refernce, reference in enumerate(lst_reference):
            for n in range(1, n_max + 1):
                counter_reference = Counter(ngrams(reference, n))
                dic_id2counter_reference[(i_refernce, n)] = counter_reference
        
        # 某对语料的各个候选句子
        for candidate in lst_candidate:
            # candidate: List[Str] | ex. ['a', 'b']
            for n in range(1, n_max + 1):
                counter_candidate = Counter(ngrams(candidate, n))
                # counter_candidate: Counter | ex. Counter({('a',): 1, ('b',): 1}) | ex. Counter({('a', 'b'): 1})
                
                # 指定 n 时的 n-gram 个数
                num_gramn_candidate = sum(counter_candidate.values())
                # 在 bleu 标准下，统计某个候选句子的各个 n-gram 在所有参考句子中的“裁剪个数”
                # 某个候选句子的某种 n-gram 的裁剪个数：
                #     最大值为它在候选句子中的个数
                #     裁剪：它在某个参考句子中的个数，并取所有参考句子的最大值
                num_gramn_clip_candidate = 0
                for gramn_candidate in counter_candidate:
                    cnt_gramn_candidate = counter_candidate[gramn_candidate]
                    cnt_gramn_maxmatch = 0
                    for i_refernce, reference in enumerate(lst_reference):
                        counter_reference = dic_id2counter_reference[(i_refernce, n)]
                        cnt_gramn_maxmatch = max(cnt_gramn_maxmatch, counter_reference.get(gramn_candidate, 0))
                    num_gramn_clip_candidate += min(cnt_gramn_candidate, cnt_gramn_maxmatch)
                dic_n2tot_gramn_candidate[n] += num_gramn_candidate
                dic_n2tot_gramn_clip_candidate[n] += num_gramn_clip_candidate
                
            # blue 长度惩罚：惩罚短的候选句子
            len_candidate = len(candidate)
            lst_len_reference = [len(reference) for reference in lst_reference]
            # 用自定义 min 找到与当前候选句子的长度最接近的参考句子，若有多个最接近，则取最短的参考句子，记录该长度
            # ex. (10 | 11, 12) → (10 | 11)
            # ex. (10 | 9, 11) → (10 | 9)
            len_reference_closest = min(
                lst_len_reference, 
                key=lambda len_reference: 
                    (abs(len_reference - len_candidate), len_reference)
            )
            tot_len_candidate += len_candidate
            tot_len_reference_closest += len_reference_closest
          
    # 若总候选句子长度小于总最接近参考句子长度，则有小于 1 的长度惩罚系数
    if tot_len_candidate < tot_len_reference_closest:
        coef_penalty = e ** (1 - tot_len_reference_closest / (tot_len_candidate + epsilon))
        
    # 对每个 n 分别求 bleu 精度，加权求和
    for n in range(1, n_max + 1):
        weight = 1 / n_max
        tot_gramn_candidate = dic_n2tot_gramn_candidate[n]
        # 当至少一个句子长度大于 n 时才有意义
        if tot_gramn_candidate > 0:
            tot_gramn_clip_candidate = dic_n2tot_gramn_clip_candidate[n]
            # 当至少一个 n-gram 出现在某个参考句子时才有意义
            if tot_gramn_clip_candidate > 0:
                precision = tot_gramn_clip_candidate / tot_gramn_candidate 
                metric_bleu += weight * math.log(precision)
    # corpus level bleu
    metric_bleu = 100 * coef_penalty * math.exp(metric_bleu)    

    return metric_bleu

#----------------------------------------
# Success, Inform
def getMetricSuccessInform(lst_dic_turn, data, reader):
    
    # lst_dic_turn: List[Dict] | each dict is a turn
    
    dic_id2dial = {}
    cnt_dial = 0
    sum_metric_inform_dial = 0
    sum_metric_success_dial = 0
    flag_eval_domain_gen = True
    flag_eval_belief_gen = False
    lst_request_can = ['phone', 'address', 'postcode', 'reference', 'id']
    
    # 输入为不同对话中的各个 `dic_turn` 拼接得到的列表
    # 根据 `id_dial` 归类各个对话的所有 `dic_turn`
    for dic_turn in lst_dic_turn:
        id_dial = dic_turn['dial_id']
        dic_id2dial.setdefault(id_dial, []).append(dic_turn)
    # dic_id2dial: Dict | key is `id_dial`, value is the list of `dic_turn`
    
    # 对每个对话
    # 根据领域 `domain` 和原始目标 `dic_goal_raw`，确定目标 `dic_goal`，再得到需求 `dic_request`
    for id_dial, dial in dic_id2dial.items():
        dic_goal_raw = data[id_dial]['goal']
        dic_goal = {}
        dic_request = {}
        dic_result_db_offer = {}
        dic_result_request_offer = {}
        cnt_dial += 1
        
        # 某个对话的各个领域，得到领域下的目标
        for domain in ontology.all_domains:
            if domain in dic_goal_raw:
                dic_goal[domain] = {'informable': {}, 'requestable': [], 'booking': {}}
                
                if 'info' in dic_goal_raw[domain]:
                    # booking, requestable
                    if 'book' in dic_goal_raw[domain]:
                        dic_goal[domain]['booking'] = \
                            {slot: value for slot, value in dic_goal_raw[domain]['book'].items()}
                        dic_goal[domain]['requestable'].append('reference')
                    if 'reqt' in dic_goal_raw[domain]:
                        if domain in ['train']:
                            lst_domain_request = ['id']
                        else:
                            lst_domain_request = ['phone', 'address', 'postcode', 'reference', 'id']
                        for request_raw in dic_goal_raw[domain]['reqt']:
                            if request_raw in lst_domain_request:
                                dic_goal[domain]['requestable'].append(request_raw)
                                
                    # informable
                    for slot, value in dic_goal_raw[domain]['info'].items():
                        slot, value = clean_dataset.clean_slot_values(domain, slot, value) # need
                        value = ' '.join([token.text for token in nlp(value)]).strip()
                        dic_goal[domain]['informable'][slot] = value
        
        # 对于某个对话的各个领域，得到领域下的需求
        #     初始化查数据库后的已提供的具体值
        #     初始化在回答中的已提供的某些需求
        for domain in dic_goal.keys():
            dic_request[domain] = [i for i in dic_goal[domain]['requestable']]
            dic_result_db_offer[domain] = []
            dic_result_request_offer[domain] = []
        
        # 某个对话的各轮，都参与更新：查数据库后的已提供的具体值、回答中的已提供的某些需求
        for i_turn in range(len(dial)):
            dic_turn = dial[i_turn]
            # dic_turn: Dict 
            #     key: ['dial_id', 'turn_num', 'user', 'dspn', 'dspn_gen', 'bsdx',
            #           'bspn', 'bspn_gen', 'aspn', 'aspn_gen', 'resp', 'resp_gen', 'pointer']
            
            # 对于目标中的各个领域，只有是该对话的该轮的领域时，才继续评估
            key_domain = 'dspn_gen' if flag_eval_domain_gen else 'dspn'
            str_domain_turn = dic_turn[key_domain]
            lst_domain_turn = [i.strip(' [],') for i in str_domain_turn.split()]
            for domain in dic_goal.keys():
                if domain not in lst_domain_turn:
                    continue
                
                # 只有当回答中出现待填充的 `[value_name]` 或 `[value_id]` 时，才更新领域的已提供的具体值
                str_resp_gen = dic_turn['resp_gen']     
                if '[value_name]' in str_resp_gen or '[value_id]' in str_resp_gen:
                    # 只有目标领域是特定领域时，才用具体值更新已提供，否则用 `[value_name]` 代替
                    if domain not in ['restaurant', 'hotel', 'attraction', 'train']:
                        dic_result_db_offer[domain] = ['[value_name]']
                    else:
                        # 建立 belief 的字典
                        dic_belief = {}
                        key_belief = 'bspn_gen' if flag_eval_belief_gen else 'bspn'
                        str_belief = dic_turn[key_belief]
                        # str_belief: Str | ex. [hotel] area south stars 4 [restaurant] time 14:30

                        # 用正则表达式匹配到 `str_belief` 中的各个领域的字符串
                        pattern_belief = re.compile(r'\[(.*?)\]([^[]*)')
                        # pattern_belief: Pattern | find all like `[str_a]str_b`
                        #    ex. [hotel] area south stars 4 
                        #    ex. [restaurant] time 14:30
                        lst_tup_belief = re.findall(pattern_belief, str_belief)

                        # 每个领域的字符串得到一个元组 `tup_belief`
                        for tup_belief in lst_tup_belief:
                            domain_belief, str_pairs = tup_belief
                            # domain_belief: Str | ex. hotel
                            # str_pairs: Str | ex.  area south stars 4 
                            dic_belief[domain_belief] = {}
                            lst_word = str_pairs.strip().split()
                            # lst_word: List[Str] | ex. ['area', 'south', 'stars', '4']
                            slot, lst_value = None, []

                            # 逐个词语，找到 slot 词，紧跟在它后面的是对应的 value
                            for word in lst_word:
                                if word in ontology.all_slots:
                                    if slot is not None:
                                        dic_belief[domain_belief][slot] = ' '.join(lst_value)
                                        # slot: Str | ex. area
                                        # lst_value: List[Str] | ex. ['south']
                                    slot, lst_value = word, []
                                else:
                                    lst_value.append(word)
                            if slot is not None:
                                dic_belief[domain_belief][slot] = ' '.join(lst_value)
                        
                        # 用 `dic_belief[domain]` 作为约束查询数据库，得到领域的具体值
                        lst_result_db = []
                        if domain in dic_belief.keys():
                            lst_result_db = reader.db.queryJsons(domain, dic_belief[domain])
                            
                        # 若有新的具体值，则更新领域的已提供的具体值的列表
                        lst_result_db_old = dic_result_db_offer[domain]
                        set_difference_result_db = set(lst_result_db) - set(lst_result_db_old)
                        if len(set_difference_result_db):
                            dic_result_db_offer[domain] = [i for i in lst_result_db]

                # 对于指定要评估的需求，只有需求在回答中存在时，才更新领域的已提供的需求
                for request_can in lst_request_can:
                    str_value_request_can = f'[value_{request_can}]'
                    if str_value_request_can in str_resp_gen:
                        str_pointer = dic_turn['pointer']
                        if request_can in ['reference'] \
                        and not ('booked' in str_pointer or 'ok' in str_pointer):
                            continue
                        dic_result_request_offer[domain].append(request_can)
                        
        # 上面已经根据该对话的各轮及该轮的领域，更新了已提供的数据库具体值和需求
        # 还要在不看对话的情况下，考虑部分特殊情况，用 `[value_name]` 作为已提供
        for domain in dic_goal.keys():
            if 'name' in dic_goal[domain]['informable'].keys() \
            or domain not in ['restaurant', 'hotel', 'attraction', 'train'] \
            or (domain in ['train'] and \
                not (len(dic_result_db_offer[domain]) \
                    or 'id' in dic_goal[domain]['requestable'])):
                dic_result_db_offer[domain] = ['[value_name]']
            
        # 该对话是否满足 inform
        # 用领域目标作为约束查询数据库，检测 `[value_name]` 是否已提供，或者具体值是否与目标答案有交集
        metric_inform_dial = 0
        sum_metric_inform_domain = 0
        for domain in dic_goal.keys():
            if '[value_name]' in dic_result_db_offer[domain]:
                sum_metric_inform_domain += 1
            else:
                lst_result_db_goal = reader.db.queryJsons(domain, dic_goal[domain]['informable'])
                lst_result_db_offer = [i for i in dic_result_db_offer[domain]]
                set_intersection_result_db = set(lst_result_db_offer) & set(lst_result_db_goal)
                if len(set_intersection_result_db):
                    sum_metric_inform_domain += 1
        # 若所有领域都提供了：至少一个按目标约束查询得到的答案的具体值
        #     即为各个目标领域都提供了至少一个实体，则该对话的 `metric_inform` 为 1
        if sum_metric_inform_domain == len(dic_goal):
            metric_inform_dial = 1
        sum_metric_inform_dial += metric_inform_dial
            
        # 该对话是否满足 success
        # 有不等式 metric_success <= metric_inform，即：有 success 必有 inform
        metric_success_dial = 0
        if metric_inform_dial:
            sum_metric_success_domain = 0
            for domain in dic_goal.keys():
                lst_result_request_goal = [i for i in dic_request[domain]]
                lst_result_request_offer = [i for i in dic_result_request_offer[domain]]
                if set(lst_result_request_goal).issubset(set(lst_result_request_offer)):
                    sum_metric_success_domain += 1
            # 若为目标的各个领域都提供了：目标的所有需求，则该对话的 `metric_success` 为 1
            if sum_metric_success_domain == len(dic_goal):
                metric_success_dial = 1
        sum_metric_success_dial += metric_success_dial
    
    return sum_metric_inform_dial, sum_metric_success_dial, cnt_dial

#----------------------------------------
# 展示对话
def getTurnStrResult(lst_dic_turn, str_show_choice='r'):
    
    # lst_dic_turn: List[Dict] | each dict is a turn
    # str_show_choice: str | ex. bar
    
    dic_id2dial = {}
    cnt_dial = 0
    str_result_lst_dic_turn = ''
    dic_choice2name = {'b': 'bspn', 'a': 'aspn', 'r': 'resp'}
    
    # 输入为不同对话中的各个 `dic_turn` 拼接得到的列表
    # 根据 `id_dial` 归类各个对话的所有 `dic_turn`
    for dic_turn in lst_dic_turn:
        id_dial = dic_turn['dial_id']
        dic_id2dial.setdefault(id_dial, []).append(dic_turn)
    # dic_id2dial: Dict | key is `id_dial`, value is the list of `dic_turn` 
    
    for id_dial in dic_id2dial.keys():
        cnt_dial += 1
        dial = dic_id2dial[id_dial]
        str_result_lst_dic_turn += f'Dial No.{cnt_dial}  | ID: {id_dial}\n'
        for i_turn, dic_turn in enumerate(dial):
            str_user = dic_turn['user']
            str_result_lst_dic_turn += f'Turn {i_turn} | user: {str_user}\n'
            for choice in dic_choice2name.keys():
                if choice in str_show_choice:
                    name_element = dic_choice2name[choice]
                    str_element  = dic_turn[name_element]
                    str_element_gen = dic_turn[name_element + '_gen']
                    str_result_lst_dic_turn += f'Turn {i_turn} | {name_element}: {str_element}\n'
                    str_result_lst_dic_turn += f'Turn {i_turn} | {name_element}_gen: {str_element_gen}\n'
        str_result_lst_dic_turn += '\n'
        
    return str_result_lst_dic_turn

def getTurnJsonResult(lst_dic_turn):
    
    dic_id2dial = {}
    dic_id2dial_json = {}
    cnt_dial = 0
    str_result_lst_dic_turn = ''
    dic_choice2name = {'b': 'bspn', 'a': 'aspn', 'r': 'resp'}
    
    # 输入为不同对话中的各个 `dic_turn` 拼接得到的列表
    # 根据 `id_dial` 归类各个对话的所有 `dic_turn`
    for dic_turn in lst_dic_turn:
        id_dial = dic_turn['dial_id']
        dic_id2dial.setdefault(id_dial, []).append(dic_turn)
    # dic_id2dial: Dict | key is `id_dial`, value is the list of `dic_turn` 
    
    for id_dial in dic_id2dial.keys():
        cnt_dial += 1
        dial = dic_id2dial[id_dial]
        dic_id2dial_json[id_dial] = []
        for i_turn, dic_turn in enumerate(dial):
            dic_turn_json = {}
            dic_turn_json['response'] = dic_turn['resp_gen']
            lst_bspn = dic_turn['bspn_gen'].split()
            dic_bspn = {}
            key_domain = ''
            key_slot = ''
            lst_slot_part = [] 
            str_slot_part = ''
            for i_word, word in enumerate(lst_bspn):
                if word.startswith('[') and word.endswith(']'):
                    if word[1:-1] in ontology.all_domains:
                        # 下一领域前，要把缓存 slot value 清空
                        if len(key_domain) and len(key_slot) and len(lst_slot_part):
                            str_slot_part = ' '.join(lst_slot_part)
                            dic_bspn[key_domain][key_slot] = str_slot_part
                            lst_slot_part = []
                        # 新领域
                        key_domain = word[1:-1]
                        dic_bspn.setdefault(key_domain, {})
                    else:
                        # 无效领域
                        key_domain = ''
                elif len(key_domain):
                    if word in ontology.get_slot:
                        # 该领域的下一 slot key 前，要把缓存 slot value 清空
                        if len(key_slot) and len(lst_slot_part):
                            str_slot_part = ' '.join(lst_slot_part)
                            dic_bspn[key_domain][key_slot] = str_slot_part
                            lst_slot_part = []
                        # 新 slot key
                        key_slot = word
                    elif len(key_slot):
                        # 某个 slot key 下，slot value 可能需要多个词拼接
                        lst_slot_part.append(word)
            # 最后要把缓存 slot value 清空
            if len(key_domain) and len(key_slot) and len(lst_slot_part):
                str_slot_part = ' '.join(lst_slot_part)
                dic_bspn[key_domain][key_slot] = str_slot_part
                lst_slot_part = []          
                    
            dic_turn_json['state'] = dic_bspn
            
            dic_id2dial_json[id_dial].append(dic_turn_json)
            
    return dic_id2dial_json

#----------------------------------------
# 测试
if __name__ == '__main__':
#     # lst2_candidate = [['a b', 'a c']]
#     # lst2_reference = [['a b d', 'e c d', 'a a c c']]
#     lst2_candidate = [
#         ['It is a guide to action which ensures that the military always obeys the commands of the party']
#     ]
#     lst2_reference = [
#         ['It is a guide to action that ensures that the military will forever heed Party commands',
#          'It is the guiding principle which guarantees the military forces always being under the command of the Party',
#          'It is the practical guide for the army always to heed the directions of the party']
#     ]
#     n_max = 4
#     zip_pair = zip(lst2_candidate, lst2_reference)
#     print(zip_pair)
#     metric_bleu = getMetricBleu(zip_pair, n_max)
#     print(metric_bleu)
    
    1
    q = torch.tensor([0.4, 0.6])
    p1 = torch.tensor([0.3, 0.7])
    p2 = torch.tensor([0.2, 0.8])
    f1 = F.kl_div(q.log(), p1, reduction='sum')
    f2 = F.kl_div(q.log(), p2, reduction='sum')
    #f2 = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p2, dim=-1), reduction='sum')
    print(f'1 {(p1 * torch.log(p1 / q)).sum()}')
    print(f'2 {(p2 * torch.log(p2 / q)).sum()}')
    print(f'1F {f1}')
    print(f'2F {f2}')

    
    