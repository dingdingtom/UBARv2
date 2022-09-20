
#----------------------------------------
# 配置
class _Config:
    def __init__(self):
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):
        
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }
        self.dir_data_processed = './data/multi-woz-processed/'
        self.dic_element_name2info = {
            'user': {'sos': '<sos_u>', 'eos': '<eos_u>'},
            'bspn': {'sos': '<sos_b>', 'eos': '<eos_b>'},
            'db': {'sos': '<sos_db>', 'eos': '<eos_db>'},
            'aspn': {'sos': '<sos_a>', 'eos': '<eos_a>'},
            'resp': {'sos': '<sos_r>', 'eos': '<eos_r>'}
        }
        self.dir_gpt = 'distilgpt2'
        self.exp_domains = ['all'] # hotel, train, attraction, restaurant, taxi, all, except
        self.file_data = 'data_for_damd.json'
        self.key_metric_update = 'score'
        self.limit_vocab_aspn = False
        self.limit_vocab_bspn = False
        self.lst_key_metric = ['inform', 'success', 'bleu', 'score']
        self.lst_name_element = ['user', 'bspn', 'db', 'aspn', 'resp']
        self.maxlen = 1024
        self.method_metric_update = max
        self.pad_id = -1
        self.path_domain_files = './data/multi-woz-processed/domain_files.json'
        self.path_lst_test = './data/multi-woz/testListFile.json'
        self.path_lst_val = './data/multi-woz/valListFile.json'
        self.path_multi_act_train = './data/multi-woz-processed/multi_act_mapping_train.json'
        self.path_slot_value_set_processed = 'db/value_set_processed.json'
        self.path_vocab_eval = None
        self.path_vocab_train = './data/multi-woz-processed/vocab'
        self.use_last_gen = True
        self.use_multi_act_train = False
        self.value_metric_best = None
        self.vocab_size = 3000
        
        self.use_true_bspn_for_ctr_eval = True        
        self.use_true_domain_for_ctr_eval = True
        self.limit_bspn_vocab = False
        self.limit_aspn_vocab = False
        self.same_eval_as_cambridge = True
        self.same_eval_act_f1_as_hdsa = False
        
        return

#----------------------------------------
config_global = _Config()
