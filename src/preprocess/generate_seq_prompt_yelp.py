import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING, save_jsonl

rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10
rating_threshold = 3
max_neg_num = 9
lm_hist_max = 30

item_name_map = {
    'ml-1m': 'movie',
    'ml-25m': 'movie',
    'Yelp': 'businesses',
}

def generate_ctr_data(sequence_data, lm_hist_idx, uid_set):
    # print(list(lm_hist_idx.values())[:10])
    full_data = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        for idx in range(start_idx, len(item_seq)):
            full_data.append([uid, idx, 1 if rating_seq[idx] > rating_threshold else 0])
    print('user num', len(uid_set), 'data num', len(full_data))
    print(full_data[:5])
    return full_data



if __name__ == '__main__':
    random.seed(12345)
    DATA_DIR = '../../data/'
    DATA_SET_NAME = 'Yelp'
    print(DATA_SET_NAME)
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data_6')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')
    USER2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'user2attributes.json')


    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    user2attribute = load_json(USER2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')
    print(list(item2attribute.keys())[:10])

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                  train_test_split['train'])
    print('generating ctr test dataset')
    test_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                 train_test_split['test'])
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    

    datamap = load_json(DATAMAP_PATH)

    statis = {
        'rerank_list_len': rerank_list_len,
        'attribute_ft_num': datamap['attribute_ft_num'],
        'rating_threshold': rating_threshold,
        'item_num': len(datamap['id2item']),
        'attribute_num': len(datamap['id2attribute']),
        'user_num': len(datamap['id2user']),
        'user_attribute_num': len(datamap['id2user_attribute']),
        'rating_num': 5,
        'dense_dim': 0,
    }
    save_json(statis, PROCESSED_DIR + '/stat.json')



