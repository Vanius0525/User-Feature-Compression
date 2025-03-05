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
    'yelp': 'businesses',
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


def generate_rerank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    full_data = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        idx = start_idx
        seq_len = len(item_seq)
        while idx < seq_len:
            end_idx = min(idx + rerank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            neg_sample = random.sample(item_set, neg_sample_num)
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in
                             chosen_rating] + [0 for _ in range(neg_sample_num)]
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([uid, idx, candidates, candidate_lbs])
            idx = end_idx
    print('user num', len(uid_set), 'data num', len(full_data))
    print(full_data[:5])
    return full_data


def get_hist_text(hist_item_seq, hist_rating_seq, itemid2title, id2cate_name, items2attributes):
    history_texts = []
    for iid, rating in zip(hist_item_seq, hist_rating_seq):
        attr = id2cate_name[str(iid)]
        tmp = '{} ({}): {} stars\n'.format(itemid2title[str(iid)], attr, int(rating))
        history_texts.append(tmp)
    history_texts = ''.join(history_texts)
    return history_texts

def generate_hist_prompt(sequence_data, item2attribute, datamap, lm_hist_idx, dataset_name):
    itemid2title = datamap['itemid2title']
    item_set = list(itemid2title.keys())
    id2cate_name = datamap['id2cate_name']
    id2user = datamap['id2user']
    embedding_data, generative_data = {}, {}
    print('item2attribute', list(item2attribute.items())[:10])
    item_name = item_name_map[dataset_name]
    for uid, item_rating in sequence_data.items():
        item_seq, rating_seq = item_rating
        cur_idx = lm_hist_idx[uid]
        hist_item_seq = item_seq[:cur_idx]
        hist_rating_seq = rating_seq[:cur_idx]
        user = id2user[uid]
        history_text = get_hist_text(hist_item_seq, hist_rating_seq, itemid2title, id2cate_name, item2attribute)

        embedding = {
            'input': f'The user\'s {item_name} rating sequence:\n{history_text}',
            'output': ''
        }
        hist_prompt = f'Given the user\'s {item_name} rating sequence:\n{history_text}The next {item_name} the user likes: '
        generative = {
            'input': hist_prompt,
            'output': ''
        }
        embedding_data[user] = embedding
        generative_data[user] = generative
    print('embedding data num', len(embedding_data), len(sequence_data))
    print(list(embedding_data.items())[0])
    print('generative data num', len(generative_data), len(sequence_data))
    print(list(generative_data.items())[0])
    return embedding_data, generative_data


def generate_item_prompt(item2attribute, datamap, dataset_name):
    itemid2title = datamap['itemid2title']
    id2cate_name = datamap['id2cate_name']
    id2item = datamap['id2item']
    embedding_data, generative_data = {}, {}
    for iid, title in itemid2title.items():
        item = id2item[iid]
        item_name = item_name_map[dataset_name]
        attr = id2cate_name[iid]
        embedding = {
            'input': f'The next {item_name} the user likes: {title} ({attr}).',
            'output': ''
        }
        generative = {
            'input': f'{title} ({attr})',
            'output': ''
        }
        embedding_data[item] = embedding
        generative_data[item] = generative
    print('embedding data num', len(embedding_data), len(item2attribute))
    print(list(embedding_data.items())[0])
    print('generative data num', len(generative_data), len(item2attribute))
    print(list(generative_data.items())[0])
    return embedding_data, generative_data


def generate_sft_dataset(dataset_name, uid_set, sequence_data, itemid2title, id2cate_name, item2attribute, train_ratio):
    data = []
    item_name = item_name_map[dataset_name]
    hist_prompt_len = 0
    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        for i in range(lm_hist_max, len(item_seq)):
            if rating_seq[i] > rating_threshold:
                target_item = item_seq[i]
                hist_item_seq = item_seq[max(0, i - lm_hist_max): i]
                hist_rating_seq = rating_seq[max(0, i - lm_hist_max): i]
                history_text = get_hist_text(hist_item_seq, hist_rating_seq, itemid2title, id2cate_name, item2attribute)
                hist_prompt = f'Given the user\'s {item_name} rating sequence:\n{history_text}The next {item_name} the user likes: '
                attr = id2cate_name[str(target_item)]
                generative = {
                    # 'input': f'Given the user {item_name} rating sequence:\n{history_text}\nPredict the next {item_name} '
                    #          f'the user will like.',
                    'input': hist_prompt,
                    'output':  f'{itemid2title[str(target_item)]} ({attr})',
                }
                hist_prompt_len += len(hist_prompt)
                data.append(generative)
    random.shuffle(data)
    print(len(data), data[0])
    print('avg len', hist_prompt_len / len(data))
    if dataset_name == 'ml-25m':
        data = data[:300000]
    train_data = data[: int(len(data) * train_ratio)]
    valid_data = data[int(len(data) * train_ratio):]
    return train_data, valid_data


def generate_emb_sft_dataset(dataset_name, uid_set, sequence_data, itemid2title, id2cate_name, item2attribute, train_ratio, data_type='a'):
    data = []
    item_name = item_name_map[dataset_name]
    item_name_set = list(itemid2title.values())
    for uid in uid_set:
        item_seq, rating_seq = sequence_data[str(uid)]
        for i in range(lm_hist_max, len(item_seq)):
            target_item = item_seq[i]
            hist_item_seq = item_seq[max(0, i - lm_hist_max): i]
            hist_rating_seq = rating_seq[max(0, i - lm_hist_max): i]
            history_text = get_hist_text(hist_item_seq, hist_rating_seq, itemid2title, id2cate_name, item2attribute)
            label = 1 if rating_seq[i] > rating_threshold else 0
            attr = id2cate_name[str(target_item)]
            embedding = {
                'text1': f'The user\'s {item_name} rating sequence:\n{history_text}',
                'text2': f'The next {item_name} the user likes: {itemid2title[str(target_item)]} ({attr}).',
                'label': label,
            }
            data.append(embedding)
    random.shuffle(data)
    print(len(data), data[0])
    if dataset_name == 'ml-25m':
        data = data[:300000]
    train_data = data[: int(len(data) * train_ratio)]
    valid_data = data[int(len(data) * train_ratio):]
    return train_data, valid_data



if __name__ == '__main__':
    random.seed(12345)
    DATA_DIR = '../data/'
    DATA_SET_NAME = 'yelp'
    print(DATA_SET_NAME)
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')

    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')
    # print(list(item2attribute.keys())[:10])

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                  train_test_split['train'])
    print('generating ctr test dataset')
    test_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                 train_test_split['test'])
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None

    print('generating reranking train dataset')
    train_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
                                        train_test_split['train'], item_set)
    print('generating reranking test dataset')
    test_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
                                       train_test_split['test'], item_set)
    print('save reranking data')
    save_pickle(train_rerank, PROCESSED_DIR + '/rerank.train')
    save_pickle(test_rerank, PROCESSED_DIR + '/rerank.test')
    train_rerank, test_rerank = None, None

    datamap = load_json(DATAMAP_PATH)

    statis = {
        'rerank_list_len': rerank_list_len,
        'attribute_ft_num': datamap['attribute_ft_num'],
        'rating_threshold': rating_threshold,
        'item_num': len(datamap['id2item']),
        'attribute_num': len(datamap['id2attribute']),
        'rating_num': 5,
        'dense_dim': 0,
    }
    save_json(statis, PROCESSED_DIR + '/stat.json')

    print('generating item prompt')
    item_emb_prompt, item_gen_prompt = generate_item_prompt(item2attribute, datamap, DATA_SET_NAME)
    print('generating history prompt')
    hist_emb_prompt, hist_gen_prompt = generate_hist_prompt(sequence_data, item2attribute,
                                                            datamap, train_test_split['lm_hist_idx'], DATA_SET_NAME)
    print('save prompt data')

    prompt_dir = os.path.join(PROCESSED_DIR, 'prompt')
    os.makedirs(prompt_dir, exist_ok=True)
    save_json(item_emb_prompt, prompt_dir + '/emb_prompt.item')
    save_json(hist_emb_prompt, prompt_dir + '/emb_prompt.hist')
    save_json(hist_gen_prompt, prompt_dir + '/gen_prompt.hist')
    save_json(item_gen_prompt, prompt_dir + '/gen_prompt.item')

    # emb_train = get_random_train(hist_emb_prompt, 50000)
    # save_jsonl(emb_train, f'../gritlm/training/toy_data/{DATA_SET_NAME}_emb.jsonl')
    gen_train, gen_valid = generate_sft_dataset(DATA_SET_NAME, train_test_split['train'], sequence_data,
                                                datamap['itemid2title'], datamap['id2cate_name'], item2attribute, 0.9)
    print('gen', len(gen_train), len(gen_valid))
    save_json(gen_train, f'{prompt_dir}/gen_train.json')
    save_json(gen_valid, f'{prompt_dir}/gen_valid.json')

    gen_train, gen_valid = generate_emb_sft_dataset(DATA_SET_NAME, train_test_split['train'], sequence_data,
                                                datamap['itemid2title'], datamap['id2cate_name'], item2attribute, 0.9)
    print('emb', 'train', len(gen_train), 'test', len(gen_valid))
    save_json(gen_train, f'{prompt_dir}/emb_train.json')
    save_json(gen_valid, f'{prompt_dir}/emb_valid.json')




