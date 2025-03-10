'''
split train/test by user IDs, train: test= 9: 1
RS history: recent rated 10 items (pos & neg), ID & attributes & rating
LM history: one lm history for each user(max_len=30, item ID, attributes, rating)
attribute: category
rating >= 4 as positive, rating < 4 as negative, no negative sampling
'''
import os
import random
import tqdm
import html
from collections import defaultdict
import numpy as np
import json
from datetime import date, datetime
from string import ascii_letters, digits, punctuation, whitespace
from pre_utils import set_seed, parse, add_comma, save_json, correct_title

lm_hist_max = 30
train_ratio = 0.9
rating_score = 0.0  # rating score smaller than this score would be deleted
# user 30-core item 15-core
user_core = 30
item_core = 20
attribute_core = 0

def is_proper_float(s):
    # 检查字符串中是否只有一个小数点
    if s.count('.') == 1:
        left, right = s.split('.')
        # 检查小数点左侧是否为数字，右侧是否为非空数字
        return left.isdigit() and right.isdigit() and int(right) != 0
    return False

def yelp(data_file, rating_score):
    datas = []
    all_num, remain_num = 0, 0
    with open(data_file, 'r') as f:
        for line in f:
            inter = json.loads(line)
            all_num += 1
            user = inter.get('user_id', None)
            item = inter.get('business_id', None)
            rating = inter.get('stars', None)
            time = inter.get('date', None)
            if user is not None and item is not None and time is not None and rating is not None:
                if rating < rating_score:
                    continue
                # if is_proper_float(str(rating)):
                #     print('rating', rating)
                time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                datas.append((user, item, time, int(rating)))
                remain_num += 1

    print('total review', all_num, 'remain', remain_num, remain_num / all_num)
    return datas

def bucket_item_review(n):
    if n == 0: return "item_review_0"
    elif 1 <= n <= 5: return "item_review_1-5"
    elif 6 <= n <= 20: return "item_review_6-20"
    elif 21 <= n <= 100: return "item_review_21-100"
    else: return "item_review_101+"

def yelp_meta(meta_file, data_maps):  # return the metadata of products
    datas = {}
    item_asins = set(data_maps['item2id'].keys())
    with open(meta_file, 'r') as f:
        for line in f:
            inter = json.loads(line)
            iid = inter.get('business_id', None)
            title = inter.get('name', None)
            categories = inter.get('categories', None)
            city = inter.get('city', None)
            state = inter.get('state', None)
            postal_code = inter.get('postal_code', None)
            stars = inter.get('stars', 0.0)
            review_count = inter.get('review_count', 0)
            # todo: bucket
            if iid not in item_asins:
                continue
            if categories is not None and title is not None and iid is not None and city is not None:
                categories = categories.split(',')
                new_info = {
                    'categories': categories[:1],
                    'title': title,
                    'cate_name': ','.join(categories[:2]),
                    'city': city,
                    'state': state if state is not None else "", 
                    'postal_code': postal_code if postal_code is not None else "", 
                    'stars': int(stars * 2) / 2,
                    'review_count': review_count
                }
                datas[iid] = new_info
    meta_set = set(datas.keys())
    return datas, item_asins.difference(meta_set)

def bucket_fans(n):
    if n == 0: return "fans_0"
    elif 1 <= n <= 5: return "fans_1-5"
    elif 6 <= n <= 20: return "fans_6-20"
    elif 21 <= n <= 100: return "fans_21-100"
    else: return "fans_101+"

def bucket_review_cnt(n):
    if n == 0: return "review_0"
    elif 1 <= n <= 5: return "review_1-5"
    elif 6 <= n <= 20: return "review_6-20"
    elif 21 <= n <= 100: return "review_21-100"
    else: return "review_101+"

def bucket_user_useful(n):
    if n == 0: return "useful_0"
    elif 1 <= n <= 2: return "useful_1-2"
    elif 3 <= n <= 10: return "useful_3-10"
    elif 11 <= n <= 100: return "useful_11-100"
    else: return "useful_101+"

def bucket_user_funny(n):
    if n == 0: return "funny_0"
    elif 1 <= n <= 2: return "funny_1-2"
    elif 3 <= n <= 10: return "funny_3-10"
    elif 11 <= n <= 100: return "funny_11-100"
    else: return "funny_101+"

def bucket_user_cool(n):
    if n == 0: return "cool_0"
    elif 1 <= n <= 2: return "cool_1-2"
    elif 3 <= n <= 10: return "cool_3-10"
    elif 11 <= n <= 100: return "cool_11-100"
    else: return "cool_101+"

def bucket_compliment_photos(n):
    if n == 0: return "compliment_photos_0"
    elif 1 <= n <= 2: return "compliment_photos_1-2"
    elif 3 <= n <= 10: return "compliment_photos_3-10"
    elif 11 <= n <= 100: return "compliment_photos_11-100"
    else: return "compliment_photos_101+"

def bucket_compliment_writer(n):
    if n == 0: return "compliment_writer_0"
    elif 1 <= n <= 2: return "compliment_writer_1-2"
    elif 3 <= n <= 10: return "compliment_writer_3-10"
    elif 11 <= n <= 100: return "compliment_writer_11-100"
    else: return "compliment_writer_101+"
    
def bucket_compliment_funny(n):
    if n == 0: return "compliment_funny_0"
    elif 1 <= n <= 2: return "compliment_funny_1-2"
    elif 3 <= n <= 10: return "compliment_funny_3-10"
    elif 11 <= n <= 100: return "compliment_funny_11-100"
    else: return "compliment_funny_101+"

def bucket_compliment_cool(n):
    if n == 0: return "compliment_cool_0"
    elif 1 <= n <= 2: return "compliment_cool_1-2"
    elif 3 <= n <= 10: return "compliment_cool_3-10"
    elif 11 <= n <= 100: return "compliment_cool_11-100"
    else: return "compliment_cool_101+"

def bucket_compliment_plain(n):
    if n == 0: return "compliment_plain_0"
    elif 1 <= n <= 2: return "compliment_plain_1-2"
    elif 3 <= n <= 10: return "compliment_plain_3-10"
    elif 11 <= n <= 100: return "compliment_plain_11-100"
    else: return "compliment_plain_101+"

def bucket_compliment_note(n):
    if n == 0: return "compliment_note_0"
    elif 1 <= n <= 2: return "compliment_note_1-2"
    elif 3 <= n <= 10: return "compliment_note_3-10"
    elif 11 <= n <= 100: return "compliment_note_11-100"
    else: return "compliment_note_101+"

def bucket_compliment_list(n):
    if n == 0: return "compliment_list_0"
    elif 1 <= n <= 2: return "compliment_list_1-2"
    elif 3 <= n <= 10: return "compliment_list_3-10"
    elif 11 <= n <= 100: return "compliment_list_11-100"
    else: return "compliment_list_101+"

def bucket_compliment_cute(n):
    if n == 0: return "compliment_cute_0"
    elif 1 <= n <= 2: return "compliment_cute_1-2"
    elif 3 <= n <= 10: return "compliment_cute_3-10"
    elif 11 <= n <= 100: return "compliment_cute_11-100"
    else: return "compliment_cute_101+"

def bucket_compliment_hot(n):
    if n == 0: return "compliment_hot_0"
    elif 1 <= n <= 2: return "compliment_hot_1-2"
    elif 3 <= n <= 10: return "compliment_hot_3-10"
    elif 11 <= n <= 100: return "compliment_hot_11-100"
    else: return "compliment_hot_101+"

def bucket_compliment_more(n):
    if n == 0: return "compliment_more_0"
    elif 1 <= n <= 2: return "compliment_more_1-2"
    elif 3 <= n <= 10: return "compliment_more_3-10"
    elif 11 <= n <= 100: return "compliment_more_11-100"
    else: return "compliment_more_101+"

def bucket_compliment_profile(n):
    if n == 0: return "compliment_profile_0"
    elif 1 <= n <= 2: return "compliment_profile_1-2"
    elif 3 <= n <= 10: return "compliment_profile_3-10"
    elif 11 <= n <= 100: return "compliment_profile_11-100"
    else: return "compliment_profile_101+"

def yelp_user(user_file, data_maps):
    user_meta = {}
    user_ids = set(data_maps['user2id'].keys())
    with open(user_file, 'r') as f:
        for line in f:
            user = json.loads(line)
            uid = user.get('user_id', None)
            review_count = user.get('review_count', 0)
            useful = user.get('useful', 0)
            funny = user.get('funny', 0)
            cool = user.get('cool', 0)
            fans = user.get('fans', 0)
            average_stars = user.get('average_stars', 0)
            compliment_hot = user.get('compliment_hot', 0)
            compliment_more = user.get('compliment_more', 0)
            compliment_profile = user.get('compliment_profile', 0)
            compliment_cute = user.get('compliment_cute', 0)
            compliment_list = user.get('compliment_list', 0)
            compliment_note = user.get('compliment_note', 0)
            compliment_plain = user.get('compliment_plain', 0)
            compliment_cool = user.get('compliment_cool', 0)
            compliment_funny = user.get('compliment_funny', 0)
            compliment_writer = user.get('compliment_writer', 0)
            compliment_photos = user.get('compliment_photos', 0)
            # todo: bucket
            if uid not in user_ids:
                continue
            user_info = {
                "review_count": bucket_review_cnt(review_count),
                "useful": bucket_user_useful(useful),
                "funny": bucket_user_funny(funny),
                "cool": bucket_user_cool(cool),
                "fans": bucket_fans(fans),
                "average_stars": int(average_stars * 2) / 2,
                "compliment_hot": bucket_compliment_hot(compliment_hot),
                "compliment_more": bucket_compliment_more(compliment_more),
                "compliment_profile": bucket_compliment_profile(compliment_profile),
                "compliment_cute": bucket_compliment_cute(compliment_cute),
                "compliment_list": bucket_compliment_list(compliment_list),
                "compliment_note": bucket_compliment_note(compliment_note),
                "compliment_plain": bucket_compliment_plain(compliment_plain),
                "compliment_cool": bucket_compliment_cool(compliment_cool),
                "compliment_funny": bucket_compliment_funny(compliment_funny),
                "compliment_writer": bucket_compliment_writer(compliment_writer),
                "compliment_photos": bucket_compliment_photos(compliment_photos),
            }
            user_meta[uid] = user_info
        return user_meta


# categories and brand is all attribute
def get_attribute_ml(meta_infos, datamaps):
    attributes = defaultdict(int)
    for iid, info in meta_infos.items():
        for cate in info['categories']:
            # attributes[cates[1].strip()] += 1
            attributes[cate] += 1

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in meta_infos.items():
        new_meta[iid] = []
        for cate in info['categories']:
            new_meta[iid].append(cate)
        # if len(new_meta[iid]) > 2:
        #     print(new_meta[iid], info['categories'])
    # mapping
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    itemid2title = {}
    id2cate_name = {}

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
        itemid2title[item_id] = meta_infos[iid]['title']
        id2cate_name[item_id] = meta_infos[iid]['cate_name']


    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, '
          f'Avg.:{np.mean(attribute_lens):.4f}')
    # update datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['attributeid2num'] = attributeid2num
    datamaps['itemid2title'] = itemid2title
    datamaps['items2attributes'] = items2attributes
    datamaps['id2cate_name'] = id2cate_name
    datamaps['attribute_ft_num'] = 1

    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes

def get_user_attribute(user_meta, datamaps):
    attributes = defaultdict(int)
    for uid, attr_dict in user_meta.items():
        for key, value in attr_dict.items():
            combined_attr = f"{key}_{value}"
            attributes[combined_attr] += 1

    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    for attr in attributes.keys():
        if attr not in attribute2id:
            attribute2id[attr] = attribute_id
            id2attribute[attribute_id] = attr
            attribute_id += 1

    users2attributes = {}
    for uid, attr_dict in user_meta.items():
        user_id = datamaps['user2id'][uid]
        attr_list = [attribute2id[f"{key}_{attr_dict[key]}"] for key in attr_dict]
        users2attributes[user_id] = attr_list

    datamaps['user_attribute2id'] = attribute2id
    datamaps['id2user_attribute'] = id2attribute
    datamaps['users2attributes'] = users2attributes

    return len(attribute2id), datamaps, users2attributes

def get_interaction(datas):  # return a dict, key is user and value is a list of items
    user_seq = {}
    for data in datas:
        user, item, time, rating = data
        if user in user_seq:
            user_seq[user].append((item, time, rating))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time, rating))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # sorting by time
        items = []
        for t in item_time:
            items.append([t[0], t[2]])
        user_seq[user] = items
    return user_seq


# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    rating_count = defaultdict(int)
    for user, items in user_items.items():
        for item, rating in items:
            user_count[user] += 1
            item_count[item] += 1
            rating_count[rating] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, rating_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, rating_count, False
    return user_count, item_count, rating_count, True  # guarantee Kcore


# filter K-core
def filter_Kcore(user_items, user_core, item_core):
    user_count, item_count, rating_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:  # delete user
                user_items.pop(user)
            else:
                for item, rating in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove([item, rating])
        user_count, item_count, rating_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(user_items):  # user_items dict
    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid
    id2user = {}  # uid 2 raw
    id2item = {}  # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    lm_hist_idx = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = user_id
            id2user[user_id] = user
            user_id += 1
        iids = []  # item id lists
        ratings = []
        for item, rating in items:
            if item not in item2id:
                item2id[item] = item_id
                id2item[item_id] = item
                item_id += 1
            iids.append(item2id[item])
            ratings.append(rating)
        uid = user2id[user]
        lm_hist_idx[uid] = min((len(iids) + 1) // 2, lm_hist_max)
        final_data[uid] = [iids, ratings]
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
    }
    return final_data, user_id - 1, item_id - 1, data_maps, lm_hist_idx


def update_data(user_items, item_diff, id2item):
    new_data = {}
    lm_hist_idx = {}
    for user, user_data in user_items.items():
        iids, ratings = user_data
        new_idds, new_ratings = [], []
        for id, rating in zip(iids, ratings):
            if id2item[id] not in item_diff:
                new_idds.append(id)
                new_ratings.append(rating)
        new_data[user] = [new_idds, new_ratings]
        lm_hist_idx[user] = min((len(iids) + 1) // 2, lm_hist_max)
        # item_num += len(new_idds)
    item_num = len(id2item) - len(item_diff)
    return new_data, item_num, lm_hist_idx


def preprocess(data_file, meta_file, user_file, processed_dir):
    datas = yelp(data_file, rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_file} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    if item_core > 0 or user_core > 0:
        user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
        print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_count, item_count, rating_count, _ = check_Kcore(user_items, user_core=user_core,
                                                          item_core=item_core)  ## user_count: number of interaction for each user
    user_items, user_num, item_num, data_maps, lm_hist_idx = id_map(user_items)

    print('get meta infos')
    meta_infos, item_diff = yelp_meta(meta_file, data_maps)
    if item_diff:
        print('diff num', len(item_diff))
        user_items, item_num, lm_hist_idx = update_data(user_items, item_diff, data_maps['id2item'])
    else:
        print('no different item num')
    '''
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        'lm_hist_idx': lm_hist_idx
    }
    '''

    print('get user infos')
    user_meta = yelp_user(user_file, data_maps)

    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    # rating_count_list = list(rating_count.values())
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100

    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)
    print(rating_count, (rating_count[4] + rating_count[5]) / sum(list(rating_count.values())))
    print((rating_count[5]) / sum(list(rating_count.values())))

    # train/test split
    user_set = list(user_items.keys())
    random.shuffle(user_set)
    train_user = user_set[:int(len(user_set) * train_ratio)]
    test_user = user_set[int(len(user_set) * train_ratio):]
    train_test_split = {
        'train': train_user,
        'test': test_user,
        'lm_hist_idx': lm_hist_idx
    }
    print('user items sample:', user_items[user_set[0]])

    print('Begin extracting meta infos...')

    attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_ml(meta_infos, data_maps)
    user_attr_num, datamaps, user2attributes = get_user_attribute(user_meta, data_maps)
    print(f'User Attribute Num: {user_attr_num}')
    for uid, items in user_items.items():
        item, rating = items
        for i in item:
            if i not in item2attributes:
                print('not in', i)

    sample_item = list(datamaps['itemid2title'].items())[:20]
    print('itemid2title sample')
    for itemid, title in sample_item:
        cate = item2attributes[itemid]
        print('Title:', title)
        print('Category:', datamaps['id2attribute'][cate[0]])


    print(f'{meta_file} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # -------------- Save Data ---------------
    save_data_file = processed_dir + '/sequential_data.json'  # interaction sequence between user and item
    item2attributes_file = processed_dir + '/item2attributes.json'  # item and corresponding attributes
    datamaps_file = processed_dir + '/datamaps.json'  # datamap
    split_file = processed_dir + '/train_test_split.json'  # train/test splitting
    user2attributes_file = processed_dir + '/user2attributes.json'
    '''
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        datamaps['attribute2id'] = attribute2id
        datamaps['id2attribute'] = id2attribute
        datamaps['attributeid2num'] = attributeid2num
        datamaps['itemid2title'] = itemid2title
    }
    '''

    save_json(user_items, save_data_file)
    save_json(item2attributes, item2attributes_file)
    save_json(datamaps, datamaps_file)
    save_json(train_test_split, split_file)
    save_json(user2attributes, user2attributes_file)


if __name__ == '__main__':
    set_seed(1234)
    DATA_DIR = '../../data/'
    DATA_SET_NAME = 'Yelp'
    DATA_FILE = os.path.join(DATA_DIR, DATA_SET_NAME + '/yelp_academic_dataset_review.json')
    META_FILE = os.path.join(DATA_DIR, DATA_SET_NAME + '/yelp_academic_dataset_business.json')
    USER_FILE = os.path.join(DATA_DIR, DATA_SET_NAME + '/yelp_academic_dataset_user.json')
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data_4')

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    preprocess(DATA_FILE, META_FILE, USER_FILE, PROCESSED_DIR)

