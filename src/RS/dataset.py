import pickle
import os
import numpy as np

import torch
import torch.utils.data as Data

from utils import load_json, load_pickle


class MyDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='ctr', max_hist_len=10,
                 augment=False, aug_prefix=None, kg_aug=False, kg_aug_dim=100,
                 group_aug=False, load=False):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.kg_aug = kg_aug
        self.kg_aug_dim = kg_aug_dim
        self.group_aug = group_aug
        self.set = set
        self.data = load_pickle(data_path + f'/{task}.{set}')
        if self.group_aug:
            self.stat = load_json(data_path + f'/{aug_prefix}_group_stat.json')
        elif self.augment:
            self.stat = load_json(data_path + f'/{aug_prefix}_stat.json')
        else:
            self.stat = load_json(data_path + f'/stat.json')
        self.item_num = self.stat['item_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']
        self.dense_dim = self.stat['dense_dim']

        self.user_attr_num = self.stat['user_attribute_num']
        self.user_attr_ft_num = 17 # todo

        print('dense dim!!!', self.dense_dim)
        if task == 'rerank':
            self.max_list_len = self.stat['rerank_list_len']
        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        self.user2attribution = load_json(data_path + '/user2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        self.aug_num = 0
        if group_aug:
            self.hist_aug_data = load_json(data_path + f'group_data/{aug_prefix}_augment.hist')
            self.item_aug_data = load_json(data_path + f'group_data/{aug_prefix}_augment.item')
            self.group_item_aug_data = load_json(data_path + f'group_data/{aug_prefix}_augment.gitm')
            self.group_pos_hst_aug_data = load_json(data_path + f'group_data/{aug_prefix}_augment.gphst')
            self.group_neg_hst_aug_data = load_json(data_path + f'group_data/{aug_prefix}_augment.gnhst')
            self.item_group_map = load_json(data_path + f'group_data/item/result/id2group.json')
            self.user_group_map = load_json(data_path + f'group_data/user/id2pos_neg_group.json')
            self.aug_num += 5
        else:
            if augment:
                if load:
                    print('load original vector')
                    self.hist_aug_data = load_json(data_path + f'/{aug_prefix}_augment.hist')
                    self.item_aug_data = load_json(data_path + f'/{aug_prefix}_augment.item')
                else:
                    self.hist_aug_data = load_json(data_path + f'/{aug_prefix}_augment.hist', float_type=np.float32)
                    self.item_aug_data = load_json(data_path + f'/{aug_prefix}_augment.item', float_type=np.float32)
                self.aug_num += 2
            # print('item key', list(self.item_aug_data.keys())[:6], len(self.item_aug_data), self.item_num)

        if kg_aug:
            raw_kg_map = load_json(os.path.join(data_path, 'KGE', 'itm_ent_embed.map'))
            default = [0 for _ in range(kg_aug_dim)]
            self.kg_map = {iid: raw_kg_map.get(item, default) for iid, item in self.id2item.items()}
            self.aug_num += 1

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        if self.task == 'ctr':
            uid, seq_idx, lb = self.data[_id]
            user_attr_id = self.user2attribution[str(uid)]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            iid = item_seq[seq_idx]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            attri_id = self.item2attribution[str(iid)]
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            out_dict = {
                'iid': torch.tensor(iid).long(),
                'aid': torch.tensor(attri_id).long(),
                'lb': torch.tensor(lb).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long(),
                'uid_attr': torch.tensor(user_attr_id).long()
            }
            if self.augment:
                item_aug_vec = self.item_aug_data[str(self.id2item[str(iid)])]
                # print(self.id2user[str(uid)])
                # print(list(self.hist_aug_data.keys())[:50])
                hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
                out_dict['item_aug_vec'] = torch.tensor(item_aug_vec).float()
                out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()

            if self.kg_aug:
                kg_aug_vec = self.kg_map[str(iid)]
                out_dict['kg_aug_vec'] = torch.tensor(kg_aug_vec).float()
            if self.group_aug:
                item_group_aug_vec = self.group_item_aug_data[str(self.item_group_map[str(iid)])]
                user_pos_group_aug_vec = self.group_pos_hst_aug_data[str(self.user_group_map[str(uid)][1])]
                user_neg_group_aug_vec = self.group_neg_hst_aug_data[str(self.user_group_map[str(uid)][0])]
                # try:
                #     user_neg_group_aug_vec = self.group_neg_hst_aug_data[str(self.user_group_map[str(uid)][0])]
                # except:
                #     print(str(uid))
                #     print(str(self.user_group_map[str(uid)][0]))
                #     print(self.group_neg_hst_aug_data.keys())
                #     print(self.group_neg_hst_aug_data[str(self.user_group_map[str(uid)][0])])
                out_dict['item_group_aug_vec'] = torch.tensor(item_group_aug_vec).float()
                out_dict['user_pos_group_aug_vec'] = torch.tensor(user_pos_group_aug_vec).float()
                out_dict['user_neg_group_aug_vec'] = torch.tensor(user_neg_group_aug_vec).float()
        # elif self.task == 'rerank':
        #     uid, seq_idx, candidates, candidate_lbs = self.data[_id]
        #     candidates_attr = [self.item2attribution[str(idx)] for idx in candidates]
        #     item_seq, rating_seq = self.sequential_data[str(uid)]
        #     hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
        #     hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
        #     hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
        #     hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
        #     out_dict = {
        #         'iid_list': torch.tensor(candidates).long(),
        #         'aid_list': torch.tensor(candidates_attr).long(),
        #         'lb_list': torch.tensor(candidate_lbs).long(),
        #         'hist_iid_seq': torch.tensor(hist_item_seq).long(),
        #         'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
        #         'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
        #         'hist_seq_len': torch.tensor(hist_seq_len).long()
        #     }
        #     if self.augment:
        #         item_aug_vec = [torch.tensor(self.item_aug_data[str(self.id2item[str(idx)])]).float()
        #                         for idx in candidates]
        #         hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
        #         out_dict['item_aug_vec_list'] = item_aug_vec
        #         out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
        #     if self.group_aug:
        #         item_group_aug_vec = [torch.tensor(self.group_item_aug_data[str(self.item_group_map[str(iid)])]).float()
        #                               for iid in candidates]
        #         user_pos_group_aug_vec = self.group_pos_hst_aug_data[str(self.user_group_map[str(uid)][1])]
        #         user_neg_group_aug_vec = self.group_neg_hst_aug_data[str(self.user_group_map[str(uid)][0])]

        #         out_dict['item_group_aug_vec_list'] = item_group_aug_vec
        #         out_dict['user_pos_group_aug_vec'] = torch.tensor(user_pos_group_aug_vec).float()
        #         out_dict['user_neg_group_aug_vec'] = torch.tensor(user_neg_group_aug_vec).float()
        else:
            raise NotImplementedError

        return out_dict


