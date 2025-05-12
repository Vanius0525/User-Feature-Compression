import pickle
import os
import numpy as np

import torch
import torch.utils.data as Data
import torch.nn as nn


from utils import load_json, load_pickle


class MyDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='ctr', max_hist_len=10, 
                 augment=False, aug_prefix=None, kg_aug=False, kg_aug_dim=100,
                 group_aug=False, load=False, pretrain_model_dir='pretrain_model_may_9th/bs768_lr0.0005/dim32_lr0.0005_mask0.2_bs768_ep20/best_pretrain_model.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(task, 'task')
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
        self.user_attr_ft_num = 17  # todo

        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        self.user2attribution = load_json(data_path + '/user2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        
        # self.device = device
        # self.pretrain_model_dir = pretrain_model_dir
        # print('pretrain model dir:', pretrain_model_dir)
        # # 加载预训练模型
        # self.pretrained_model = PretrainRecModel(
        #     user_attr_num=self.stat['user_attribute_num'] + 1,
        #     num_items=self.stat['item_num'] + 1,
        #     embed_dim=32,  # 假设嵌入维度为32，可根据实际情况调整
        #     max_seq_len=max_hist_len,
        #     user_attr_ft_num=self.user_attr_ft_num,
        #     attr_num=self.stat['attribute_num'] + 1,
        # ).to(device)
        # self.pretrained_model.load_state_dict(torch.load(pretrain_model_dir, map_location=device))
        # self.pretrained_model.eval()  # 设置为评估模式
        # for param in self.pretrained_model.parameters():
        #     param.requires_grad = False  # 冻结预训练模型参数


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
                'uid_attr': torch.tensor(user_attr_id).long(),
            }

        else:
            raise NotImplementedError

        return out_dict
