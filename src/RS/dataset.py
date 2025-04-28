import pickle
import os
import numpy as np

import torch
import torch.utils.data as Data

from utils import load_json, load_pickle

class MyDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='ctr', max_hist_len=30, 
                 augment=False, aug_prefix=None, kg_aug=False, kg_aug_dim=100,
                 group_aug=False, load=False, pretrain_model_dir='pretrain_model/bs128_lr0.0001/dim32_lr0.0001_mask0.15_bs128_ep20/best_pretrain_model.pt', device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(task, 'task')
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.kg_aug = kg_aug
        self.kg_aug_dim = kg_aug_dim
        self.group_aug = group_aug
        self.set = set
        print(task, 'task')
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

        # self.device = device
        # self.pretrain_model_dir = pretrain_model_dir
        # print('pretrain model dir:', pretrain_model_dir)
        # # 加载预训练模型
        # if pretrain_model_dir:
        #     self.pretrained_model = PretrainRecModel(
        #         num_users=self.stat['user_attribute_num'] + 1,
        #         num_items=self.stat['item_num'] + 1,
        #         embed_dim=32,  # 假设嵌入维度为32，可根据实际情况调整
        #         max_seq_len=max_hist_len,
        #         user_attr_num=self.stat['user_attribute_num'] + 1,
        #         user_attr_ft_num=self.user_attr_ft_num
        #     )
        #     self.pretrained_model.load_state_dict(torch.load(pretrain_model_dir, map_location=device))
        #     self.pretrained_model.eval()  # 设置为评估模式
        #     for param in self.pretrained_model.parameters():
        #         param.requires_grad = False  # 冻结预训练模型参数
        # else:
        #     self.pretrained_model = None

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

            # # 检查 hist_item_seq 是否为空
            # if self.pretrained_model and len(hist_item_seq) > 0:
            #     with torch.no_grad():
            #         mask_positions = torch.zeros(len(hist_item_seq), dtype=torch.bool).to(self.device)
            #         mask_positions[0] = True  # 确保至少有一个位置为 True
            #         _, _, user_emb = self.pretrained_model(
            #             user_ids=torch.tensor(user_attr_id).to(self.device).unsqueeze(0),
            #             item_seq=torch.tensor(hist_item_seq).to(self.device).unsqueeze(0),
            #             mask_positions=mask_positions.unsqueeze(0)
            #         )
            #     out_dict['user_emb_pretrained'] = user_emb.squeeze(0)  # 添加用户表示到输出字典
            #     # print('user emb shape in use:', out_dict['user_emb_pretrained'].shape)
            #     # print('user emb in use:', out_dict['user_emb_pretrained'])
            # else:
            #     # 如果 hist_item_seq 为空，生成默认用户嵌入
            #     out_dict['user_emb_pretrained'] = torch.zeros(1, self.pretrained_model.item_pred_head.out_features)  # 提供默认值
            #     # print('Default user emb generated due to empty hist_item_seq')

        else:
            raise NotImplementedError

        return out_dict
