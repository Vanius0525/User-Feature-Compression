"""
UPRec tasks:
  1. Mask Item Prediction
  2. User Attribute Prediction
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import MyDataset
from utils import setup_seed
from tqdm import tqdm

MASK_ITEM_ID = 0

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/yelp/proc_data_6', help='data directory')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--embed_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--max_seq_len', type=int, default=10, help='maximum sequence length')
    parser.add_argument('--mask_prob', type=float, default=0.15, help='masking probability')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./pretrain_model', help='directory to save the model')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device')
    parser.add_argument('--save_embed', default='false', type=str, help='whether to save embedding')
    return parser.parse_args()

# dataset class
class UPRecDataset(MyDataset):
    def __init__(self, data_dir, set_type, max_seq_len, mask_prob):
        super().__init__(data_dir, set=set_type, max_hist_len=max_seq_len)
        self.mask_prob = mask_prob

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        item_seq = data['hist_iid_seq']
        mask = torch.rand(len(item_seq)) < self.mask_prob
        masked_seq = item_seq.clone()
        masked_seq[mask] = MASK_ITEM_ID
        return {
            'user_id': data['uid_attr'],
            'item_seq': masked_seq,
            'original_seq': item_seq,
            'mask': mask,
            'label': data['lb']
        }

# model class
class PretrainRecModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, max_seq_len, user_attr_num, user_attr_ft_num):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=0.1),
            num_layers=2
        )
        self.item_pred_head = nn.Linear(embed_dim, num_items)
        self.user_attr_pred_heads = nn.ModuleList([nn.Linear(embed_dim, user_attr_num) for _ in range(user_attr_ft_num)])

    def forward(self, user_ids, item_seq, mask_positions):
        if item_seq.max() >= self.item_embedding.num_embeddings:
            raise ValueError(f"item_seq contains index {item_seq.max()} which is out of range for item_embedding with size {self.item_embedding.num_embeddings}")
        if user_ids.max() >= self.user_embedding.num_embeddings:
            raise ValueError(f"user_ids contains index {user_ids.max()} which is out of range for user_embedding with size {self.user_embedding.num_embeddings}")
        item_emb = self.item_embedding(item_seq)
        pos_emb = self.pos_embedding(torch.arange(item_seq.size(1), device=item_seq.device))
        seq_emb = item_emb + pos_emb
        encoded_seq = self.transformer(seq_emb)
        masked_item_logits = self.item_pred_head(encoded_seq[mask_positions])
        user_emb = encoded_seq.mean(dim=1)
        user_attr_logits = torch.stack([head(user_emb) for head in self.user_attr_pred_heads], dim=1)
        return masked_item_logits, user_attr_logits

# training function
def train(args):
    device = args.device
    setup_seed(args.seed)
    train_dataset = UPRecDataset(args.data_dir, 'train', args.max_seq_len, args.mask_prob)
    print(f"Training dataset size: {len(train_dataset)}")  # 打印训练数据量
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    user_attr_ft_num = train_dataset.user_attr_ft_num
    print(f"User attribute feature number: {user_attr_ft_num}")

    model = PretrainRecModel(
        num_users=train_dataset.user_attr_num + 1,
        num_items=train_dataset.item_num + 1,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        user_attr_num=train_dataset.user_attr_num + 1,
        user_attr_ft_num=user_attr_ft_num  
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            user_ids = batch['user_id'].to(device)
            item_seq = batch['item_seq'].to(device)
            original_seq = batch['original_seq'].to(device)
            mask = batch['mask'].to(device)

            mask_positions = mask.nonzero(as_tuple=True)
            masked_item_logits, user_attr_logits = model(user_ids, item_seq, mask_positions)

            item_loss = criterion(masked_item_logits, original_seq[mask])
            
            user_attr_loss = 0
            for i in range(user_ids.size(1)):
                user_attr_loss += criterion(user_attr_logits[:, i], user_ids[:, i])
            user_attr_loss /= user_ids.size(1)

            loss = item_loss + user_attr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s")
       
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'pretrain_model.pt'))

if __name__ == '__main__':
    args = parse_args()
    train(args)
