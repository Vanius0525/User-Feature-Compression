"""
UPRec tasks:
  1. Mask Item Prediction
  2. User Attribute Prediction
"""

import argparse
import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MyDataset
from utils import setup_seed
from tqdm import tqdm

MASK_ITEM_ID = 0

def setup_logging(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_filename = os.path.join(save_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def log_experiment_summary(args, total_loss, log_file, elapsed_time):
    summary_file = os.path.join(args.base_save_dir, "summary.txt")
    with open(summary_file, "a") as f:
        summary_line = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"Exp: {args.exp_name} | "
            f"EmbedDim: {args.embed_dim}, LR: {args.lr}, MaskProb: {args.mask_prob}, "
            f"Batch: {args.batch_size}, Epochs: {args.epochs} | "
            f"Final Train Loss: {total_loss:.4f}, Time: {elapsed_time:.2f}s | "
            f"LogFile: {log_file}\n"
        )
        f.write(summary_line)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/yelp/proc_data_6', help='data directory')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--embed_dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--max_seq_len', type=int, default=30, help='maximum sequence length')
    parser.add_argument('--mask_prob', type=float, default=0.2, help='masking probability')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./pretrain_model_new', help='directory to save the model and logs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device')
    parser.add_argument('--save_embed', default='false', type=str, help='whether to save embedding')
    args = parser.parse_args()
    
    args.exp_name = f"dim{args.embed_dim}_lr{args.lr}_mask{args.mask_prob}_bs{args.batch_size}_ep{args.epochs}"
    args.base_save_dir = args.save_dir
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return args

class PretrainRecModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, max_seq_len, user_attr_num, user_attr_ft_num, attr_num, attr_ft_num):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.attr_embedding = nn.Embedding(attr_num, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=0.1, batch_first=True),  # Set batch_first=True
            num_layers=2
        )
        self.item_pred_head = nn.Linear(embed_dim, num_items)
        self.user_attr_pred_heads = nn.ModuleList([nn.Linear(embed_dim, user_attr_num) for _ in range(user_attr_ft_num)])

    def forward(self, user_ids, attr_seq, mask_positions):
        if user_ids.max() >= self.user_embedding.num_embeddings:
            raise ValueError(f"user_ids contains index {user_ids.max()} which is out of range for user_embedding with size {self.user_embedding.num_embeddings}")
        
        # 获取物品属性的embedding
        attr_emb = self.attr_embedding(attr_seq).sum(dim=2)  # 对每个物品的多个属性求和
        pos_emb = self.pos_embedding(torch.arange(attr_seq.size(1), device=attr_seq.device))
        
        # 将物品属性embedding与位置embedding相加
        seq_emb = attr_emb + pos_emb
        encoded_seq = self.transformer(seq_emb)
        masked_item_logits = self.item_pred_head(encoded_seq[mask_positions])
        user_emb = encoded_seq.mean(dim=1)
        user_attr_logits = torch.stack([head(user_emb) for head in self.user_attr_pred_heads], dim=1)
        return masked_item_logits, user_attr_logits

class UPRecDataset(MyDataset):
    def __init__(self, data_dir, set_type, max_seq_len, mask_prob, task='ctr'):
        super().__init__(data_dir, set=set_type, task=task, max_hist_len=max_seq_len)
        self.mask_prob = mask_prob

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        item_seq = data['hist_iid_seq']
        attr_seq = data['hist_aid_seq']  # 物品属性序列
        mask = torch.rand(len(item_seq)) < self.mask_prob  # 随机生成mask，概率为self.mask_prob（默认为0.2）
        masked_seq = item_seq.clone()
        masked_seq[mask] = MASK_ITEM_ID  # 将被mask的item替换为MASK_ITEM_ID
        return {
            'user_id': data['uid_attr'],
            'attr_seq': attr_seq,  # 返回物品属性序列
            'original_seq': item_seq,
            'mask': mask,
            'label': data['lb']
        }

def validate(args, model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            user_ids = batch['user_id'].to(device)
            attr_seq = batch['attr_seq'].to(device)  # 加载物品属性序列
            original_seq = batch['original_seq'].to(device)
            mask = batch['mask'].to(device)
            mask_positions = mask.nonzero(as_tuple=True)
            masked_item_logits, user_attr_logits = model(user_ids, attr_seq, mask_positions)
            item_loss = criterion(masked_item_logits, original_seq[mask])
            user_attr_loss = 0
            for i in range(user_ids.size(1)):
                user_attr_loss += criterion(user_attr_logits[:, i], user_ids[:, i])
            user_attr_loss /= user_ids.size(1)
            loss = item_loss + user_attr_loss
            total_val_loss += loss.item() * user_ids.size(0)
            total_samples += user_ids.size(0)
    return total_val_loss / total_samples

# # Custom collate function to handle variable-length sequences
# def custom_collate_fn(batch):
#     try:
#         return torch.utils.data._utils.collate.default_collate(batch)
#     except RuntimeError as e:
#         logging.error(f"Collate function error: {e}")
#         raise

def train(args):
    device = args.device
    setup_seed(args.seed)
    
    train_dataset = UPRecDataset(args.data_dir, 'train', args.max_seq_len, args.mask_prob)
    val_dataset = UPRecDataset(args.data_dir, 'test', args.max_seq_len, args.mask_prob)
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    user_attr_ft_num = train_dataset.user_attr_ft_num
    logging.info(f"User attribute feature number: {user_attr_ft_num}")

    model = PretrainRecModel(
        num_users=train_dataset.user_attr_num + 1,
        num_items=train_dataset.item_num + 1,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        user_attr_num=train_dataset.user_attr_num + 1,
        user_attr_ft_num=user_attr_ft_num,
        attr_num=train_dataset.attr_num + 1,  # 添加物品属性数量
        attr_ft_num=train_dataset.attr_ft_num
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    total_loss_all_epochs = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            user_ids = batch['user_id'].to(device)
            attr_seq = batch['attr_seq'].to(device)  # 加载物品属性序列
            original_seq = batch['original_seq'].to(device)
            mask = batch['mask'].to(device)

            mask_positions = mask.nonzero(as_tuple=True)
            masked_item_logits, user_attr_logits = model(user_ids, attr_seq, mask_positions)

            item_loss = criterion(masked_item_logits, original_seq[mask])
            user_attr_loss = 0
            for i in range(user_ids.size(1)):
                user_attr_loss += criterion(user_attr_logits[:, i], user_ids[:, i])
            user_attr_loss /= user_ids.size(1)

            loss = 1.0 * item_loss + 0.3 * user_attr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        total_loss_all_epochs = total_loss  
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s")
        
        val_loss = validate(args, model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_save_path = os.path.join(args.save_dir, 'best_pretrain_model.pt')
            torch.save(model.state_dict(), best_model_save_path)
            logging.info(f"New best model saved to {best_model_save_path} with Validation Loss: {best_val_loss:.4f}")

    training_time = time.time() - start_time
    logging.info(f"Training finished in {training_time:.2f}s")
    
    final_model_save_path = os.path.join(args.save_dir, 'pretrain_model.pt')
    torch.save(model.state_dict(), final_model_save_path)
    logging.info(f"Final model saved to {final_model_save_path}")
    
    log_experiment_summary(args, total_loss_all_epochs, log_file, training_time)

if __name__ == '__main__':
    args = parse_args()
    log_file = setup_logging(args.save_dir)
    logging.info("Starting training experiment: %s", args.exp_name)
    train(args)
    logging.info("Training completed.")
