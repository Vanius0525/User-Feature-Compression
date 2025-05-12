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
MASK_ATTR_ID = 0

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
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./pretrain_model_new', help='directory to save the model and logs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', default='cpu', type=str, help='device')
    parser.add_argument('--save_embed', default='false', type=str, help='whether to save embedding')
    args = parser.parse_args()
    
    args.exp_name = f"dim{args.embed_dim}_lr{args.lr}_mask{args.mask_prob}_bs{args.batch_size}_ep{args.epochs}"
    args.base_save_dir = args.save_dir
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return args

class PretrainRecModel(nn.Module):
    def __init__(self, user_attr_num, num_items, embed_dim, max_seq_len, user_attr_ft_num, attr_num):
        super().__init__()
        self.attr_embedding = nn.Embedding(attr_num, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len+2, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dropout=0.5),
            num_layers=2
        )
        self.item_pred_head = nn.Linear(embed_dim, num_items)
        self.user_attr_pred_heads = nn.ModuleList([nn.Linear(embed_dim, user_attr_num) for _ in range(user_attr_ft_num)])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids, mask_positions):
        batch_size, seq_len = input_ids.shape
        # print("input_ids_shape:", input_ids.shape)
        # print("batch_size:", batch_size, "seq_len:", seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        
        attr_emb = self.attr_embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        pos_emb = self.pos_embedding(position_ids)  # (batch_size, seq_len, embed_dim)
        # print("attr_emb_shape:", attr_emb.shape, "pos_emb_shape:", pos_emb.shape)
        seq_emb = attr_emb + pos_emb  # (batch_size, seq_len, embed_dim)
        seq_emb = self.dropout(seq_emb)  # (batch_size, seq_len, embed_dim)
        seq_emb = seq_emb.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        
        states = self.transformer(seq_emb)  # (seq_len, batch_size, embed_dim)
        states = states.transpose(0, 1)  # (batch_size, seq_len, embed_dim)

        if mask_positions is not None:
            masked_item_logits = self.item_pred_head(states[mask_positions])
        else:
            masked_item_logits = None 

        user_emb = states.max(dim=1).values  # 使用最大池化
        user_attr_logits = torch.stack([head(user_emb) for head in self.user_attr_pred_heads], dim=1)
        return masked_item_logits, user_attr_logits, user_emb

class UPRecDataset(MyDataset):
    def __init__(self, data_dir, set_type, max_seq_len, mask_prob, task='ctr'):
        super().__init__(data_dir, set=set_type, task=task, max_hist_len=max_seq_len)
        self.mask_prob = mask_prob

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        item_seq = data['hist_iid_seq']
        attr_seq = data['hist_aid_seq']
        attr_seq = attr_seq.squeeze(-1)
        # print("item_seq_shape:", item_seq.shape, "attr_seq_shape:", attr_seq.shape)
        mask = torch.rand(len(item_seq)) < self.mask_prob 
        masked_seq = item_seq.clone()
        masked_seq[mask] = MASK_ITEM_ID  
        attr_mask = mask.clone()
        masked_attr_seq = attr_seq.clone()
        masked_attr_seq[attr_mask] = MASK_ATTR_ID  
        
        return {
            'user_attr': data['uid_attr'],
            'attr_seq': attr_seq,  
            'original_seq': item_seq,
            'mask': mask,
            'label': data['lb'],
            'masked_seq': masked_seq,
            'masked_attr_seq': masked_attr_seq
        }

def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            user_attr = batch['user_attr'].to(device)
            attr_seq = batch['masked_attr_seq'].to(device)
            original_seq = batch['original_seq'].to(device)
            mask = batch['mask'].to(device)
            mask_positions = mask.nonzero(as_tuple=True)
            masked_item_logits, user_attr_logits, _ = model(attr_seq, mask_positions)
            item_loss = criterion(masked_item_logits, original_seq[mask])
            user_attr_loss = 0
            for i in range(user_attr.size(1)):
                user_attr_loss += criterion(user_attr_logits[:, i], user_attr[:, i])
            user_attr_loss /= user_attr.size(1)
            loss = item_loss + 0.3 * user_attr_loss
            total_val_loss += loss.item() * user_attr.size(0)
            total_samples += user_attr.size(0)
    return total_val_loss / total_samples


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
        user_attr_num=train_dataset.user_attr_num + 1,
        num_items=train_dataset.item_num + 1,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        user_attr_ft_num=user_attr_ft_num,
        attr_num=train_dataset.attr_num + 1, 
    ).to(device)
    print("item_num:", train_dataset.item_num, "user_attr_num:", train_dataset.user_attr_num, "attr_num:", train_dataset.attr_num)

    optimizer = optim.Adam(model.parameters(), lr=1e-3) # according to the paper
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    total_loss_all_epochs = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            user_attr = batch['user_attr'].to(device)
            attr_seq = batch['masked_attr_seq'].to(device)  # 使用掩码后的属性序列
            original_seq = batch['original_seq'].to(device) # item sequence
            # print("original_seq_shape:", original_seq.shape)
            # print("attr_seq_shape:", attr_seq.shape)
            mask = batch['mask'].to(device)
            # print("mask_shape:", mask.shape)
            # print("mask:", mask)
            mask_positions = mask.nonzero(as_tuple=True)
            # print("mask_positions_shape:", mask_positions.shape)
            # print("mask_positions:", mask_positions)
            # print("attr_seq shape:", attr_seq.shape)
            masked_item_logits, user_attr_logits, _ = model(attr_seq, mask_positions)
            # print("masked_item_logits_shape:", masked_item_logits.shape)
            # print("original_seq[mask]_shape:", original_seq[mask].shape)
            item_loss = criterion(masked_item_logits, original_seq[mask])
            user_attr_loss = 0
            for i in range(user_attr.size(1)):
                # print("shape before loss:", user_attr_logits[:, i].shape, user_attr[:, i].shape)
                user_attr_loss += criterion(user_attr_logits[:, i], user_attr[:, i])
            user_attr_loss /= user_attr.size(1)
            # print("user_attr_loss:", user_attr_loss, "item_loss:", item_loss)
            loss = 1.0 * item_loss + 0.3 * user_attr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - epoch_start
        total_loss_all_epochs = total_loss  
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {total_loss:.4f}, Time: {epoch_time:.2f}s")
        
        val_loss = validate(model, val_loader, criterion, device)
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
