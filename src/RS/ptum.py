import os
import argparse
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

class PTUMDataset(MyDataset):
    def __init__(self, data_dir, set_type, max_seq_len, mask_prob, task='ctr'):
        super().__init__(data_dir, set=set_type, task=task, max_hist_len=max_seq_len)
        self.mask_prob = mask_prob

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        item_seq = data['hist_iid_seq']
        mask = torch.rand(len(item_seq)) < self.mask_prob
        masked_seq = item_seq.clone()
        masked_seq[mask] = MASK_ITEM_ID

        next_two_items = item_seq[-2:] if len(item_seq) >= 2 else [MASK_ITEM_ID, MASK_ITEM_ID]
        return {
            'masked_seq': masked_seq,
            'original_seq': item_seq,
            'mask': mask,
            'next_two_items': torch.tensor(next_two_items).long()
        }

class PTUMModel(nn.Module):
    def __init__(self, num_items, embed_dim, max_seq_len):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len + 2, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dropout=0.5),
            num_layers=2
        )
        self.item_pred_head = nn.Linear(embed_dim, num_items)
        self.next_item_pred_head = nn.Linear(embed_dim, num_items)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input_ids, mask_positions):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        item_emb = self.item_embedding(input_ids)
        pos_emb = self.pos_embedding(position_ids)
        seq_emb = item_emb + pos_emb
        seq_emb = self.dropout(seq_emb).transpose(0, 1)
        states = self.transformer(seq_emb).transpose(0, 1)

        if mask_positions is not None:
            masked_item_logits = self.item_pred_head(states[mask_positions])
        else:
            masked_item_logits = None

        next_item_logits = self.next_item_pred_head(states[:, -1])
        return masked_item_logits, next_item_logits

def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            masked_seq = batch['masked_seq'].to(device)
            original_seq = batch['original_seq'].to(device)
            next_two_items = batch['next_two_items'].to(device)
            mask = batch['mask'].to(device)
            mask_positions = mask.nonzero(as_tuple=True)

            masked_item_logits, next_item_logits = model(masked_seq, mask_positions)
            item_loss = criterion(masked_item_logits, original_seq[mask])
            next_item_loss = criterion(next_item_logits, next_two_items)
            loss = item_loss + next_item_loss
            total_val_loss += loss.item() * masked_seq.size(0)
            total_samples += masked_seq.size(0)
    return total_val_loss / total_samples

def train(args):
    device = args.device
    setup_seed(args.seed)

    train_dataset = PTUMDataset(args.data_dir, 'train', args.max_seq_len, args.mask_prob)
    val_dataset = PTUMDataset(args.data_dir, 'test', args.max_seq_len, args.mask_prob)
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = PTUMModel(
        num_items=train_dataset.item_num + 1,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            masked_seq = batch['masked_seq'].to(device)
            original_seq = batch['original_seq'].to(device)
            next_two_items = batch['next_two_items'].to(device)
            mask = batch['mask'].to(device)
            mask_positions = mask.nonzero(as_tuple=True)

            masked_item_logits, next_item_logits = model(masked_seq, mask_positions)
            item_loss = criterion(masked_item_logits, original_seq[mask])
            next_item_loss = criterion(next_item_logits, next_two_items)
            loss = item_loss + next_item_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = validate(model, val_loader, criterion, device)
        logging.info(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_save_path = os.path.join(args.save_dir, 'best_ptum_model.pt')
            torch.save(model.state_dict(), best_model_save_path)
            logging.info(f"New best model saved to {best_model_save_path} with Validation Loss: {best_val_loss:.4f}")

    training_time = time.time() - start_time
    logging.info(f"Training finished in {training_time:.2f}s")
    final_model_save_path = os.path.join(args.save_dir, 'ptum_model.pt')
    torch.save(model.state_dict(), final_model_save_path)
    logging.info(f"Final model saved to {final_model_save_path}")

if __name__ == '__main__':
    args = parse_args()
    log_file = setup_logging(args.save_dir)
    logging.info("Starting PTUM training experiment")
    train(args)
    logging.info("PTUM training completed.")
