import subprocess

# hyperparameters
batch_sizes = [768, 512, 1024]
learning_rates = [1e-4, 2e-4, 5e-4, 1e-3]

# data directory and other parameters
data_dir = '../../data/yelp/proc_data_6'
epochs = 20
embed_dim = 32
max_seq_len = 10
mask_prob = 0.20
save_dir = './pretrain_model_final'

for batch_size in batch_sizes:
    for lr in learning_rates:
        print(f"Running UPRec experiment with batch_size={batch_size}, lr={lr}")
        subprocess.run([
            'python3', '-u', 'pretrain.py',
            f'--data_dir={data_dir}',
            f'--batch_size={batch_size}',
            f'--lr={lr}',
            f'--epochs={epochs}',
            f'--embed_dim={embed_dim}',
            f'--max_seq_len={max_seq_len}',
            f'--mask_prob={mask_prob}',
            f'--save_dir={save_dir}/bs{batch_size}_lr{lr}'
        ])
