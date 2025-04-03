import subprocess
import os
import sys

# dataset_name = 'ml-1m'
dataset_name = 'yelp'
# dataset_name = 'ml-25m'

# Training args
data_dir = f'../../data/{dataset_name}/proc_data_6'
task_name = 'ctr'
layer = -1  # which layer of LLM
# aug_prefix = f'embeddings/marc_avg_all_layer{layer}'  # for original LLM representation
aug_prefix = f'embeddings/marc_avg_comp_layer{layer}'  # for compressed LLM representation

print('aug prefix: ', aug_prefix)
# augment = True
augment = False


epoch = 20
batch_size = 256
lr = '5e-4'
lr_sched = 'cosine' #
weight_decay = 0

model = 'DIN'
# model = 'DCNv1'
# model = 'DCNv2'
# model = 'DeepFM'
# model = 'xDeepFM'
# model = 'FiBiNet'
# model = 'FiGNN'
# model = 'AutoInt'
embed_size = 32
final_mlp = '200,80'
num_cross_layers = 3
dropout = 0.0
max_hist_len = 10

convert_type = 'MoE'
convert_arch = '128,32'
convert_dropout = 0.0
expert_num = 2


if dataset_name == 'ml-25m':  # ml-25m is much slower than other dataset
    bss = [256, 512]
    # bss = [128, 1024]
    experts = [2, 3, 4]
    num_worker = 8
else:
    bss = [256, 128, 512, 1024]
    # experts = [1, 2, 3]
    # experts = [4, 5, 6]
    experts = [1, 2, 3, 4, 5, 6]
    num_worker = 4

if dataset_name == 'yelp':
    lrs = ['1e-4', '2e-4', '5e-4', '1e-3']
else:
    lrs = ['1e-4', '5e-4', '1e-3', '2e-3']

# Run the train process
for lr_sched in ['cosine']:
    for batch_size in bss:
        for lr in lrs:
            expert_num = 0
            print('---------------bs, lr, epoch, expert , convert arch, gru----------', batch_size,
                    lr, epoch, expert_num, convert_arch, model)
            subprocess.run(['python3', '-u', 'main_ctr.py',
                            f'--save_dir=./model/{dataset_name}/{task_name}/{model}/WDA_Emb{embed_size}_epoch{epoch}'
                            f'_bs{batch_size}_lr{lr}_{lr_sched}_cnvt_arch_{convert_arch}_cnvt_type_{convert_type}'
                            f'_eprt_{expert_num}_wd{weight_decay}_drop{dropout}' + \
                            f'_hl{final_mlp}_cl{num_cross_layers}_augment_{augment}',
                            f'--data_dir={data_dir}',
                            f'--augment={augment}',
                            f'--aug_prefix={aug_prefix}',
                            f'--task={task_name}',
                            f'--max_hist_len={max_hist_len}',
                            f'--convert_arch={convert_arch}',
                            f'--convert_type={convert_type}',
                            f'--convert_dropout={convert_dropout}',
                            f'--epoch_num={epoch}',
                            f'--num_worker={num_worker}',
                            f'--batch_size={batch_size}',
                            f'--lr={lr}',
                            f'--lr_sched={lr_sched}',
                            f'--weight_decay={weight_decay}',
                            f'--algo={model}',
                            f'--embed_size={embed_size}',
                            f'--expert_num={expert_num}',
                            f'--final_mlp_arch={final_mlp}',
                            f'--dropout={dropout}',
                            ])
