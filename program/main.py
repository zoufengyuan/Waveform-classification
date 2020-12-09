# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:14:12 2020

@author: FengY Z
"""

import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from torch import nn, optim
from adamw import AdamW
from torch.utils.data import DataLoader
from dataset import ECGDataset,transform
from config import config
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)
    
def train_epoch(model, optimizer, criterion, train_dataloader,
                show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output,feature = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1,acc = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d\tloss:%.4f\tf1:%.3f\tacc:%.3f" % (it_count, loss.item(), f1,acc))
    return loss_meter / it_count, f1_meter / it_count
def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count,acc_meter = 0, 0, 0,0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            target = target.view(-1,1)
            inputs = inputs.to(device)
            target = target.to(device)
            output,feature = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1,acc = utils.calc_f1(target, output, threshold)
            f1_meter += f1
            acc_meter+=acc
    return loss_meter / it_count, f1_meter / it_count,acc_meter/it_count
def my_collate_fn(batch,nonsample_n=800):
    data, label = zip(*batch)
    new_data = []
    new_label = []
    batch_size = len(label)
    len_ = int(nonsample_n + np.random.rand() * (config.target_point_num - nonsample_n))
    for i in range(batch_size):
        start = int(np.random.rand() * (config.target_point_num - len_))
        tmp_data = data[i].transpose(0, 2)
        if i == 0:
            new_data = (tmp_data[start:(start + len_)].transpose(0, 2))
            new_label = label[i].view(1,1)
        else:
            new_data = torch.cat((
                new_data, (tmp_data[start:(start + len_)].transpose(0, 2))
            ), 0)
            new_label = torch.cat((new_label, label[i].view(1,1)), 0)
    return new_data.reshape((batch_size, 1, 1, -1)), new_label.reshape(
        (batch_size, -1))
    
def train(args):
    model = models.myecgnet()
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cuda')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    train_dataset = ECGDataset(config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=my_collate_fn,#再对每一个bach进行抽样
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=0)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                num_workers=0)
    print("train_datasize", len(train_dataset), "val_datasize",
          len(val_dataset))
    optimizer = AdamW(model.parameters(), lr=config.lr)
    criterion = utils.Loss_cal()
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name,
                                   time.strftime("%Y%m%d%H%M"))
    print(model_save_dir)
    os.mkdir(model_save_dir)
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    if args.resume:
        if os.path.exists(args.ckpt):
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion,
                                           train_dataloader,
                                           show_interval=10)
        val_loss, val_f1,val_acc = val_epoch(model, criterion, val_dataloader)
        print(
            '#epoch:%03d\tstage:%d\ttrain_loss:%.4f\ttrain_f1:%.3f\tval_loss:%0.4f\tval_f1:%.3f\tval_acc:%.3f\ttime:%s\n'
            % (epoch, stage, train_loss, train_f1, val_loss, val_f1,val_acc,
               utils.print_time_cost(since)))
        state = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "loss": val_loss,
            'f1': val_f1,
            'lr': lr,
            'stage': stage
        }
        save_ckpt(state, best_f1 < val_f1, model_save_dir)
        best_f1 = max(best_f1, val_f1)
        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

def val(args):
    list_threhold = [0.5]
    model = models.myecgnet()
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt,
                                         map_location='cpu')['state_dict'])
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                num_workers=8)
    for threshold in list_threhold:
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader,
                                     threshold)
        print('threshold %.2f\tval_loss:%0.3e\tval_f1:%.3f\n' %
              (threshold, val_loss, val_f1))

def test(args):
    utils.mkdirs(config.sub_dir)
    model = models.myecgnet()
    model.load_state_dict(torch.load(args.ckpt,
                                     map_location='cuda')['state_dict'])
    model = model.to(device)
    model.eval()
    sub_file = 'result.txt'
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        source_df = pd.read_csv(config.source_data_last)
        df = source_df[config.wave_features].values
        id_list = source_df['id'].to_list()
        x = transform(df).unsqueeze(0).to(device)
        output = torch.sigmoid(model(x)).squeeze().cpu().numpy()
        predict_output = [1 if x>0.5 else 0 for x in output]
        for i in range(len(id_list)):
            fout.write(str(id_list[0]))
            fout.write('\t')
            fout.write(str(predict_output[i]))
    fout.close()

def toplayer(args):
    utils.mkdirs(config.sub_dir)
    model = models.myecgnet()
    model.load_state_dict(torch.load(args.ckpt,
                                     map_location='cuda')['state_dict'])
    model = model.to(device)
    model.eval()
    sub_file = '%s.txt' % args.ex
    fout = open(sub_file, 'w', encoding='utf-8')
    with torch.no_grad():
        source_df = pd.read_csv(config.source_data_last)
        df = source_df[config.wave_features].values
        id_list = source_df['id'].to_list()
        for i in tqdm(range(len(id_list))):
            x = transform(df[i]).view(1,1,1000).unsqueeze(0).to(device)
            output = torch.sigmoid(model(x)).squeeze().cpu().detach().numpy()
            fout.write(str(id_list[i]))
            fout.write('\t')
            fout.write(str(float(output)))
            fout.write('\n')
    fout.close()
 
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt",
                        type=str,
                        help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    if (args.command == "train"):
        train(args)
    if (args.command == "test"):
        test(args)
    if (args.command == "toplayer"):
        toplayer(args)











































