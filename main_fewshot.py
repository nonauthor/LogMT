import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import transformers
from transformers import BertTokenizer,BertModel,get_scheduler
from log_dataset import LogDataset, proc_num, proc_cd2log, Log_Dataset_v2
from torch.optim import AdamW
import tqdm
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
transformers.logging.set_verbosity_error()

def save_json(data, file):
    dict_json = json.dumps(data,indent=1)
    with open(file,'w+',newline='\n') as file:
        file.write(dict_json)

def read_json(file):
    with open(file,'r+') as file:
        content = file.read()
        content = json.loads(content)
        return content

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0'
random.seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)

def devForTwoClass_OnePrompt(model, dev_loader, device):
    pred_all = []
    label_all = []
    for batch in dev_loader:
        dev_inputid = batch[0].to(device)
        dev_attenmask = batch[1].to(device)
        dev_labels = batch[2].squeeze(1).detach().cpu().numpy()
        label_all.extend(dev_labels.tolist())
        with torch.no_grad():
            out_put = model(dev_inputid, dev_attenmask)
            predict_y = out_put.softmax(dim=-1)
            predict_y_ = predict_y.argmax(dim=1).detach().cpu().numpy()
            pred_all.extend(predict_y_)

    acc = accuracy_score(label_all, pred_all)
    recall = recall_score(label_all, pred_all)
    p = precision_score(label_all, pred_all)
    f1 = f1_score(label_all, pred_all)

    return acc, recall, p, f1

def devForMask_OnePrompt(model, dev_loader, device):
    pred_all = []
    label_all = []
    for batch in dev_loader:
        dev_inputid = batch[0].to(device)
        dev_attenmask = batch[1].to(device)
        dev_mask_ids = batch[3].squeeze().to(device)
        batch_ids = torch.arange(dev_mask_ids.shape[0]).to(device)
        dev_labels = batch[2].squeeze().detach().cpu().numpy()
        label_all.extend(dev_labels.tolist())
        with torch.no_grad():
            out_put= model(dev_inputid, dev_attenmask,dev_mask_ids)
            predict_y = out_put.softmax(dim=-1)
            predict_y_ = predict_y.argmax(dim=1).detach().cpu().numpy()
            pred_all.extend(predict_y_)


    acc = accuracy_score(label_all,pred_all)
    recall = recall_score(label_all,pred_all,average='macro')
    p = precision_score(label_all,pred_all,average='macro')
    f1 = f1_score(label_all,pred_all,average='macro')
    # auc_ = roc_auc_score(dev_labels,predict_y.detach().cpu().numpy(),labels=[0,1,2,3,4,5,6,7,8],multi_class='ovo')

    return acc,recall,p,f1

def dev_OnePrompt(model, dev_loader, device):
    pred_all = []
    label_all = []
    for batch in dev_loader:
        dev_inputid = batch[0].to(device)
        dev_attenmask = batch[1].to(device)
        dev_labels = batch[2].squeeze(1).detach().cpu().numpy()
        label_all.extend(dev_labels.tolist())
        with torch.no_grad():
            out_put= model(dev_inputid, dev_attenmask)
            predict_y = out_put.softmax(dim=-1)
            predict_y_ = predict_y.argmax(dim=1).detach().cpu().numpy()
            pred_all.extend(predict_y_)


    acc = accuracy_score(label_all,pred_all)
    recall = recall_score(label_all,pred_all,average='macro')
    p = precision_score(label_all,pred_all,average='macro')
    f1 = f1_score(label_all,pred_all,average='macro')
    # auc_ = roc_auc_score(dev_labels,predict_y.detach().cpu().numpy(),labels=[0,1,2,3,4,5,6,7,8],multi_class='ovo')

    return acc,recall,p,f1


def devForTwoClass(model, dev_loader, device):
    pred_all = []
    label_all = []
    for batch in dev_loader:
        dev_inputid = batch[0].to(device)
        dev_attenmask = batch[1].to(device)
        dev_labels = batch[2].squeeze(1).detach().cpu().numpy()
        label_all.extend(dev_labels.tolist())
        with torch.no_grad():
            out_put = model(dev_inputid, dev_attenmask)
            predict_y = out_put.softmax(dim=-1)
            predict_y_ = predict_y.argmax(dim=1).detach().cpu().numpy()
            pred_all.extend(predict_y_)

    acc = accuracy_score(label_all, pred_all)
    recall = recall_score(label_all, pred_all)
    p = precision_score(label_all, pred_all)
    f1 = f1_score(label_all, pred_all)

    return acc, recall, p, f1

def devForMask(model, dev_loader, device):
    pred_all = []
    label_all = []
    for batch in dev_loader:
        dev_inputid = batch[0].to(device)
        dev_attenmask = batch[1].to(device)
        dev_mask_ids = batch[3].squeeze().to(device)
        batch_ids = torch.arange(dev_mask_ids.shape[0]).to(device)
        dev_labels = batch[2].squeeze().detach().cpu().numpy()
        label_all.extend(dev_labels.tolist())
        with torch.no_grad():
            out_put,_= model(dev_inputid, dev_attenmask,dev_mask_ids)
            predict_y = out_put.softmax(dim=-1)
            predict_y_ = predict_y.argmax(dim=1).detach().cpu().numpy()
            pred_all.extend(predict_y_)


    acc = accuracy_score(label_all,pred_all)
    recall = recall_score(label_all,pred_all,average='macro')
    p = precision_score(label_all,pred_all,average='macro')
    f1 = f1_score(label_all,pred_all,average='macro')
    # auc_ = roc_auc_score(dev_labels,predict_y.detach().cpu().numpy(),labels=[0,1,2,3,4,5,6,7,8],multi_class='ovo')

    return acc,recall,p,f1

def dev(model, dev_loader, device):
    pred_all = []
    label_all = []
    for batch in dev_loader:
        dev_inputid = batch[0].to(device)
        dev_attenmask = batch[1].to(device)
        dev_labels = batch[2].squeeze(1).detach().cpu().numpy()
        label_all.extend(dev_labels.tolist())
        with torch.no_grad():
            out_put,_= model(dev_inputid, dev_attenmask)
            predict_y = out_put.softmax(dim=-1)
            predict_y_ = predict_y.argmax(dim=1).detach().cpu().numpy()
            pred_all.extend(predict_y_)


    acc = accuracy_score(label_all,pred_all)
    recall = recall_score(label_all,pred_all,average='macro')
    p = precision_score(label_all,pred_all,average='macro')
    f1 = f1_score(label_all,pred_all,average='macro')
    # auc_ = roc_auc_score(dev_labels,predict_y.detach().cpu().numpy(),labels=[0,1,2,3,4,5,6,7,8],multi_class='ovo')

    return acc,recall,p,f1





from models.MLogtransfer_model_fewshot import ThreeExpertsModel,ThreeExpertsModel_ms,ThreeExpertsModel_flexible,\
    ThreeExpertsModel_flexible_WoTarget

def load_prompt_v2(fold_path,model):
    model_state = model.state_dict()
    for i in os.listdir(fold_path):
        # if i == 'epoch_29_prefix_encoder_ThunC_embedding_weights_params.pt':
        #     weigths = torch.load(fold_path+i)
        #     model_state.update({'prefix_encoder_source_1.embedding.weight':weigths})
        # if i == 'epoch_29_prefix_encoder_BGLC_embedding_weights_params.pt':
        #     weigths = torch.load(fold_path+i)
        #     model_state.update({'prefix_encoder_source_1.embedding.weight':weigths})
        # if i == 'epoch_29_prefix_encoder_DimC_embedding_weights_params.pt':
        #     weigths = torch.load(fold_path+i)
        #     model_state.update({'prefix_encoder_source_3.embedding.weight':weigths})
        # if i == 'epoch_29_prefix_encoder_Log2Sum_embedding_weights_params.pt':
        #     weigths = torch.load(fold_path+i)
        #     model_state.update({'prefix_encoder_source_4.embedding.weight':weigths})
        if i == 'epoch_29_prefix_encoder_Csharp2Log_embedding_weights_params.pt':
            weigths = torch.load(fold_path+i)
            model_state.update({'prefix_encoder_source_1.embedding.weight':weigths})
        # if i == 'epoch_29_prefix_encoder_ThunF_embedding_weights_params.pt':
        #     weigths = torch.load(fold_path+i)
        #     model_state.update({'prefix_encoder_source_5.embedding.weight':weigths})
        # if i == 'epoch_29_prefix_encoder_TokenC_embedding_weights_params.pt':
        #     weigths = torch.load(fold_path+i)
        #     model_state.update({'prefix_encoder_source_6.embedding.weight':weigths})
        if i == 'epoch_29_prefix_encoder_Java2Log_embedding_weights_params.pt':
            weigths = torch.load(fold_path+i)
            model_state.update({'prefix_encoder_source_2.embedding.weight':weigths})
    model.load_state_dict(model_state)
    return model

def load_prompt_v2_fast(fold_path,model,epoch,task):
    prompts = [
        f'epoch_{epoch}_prefix_encoder_BGLF_embedding_weights_params.pt',
        f'epoch_{epoch}_prefix_encoder_ThunC_embedding_weights_params.pt',
        f'epoch_{epoch}_prefix_encoder_BGLC_embedding_weights_params.pt',
        f'epoch_{epoch}_prefix_encoder_DimC_embedding_weights_params.pt',
        f'epoch_{epoch}_prefix_encoder_Log2Sum_embedding_weights_params.pt',
        f'epoch_{epoch}_prefix_encoder_Csharp2Log_embedding_weights_params.pt',
        f'epoch_{epoch}_prefix_encoder_ThunF_embedding_weights_params.pt',
        f'epoch_{epoch}_prefix_encoder_TokenC_embedding_weights_params.pt',
        f'epoch_{epoch}_prefix_encoder_Java2Log_embedding_weights_params.pt',
    ]
    if task=='BGL_C':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_BGLC_embedding_weights_params.pt')
    if task=='BGL_F':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_BGLF_embedding_weights_params.pt')
    if task=='Thund_C':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_ThunC_embedding_weights_params.pt')
    if task=='ThundF':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_ThunF_embedding_weights_params.pt')
    if task=='Log2Sum':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_Log2Sum_embedding_weights_params.pt')
    if task=='Token_C':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_TokenC_embedding_weights_params.pt')
    if task=='Dimen_C':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_DimC_embedding_weights_params.pt')
    if task=='Java2Log':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_Java2Log_embedding_weights_params.pt')
    if task=='Csharp2Log':
        prompts.remove(f'epoch_{epoch}_prefix_encoder_Csharp2Log_embedding_weights_params.pt')

    model_state = model.state_dict()
    for i,p in enumerate(prompts):
        weights = torch.load(fold_path+p)
        model_state.update({f'prefix_encoder_source_{i+1}.embedding.weight': weights})

    model.load_state_dict(model_state)
    return model


def run_BGLF_mulpt_v2_fewshot(task_name):

    token_file = '../data_zmj/anomadet/bglanoma_mul/'
    train_data = read_json('../data_FewShot/failure/BGLFI/train_4.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')
    label_dict = read_json(token_file + 'label_dict.json')
    n_class = len(label_dict)

    batch_size = 20
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 492, data_name='anoma')
    val_dataset = Log_Dataset_v2(tokenizer, valid_data, 492, data_name='anoma')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 492, data_name='anoma')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=100,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=100,
                                              shuffle=False)

    # model = ThreeExpertsModel(n_class, model_path, pre_seq_len=20)
    model = ThreeExpertsModel_flexible_WoTarget(n_class, model_path, pre_seq_len=20,num_prefix=3)
    # print(model)
    # source_prompt_fold = 'D:\ZMJ\pythonProject\LogCL\MT-BERT\save_models\\'
    source_prompt_fold = "MT-BERT/save_models_fewshot_/"
    model = load_prompt_v2_fast(source_prompt_fold, model,30,task_name)

    alpha = 0.0001
    max_epoch = 100
    num_train_step = max_epoch * len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            output,moe_loss = model(input_ids, atten_mask)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # process_bar.updata(1)
            # print('epoch = ', epoch, '   step = ', step, '   loss = ', loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))
        # 评估
        if epoch >= 5:
            model.eval()
            with torch.no_grad():
                acc, rec, p, f1 = dev(model, val_loader, device)
                print(acc)
                print(rec)
                print(p)
                print(f1)
            model.train()

    # test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = dev(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)

def run_BGLC_mulpt_v2_fewshot(task_name):
    token_file = '../data_zmj/anomadet/bglanoma/'
    train_data = read_json('../data_FewShot/anomadet/BGLAD/train_64sample.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')
    n_class = 2

    batch_size = 20
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 492, data_name='anoma')
    val_dataset = Log_Dataset_v2(tokenizer, valid_data, 492, data_name='anoma')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 492, data_name='anoma')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=100,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=100,
                                              shuffle=False)

    model = ThreeExpertsModel(n_class, model_path, pre_seq_len=20)
    # source_prompt_fold = 'D:\ZMJ\pythonProject\LogCL\MT-BERT\save_models\\'
    source_prompt_fold = "/home/zmj/task/LogCL/MT-BERT/save_models_fewshot/"
    model = load_prompt_v2_fast(source_prompt_fold, model,30,task_name)

    alpha = 0.0001
    max_epoch = 100
    num_train_step = max_epoch * len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            output,moe_loss = model(input_ids, atten_mask)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # process_bar.updata(1)
            # print('epoch = ', epoch, '   step = ', step, '   loss = ', loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))
        # 评估
        if epoch >= 5:
            model.eval()
            with torch.no_grad():
                acc, rec, p, f1 = dev(model, val_loader, device)
                print(acc)
                print(rec)
                print(p)
                print(f1)
            model.train()

    # test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = dev(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)

def run_tokens_mulpt_v2_fewshot(task_name):
    token_file = '../data_FewShot/tokenclf/altokclf/'
    train_data = read_json(token_file + 'train_8.json')
    valid_data = read_json(token_file + 'valid_4.json')
    test_data = read_json(token_file + 'test.json')
    label_dict = read_json(token_file + 'label_dict.json')
    n_class = len(label_dict)

    batch_size = 30
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    # model_path = r'D:\ZMJ\Local-model\bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 492, data_name='num')
    val_dataset = Log_Dataset_v2(tokenizer, valid_data, 492, data_name='num')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 492, data_name='num')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size,
                                              shuffle=False)

    # model = ThreeExpertsModel(n_class, model_path, pre_seq_len=20)
    model = ThreeExpertsModel_flexible(n_class, model_path, pre_seq_len=20,num_prefix=8)
    # source_prompt_fold = 'D:\ZMJ\pythonProject\LogCL\MT-BERT\save_models\\'
    source_prompt_fold = "/home/zmj/task/LogCL/MT-BERT/save_models_fewshot/"
    model = load_prompt_v2_fast(source_prompt_fold, model, 30, task_name)


    alpha = 0.0001
    max_epoch = 100
    num_train_step = max_epoch * len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            output, moe_loss = model(input_ids, atten_mask)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # process_bar.updata(1)
            # print('epoch = ', epoch, '   step = ', step, '   loss = ', loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))
        # 评估
        model.eval()
        with torch.no_grad():
            acc, rec, p, f1 = dev(model, val_loader, device)
            print(acc)
            print(rec)
            print(p)
            print(f1)
        model.train()

    # test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = dev(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)

def run_dimen_mulpt_v2_fewshot(task_name):
    token_file = '../data_zmj/dimenclf/aldimclf/'
    train_data = read_json('/home/zmj/task/LogCL/data_FewShot/dimenclf/aldimclf/' + 'train_4.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')
    label_dict = read_json(token_file + 'label_dict.json')
    n_class = len(label_dict)

    batch_size = 30
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 492, data_name='dimension')
    val_dataset = Log_Dataset_v2(tokenizer, valid_data, 492, data_name='dimension')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 492, data_name='dimension')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size,
                                              shuffle=False)

    model = ThreeExpertsModel_flexible_WoTarget(n_class, model_path, pre_seq_len=20,num_prefix=2,ms=True)
    # source_prompt_fold = 'D:\ZMJ\pythonProject\LogCL\MT-BERT\save_models\\'
    source_prompt_fold = "/home/zmj/task/LogCL/MT-BERT/save_models_fewshot/"
    model = load_prompt_v2_fast(source_prompt_fold, model, 30, task_name)

    loss_fn = nn.CrossEntropyLoss()
    alpha = 0.0001
    max_epoch = 100
    num_train_step = max_epoch*len(train_loader)

    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            mask_ids = batch[3].squeeze().to(device)
            output, moe_loss = model(input_ids, atten_mask,mask_ids)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # process_bar.updata(1)
            # print('epoch = ',epoch,'   step = ',step,'   loss = ',loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))

        #评估
        model.eval()
        with torch.no_grad():
            acc, rec, p, f1 = devForMask(model, val_loader, device)
            print(acc)
            print(rec)
            print(p)
            print(f1)
        model.train()

    #test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = devForMask(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)

def run_summy_mulpt_v2_fewshot(task_name):
    token_file = '../data_zmj/log2sumy/allg2smy/'
    train_data = read_json('../data_FewShot/log2sumy/allg2smy/train_32sample.json' )
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')
    n_class = 2

    batch_size = 40
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 472, data_name='log2sumy')
    val_dataset = Log_Dataset_v2(tokenizer, test_data, 472, data_name='log2sumy')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 472, data_name='log2sumy')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size,
                                              shuffle=False)

    model = ThreeExpertsModel(n_class, model_path, pre_seq_len=20)
    source_prompt_fold = "/home/zmj/task/LogCL/MT-BERT/save_models_fewshot/"
    model = load_prompt_v2_fast(source_prompt_fold, model, 30, task_name)

    alpha = 0.001
    max_epoch = 100
    num_train_step = max_epoch * len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            output, moe_loss = model(input_ids, atten_mask)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # process_bar.updata(1)
            # print('epoch = ', epoch, '   step = ', step, '   loss = ', loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))
        # 评估
        model.eval()
        with torch.no_grad():
            acc, rec, p, f1 = dev(model, val_loader, device)
            print(acc)
            print(rec)
            print(p)
            print(f1)
        model.train()

    # test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = dev(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)

def run_csharp_mulpt_v2_fewshot(task_name):
    token_file = '../data_zmj/code2log/csha2log/'
    train_data = read_json('../data_FewShot/code2log/csha2log/' + 'train_32Sample.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')

    n_class = 2

    batch_size = 40
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 472, data_name='code2log')
    val_dataset = Log_Dataset_v2(tokenizer, test_data, 472, data_name='code2log')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 472, data_name='code2log')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=len_dataset,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size,
                                              shuffle=False)

    model = ThreeExpertsModel_flexible(n_class, model_path, pre_seq_len=20,num_prefix=7)
    # source_prompt_fold = 'D:\ZMJ\pythonProject\LogCL\MT-BERT\save_models\\'
    source_prompt_fold = "/home/zmj/task/LogCL/MT-BERT/save_models_fewshot/"
    model = load_prompt_v2_fast(source_prompt_fold, model, 30, task_name)
    alpha = 0.0001
    max_epoch = 100
    num_train_step = max_epoch * len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            output, moe_loss = model(input_ids, atten_mask)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # process_bar.updata(1)
            # print('epoch = ', epoch, '   step = ', step, '   loss = ', loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))
        # 评估
        if epoch >= 5:
            model.eval()
            with torch.no_grad():
                acc, rec, p, f1 = dev(model, val_loader, device)
                print(acc)
                print(rec)
                print(p)
                print(f1)
            model.train()

    # test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = dev(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)

def run_java2log_mulpt_v2_fewshot(task_name):
    token_file = '../data_zmj/code2log/java2log/'
    train_data = read_json('../data_FewShot/code2log/java2log/' + 'train_32Sample.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')

    n_class = 2

    batch_size = 40
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 472, data_name='code2log')
    val_dataset = Log_Dataset_v2(tokenizer, test_data, 472, data_name='code2log')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 472, data_name='code2log')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size,
                                              shuffle=False)

    model = ThreeExpertsModel_flexible(n_class, model_path, pre_seq_len=20,num_prefix=1)
    # source_prompt_fold = 'D:\ZMJ\pythonProject\LogCL\MT-BERT\save_models\\'
    source_prompt_fold = "/home/zmj/task/LogCL/MT-BERT/save_models_fewshot/"
    model = load_prompt_v2_fast(source_prompt_fold, model, 30, task_name)
    alpha = 0.0001
    max_epoch = 100
    num_train_step = max_epoch * len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            output, moe_loss = model(input_ids, atten_mask)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # process_bar.updata(1)
            # print('epoch = ', epoch, '   step = ', step, '   loss = ', loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))
        # 评估
        model.eval()
        with torch.no_grad():
            acc, rec, p, f1 = dev(model, val_loader, device)
            print(acc)
            print(rec)
            print(p)
            print(f1)
        model.train()

    # test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = dev(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)

def run_thunderFI_mulpt_v2_fewshot(task_name):
    token_file = '../Thunderbird_ori/thunderbird_200k/'
    train_data = read_json('../data_FewShot/failure/thunderFI/train_4.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')
    label_dict = read_json(token_file + 'label_dict.json')
    n_class = len(label_dict)

    batch_size = len(train_data)
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 492, data_name='anoma')
    val_dataset = Log_Dataset_v2(tokenizer, valid_data, 492, data_name='anoma')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 492, data_name='anoma')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=100,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=100,
                                              shuffle=False)

    model = ThreeExpertsModel_flexible_WoTarget(n_class, model_path, pre_seq_len=20,num_prefix=3)
    # source_prompt_fold = 'D:\ZMJ\pythonProject\LogCL\MT-BERT\save_models\\'
    source_prompt_fold = "/home/zmj/task/LogCL/MT-BERT/save_models_fewshot/"
    model = load_prompt_v2_fast(source_prompt_fold, model, 30, task_name)

    alpha = 0.0001
    max_epoch = 100
    num_train_step = max_epoch * len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            output, moe_loss = model(input_ids, atten_mask)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # process_bar.updata(1)
            # print('epoch = ', epoch, '   step = ', step, '   loss = ', loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))
        # 评估
        if epoch >= 5:
            model.eval()
            with torch.no_grad():
                acc, rec, p, f1 = dev(model, val_loader, device)
                print(acc)
                print(rec)
                print(p)
                print(f1)
            model.train()

    # test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = dev(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)

def run_thunderAD_mulpt_v2_fewshot(task_name):
    token_file = '../Thunderbird_ori/thunderbird_AD_200k/'
    train_data = read_json('../data_FewShot/anomadet/thunderAD/train_64sample.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')
    n_class = 2

    batch_size = 20
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    train_dataset = Log_Dataset_v2(tokenizer, train_data, 492, data_name='anoma')
    val_dataset = Log_Dataset_v2(tokenizer, valid_data, 492, data_name='anoma')
    test_dataset = Log_Dataset_v2(tokenizer, test_data, 492, data_name='anoma')
    len_dataset = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                             shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size,
                                              shuffle=False)

    model = ThreeExpertsModel_flexible(n_class, model_path, pre_seq_len=20,num_prefix=8)
    # source_prompt_fold = 'D:\ZMJ\pythonProject\LogCL\MT-BERT\save_models\\'
    source_prompt_fold = "/home/zmj/task/LogCL/MT-BERT/save_models_fewshot/"
    model = load_prompt_v2_fast(source_prompt_fold, model, 30, task_name)

    alpha = 0.0001
    max_epoch = 100
    num_train_step = max_epoch * len(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)
    loss_fn.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-2)
    lr_scheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=num_train_step//10,
                                 num_training_steps=num_train_step)
    model.train()

    # process_bar = tqdm(range(num_train_step))
    for epoch in range(max_epoch):
        losses = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch[0].to(device)
            atten_mask = batch[1].to(device)
            labels = batch[2].squeeze(1).to(device)
            output,moe_loss = model(input_ids, atten_mask)
            loss_cls = loss_fn(output, labels)
            loss = (1 - alpha) * loss_cls + alpha * moe_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # process_bar.updata(1)
            # print('epoch = ', epoch, '   step = ', step, '   loss = ', loss)
            losses.append(loss)
        print('epoch = ', epoch, '   loss = ', sum(losses) / len(losses))
        # 评估
        if epoch >= 5:
            model.eval()
            with torch.no_grad():
                acc, rec, p, f1 = dev(model, val_loader, device)
                print(acc)
                print(rec)
                print(p)
                print(f1)
            model.train()

    # test
    model.eval()
    with torch.no_grad():
        acc, rec, p, f1 = dev(model, test_loader, device)
        print(acc)
        print(rec)
        print(p)
        print(f1)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="token")
    args = parser.parse_args()

    if args.task == 'Token_C':
        run_tokens_mulpt_v2_fewshot(args.task)
    elif args.task == 'BGL_C':
        run_BGLC_mulpt_v2_fewshot(args.task)
    elif args.task == 'Thund_C':
        run_thunderAD_mulpt_v2_fewshot(args.task)
    elif args.task == 'BGL_F':
        run_BGLF_mulpt_v2_fewshot(args.task)
    elif args.task == 'Thund_F':
        run_thunderFI_mulpt_v2_fewshot(args.task)
    elif args.task == 'Dimen_C':
        run_dimen_mulpt_v2_fewshot(args.task)
    elif args.task == 'Log2Sum':
        run_summy_mulpt_v2_fewshot(args.task)
    elif args.task == 'Java2Log':
        run_java2log_mulpt_v2_fewshot(args.task)
    elif args.task == 'Csharp2Log':
        run_csharp_mulpt_v2_fewshot(args.task)
    else:
        print('wrong command')
