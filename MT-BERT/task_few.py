from enum import Enum
import json
import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from torch.utils.data import TensorDataset, Subset
import sys
sys.path.append('/home/zmj/task/LogCL')
from log_dataset import Log_Dataset_v3
from transformers import BertTokenizer


def save_json(data, file):
    dict_json = json.dumps(data,indent=1)
    with open(file,'w+',newline='\n') as file:
        file.write(dict_json)

def read_json(file):
    with open(file,'r+') as file:
        content = file.read()
        content = json.loads(content)
        return content


class Task(Enum):

    Token_C = 'Token_C'

    # anomaly detection
    BGL_C = 'BGL_C'
    Thund_C = 'Thund_C'

    # failure prediction
    BGL_F = 'BGL_F'
    Thund_F = 'Thund_F'

    # dimension classification
    Dimen_C = 'Dimen_C'

    # log to summary
    Log2Sum = 'Log2Sum'

    # code to log
    Java2Log = 'Java2Log'
    Csharp2Log = 'Csharp2Log'


    def num_classes(self):
        if self == Task.Token_C:
            return 9
        elif self == Task.BGL_F:
            return 10
        elif self == Task.Thund_F:
            return 5
        elif self == Task.Dimen_C:
            return 13
        else:
            return 2


class TaskConfig:
    def __init__(self, dataset_loading_args, batch_size, metrics):
        self.dataset_loading_args = dataset_loading_args
        self.batch_size = batch_size
        self.metrics = metrics


def define_dataset_config():
    bz = 45
    datasets_config = {

        Task.Token_C: TaskConfig(("Loglue","Token_C"), batch_size=bz,
                                 metrics=[accuracy_score,recall_score,precision_score,f1_score]),
        Task.BGL_C: TaskConfig(("Loglue", "BGL_C"), batch_size=bz,
                               metrics=[accuracy_score, recall_score, precision_score, f1_score]),
        Task.Thund_C: TaskConfig(("Loglue", "Thund_C"), batch_size=bz,
                                 metrics=[accuracy_score, recall_score, precision_score, f1_score]),
        Task.BGL_F: TaskConfig(("Loglue","BGL_F"),batch_size=bz,
                               metrics=[accuracy_score,recall_score,precision_score,f1_score]),
        Task.Thund_F: TaskConfig(("Loglue", "Thund_F"), batch_size=bz,
                               metrics=[accuracy_score, recall_score, precision_score, f1_score]),
        Task.Dimen_C: TaskConfig(("Loglue", "Dimen_C"), batch_size=bz,
                               metrics=[accuracy_score, recall_score, precision_score, f1_score]),
        Task.Log2Sum: TaskConfig(("Loglue", "Log2Sum"), batch_size=bz,
                               metrics=[accuracy_score, recall_score, precision_score, f1_score]),
        Task.Java2Log: TaskConfig(("Loglue", "Java2Log"), batch_size=bz,
                               metrics=[accuracy_score, recall_score, precision_score, f1_score]),
        Task.Csharp2Log: TaskConfig(("Loglue", "Csharp2Log"), batch_size=bz,
                               metrics=[accuracy_score, recall_score, precision_score, f1_score])


    }
    return datasets_config


def define_tasks_config(datasets_config):
    tasks_config = {}
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained(r'D:\ZMJ\Local-model\bert-base-uncased')
    pre_seq_len = 20
    max_length = 512-pre_seq_len*2
    for task, task_config in datasets_config.items():

        if task == Task.Token_C:
            token_file = '../data_zmj/tokenclf/altokclf/'
            train_data = read_json(token_file + 'train.json')
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')
            label_dict = read_json(token_file + 'label_dict.json')
            n_class = len(label_dict)
            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='num')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='num')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='num')
            len_dataset = len(train_dataset)

        elif task == Task.BGL_C:
            token_file = '../data_zmj/anomadet/bglanoma/'
            train_data = read_json(token_file + 'train.json')
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')

            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='anoma')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='anoma')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='anoma')
            len_dataset = len(train_dataset)

        elif task == Task.Thund_C:
            token_file = '../Thunderbird_ori/thunderbird_AD_200k/'
            train_data = read_json(token_file + 'train.json')#[:100000]
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')

            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='anoma')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='anoma')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='anoma')
            len_dataset = len(train_dataset)

        elif task == Task.BGL_F:
            token_file = '../data_zmj/anomadet/bglanoma_mul/'
            train_data = read_json(token_file + 'train.json')#[:100000]
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')
            label_dict = read_json(token_file + 'label_dict.json')
            n_class = len(label_dict)
            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='anoma')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='anoma')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='anoma')
            len_dataset = len(train_dataset)

        elif task == Task.Thund_F:
            token_file = '../Thunderbird_ori/thunderbird_200k/'
            train_data = read_json(token_file + 'train.json')#[:100000]
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')
            label_dict = read_json(token_file + 'label_dict.json')
            n_class = len(label_dict)
            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='anoma')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='anoma')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='anoma')
            len_dataset = len(train_dataset)


        elif task == Task.Dimen_C:
            token_file = '../data_zmj/dimenclf/aldimclf/'
            train_data = read_json(token_file + 'train.json')
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')
            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='dimension')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='dimension')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='dimension')
            len_dataset = len(train_dataset)

        elif task == Task.Log2Sum:
            token_file = '../data_zmj/log2sumy/allg2smy/'
            train_data = read_json(token_file + 'train.json')
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')
            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='log2sumy')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='log2sumy')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='log2sumy')
            len_dataset = len(train_dataset)

        elif task == Task.Java2Log:
            token_file = '../data_zmj/code2log/java2log/'
            train_data = read_json(token_file + 'train.json')
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')
            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='code2log')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='code2log')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='code2log')
            len_dataset = len(train_dataset)

        elif task == Task.Csharp2Log:
            token_file = '../data_zmj/code2log/csha2log/'
            train_data = read_json(token_file + 'train.json')
            valid_data = read_json(token_file + 'valid.json')
            test_data = read_json(token_file + 'test.json')
            train_dataset = Log_Dataset_v3(tokenizer, train_data, max_length, data_name='code2log')
            val_dataset = Log_Dataset_v3(tokenizer, valid_data, max_length, data_name='code2log')
            test_dataset = Log_Dataset_v3(tokenizer, test_data, max_length, data_name='code2log')
            len_dataset = len(train_dataset)


        shuffle = len(train_dataset) > 0
        train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=task_config.batch_size,
                                                   shuffle=shuffle)
        val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=0, batch_size=task_config.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=task_config.batch_size, shuffle=False)

        tasks_config[task] = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "test_dataset": test_dataset,
            "train_dataset": train_dataset
        }
    return tasks_config
