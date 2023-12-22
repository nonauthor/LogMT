import torch
from torch.utils.data import Dataset,DataLoader
import transformers
from transformers import BertTokenizer
transformers.logging.set_verbosity_error()
import json
import numpy as np
import random
random.seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)


def save_json(data, file):
    dict_json = json.dumps(data,indent=1)
    with open(file,'w+',newline='\n') as file:
        file.write(dict_json)

def read_json(file):
    with open(file,'r+') as file:
        content = file.read()
        content = json.loads(content)
        return content




def proc_num(data,split:list=[0.7,0.1,0.2]):
    assert sum(split)==1
    random.shuffle(data)
    datasize = len(data)
    train_data = data[:int(datasize*split[0])]
    valid_data = data[int(datasize*split[0]):int(datasize*split[0])+int(datasize*split[1])]
    test_data = data[int(datasize*split[0])+int(datasize*split[1]):]
    return train_data,valid_data,test_data

def num_label_dict(data):
    label_dict = {}
    id = 0
    for i in data:
        label = i[-1]
        if label in label_dict:
            pass
        else:
            label_dict[label]=id
            id+=1
    return label_dict

def proc_cd2log(data:dict,split:list=[0.6,0.2,0.2],langu='java'):
    if langu == 'all':
        all_data = data['java'] + data['c_sharp']
        random.shuffle(all_data)
        datasize = len(all_data)
        train_data = all_data[:int(datasize * split[0])]
        valid_data = all_data[int(datasize * split[0]):int(datasize * split[0]) + int(datasize * split[1])]
        test_data = all_data[int(datasize * split[0]) + int(datasize * split[1]):]
        return train_data, valid_data, test_data

    elif langu == 'c_sharp':
        c_sharp_data = data['c_sharp']
        random.shuffle(c_sharp_data)
        datasize = len(c_sharp_data)
        train_data = c_sharp_data[:int(datasize * split[0])]
        valid_data = c_sharp_data[int(datasize * split[0]):int(datasize * split[0]) + int(datasize * split[1])]
        test_data = c_sharp_data[int(datasize * split[0]) + int(datasize * split[1]):]
        return train_data, valid_data, test_data

    elif langu == 'java':
        java_data = data['java']
        random.shuffle(java_data)
        datasize = len(java_data)
        train_data = java_data[:int(datasize * split[0])]
        valid_data = java_data[int(datasize * split[0]):int(datasize * split[0]) + int(datasize * split[1])]
        test_data = java_data[int(datasize * split[0]) + int(datasize * split[1]):]
        return train_data, valid_data, test_data
    else:
        raise Exception('langu must in ["all","java","c_sharp"]')


def proc_mulAD(data:list,split:list=[0.6,0.2,0.2]):
    assert sum(split) == 1
    random.shuffle(data)
    datasize = len(data)
    train_data = data[:int(datasize * split[0])]
    valid_data = data[int(datasize * split[0]):int(datasize * split[0]) + int(datasize * split[1])]
    test_data = data[int(datasize * split[0]) + int(datasize * split[1]):]
    return train_data, valid_data, test_data

def mulAD_label_dict(data):
    label_dict = {}
    id = 0
    for i in data:
        label = i[0]
        if label in label_dict:
            pass
        else:
            label_dict[label] = id
            id += 1
    return label_dict


def pro_summ(data,split:list=[0.7,0.1,0.2]):
    all_clean = []
    for sub_data in data:
        for i in data[sub_data]:
            if i[0] == []:
                pass
            else:
                i[0] = list(set(i[0]))
                all_clean.append(i)
    assert sum(split) == 1
    random.shuffle(data)
    datasize = len(data)
    train_data = data[:int(datasize * split[0])]
    valid_data = data[int(datasize * split[0]):int(datasize * split[0]) + int(datasize * split[1])]
    test_data = data[int(datasize * split[0]) + int(datasize * split[1]):]
    return train_data, valid_data, test_data



class LogDataset(Dataset):

    def __init__(self, tokenizer, datas, max_length,data_name='num'):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_name = data_name
        if self.data_name == 'num':
            self.label_dict = num_label_dict(self.datas)
        elif self.data_name == 'code2log':
            pass
        elif self.data_name == 'multiAD':
            self.label_dict = mulAD_label_dict(self.datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.data_name == 'num':
            data_encoding = self.tokenizer(
                self.datas[index][0],self.datas[index][1],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # label_n = self.label_dict[self.datas[index][2]]
            label = torch.LongTensor([self.label_dict[self.datas[index][2]]]).squeeze()
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'code2log':
            pass
        elif self.data_name == 'multiAD':
            data_encoding = self.tokenizer(
                self.datas[index][1],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            label = torch.LongTensor([self.label_dict[self.datas[index][0]]]).squeeze()
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        # data_encoding = self.tokenizer(
        #     self.datas[index] + self.tokenizer.eos_token,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # )
        # return data_encoding["input_ids"][0], data_encoding["attention_mask"][0]

class Log_Dataset_v2(Dataset):

    def __init__(self, tokenizer, datas, max_length,data_name='num'):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_name = data_name


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.data_name == 'num':
            data_encoding = self.tokenizer(
                self.datas[index][0],self.datas[index][1],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # label_n = self.label_dict[self.datas[index][2]]
            label = torch.LongTensor([self.datas[index][3]])
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'code2log':
            data_encoding = self.tokenizer(
                self.datas[index][0], self.datas[index][1],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            label = torch.LongTensor([self.datas[index][2]])
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'anoma':
            data_encoding = self.tokenizer(
                self.datas[index][0],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            label = torch.LongTensor([self.datas[index][1]])
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'log2sumy':
            logs = self.datas[index][0]
            if len(logs)>=10:
                logs = logs[:10]
            logs_clean = [i.strip('\n') for i in logs]
            log = ' '.join(logs_clean)
            assert len(self.datas[index][1]) == 1
            sumy = self.datas[index][1][0].strip('\n')
            data_encoding = self.tokenizer(
                log,sumy,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            label = torch.LongTensor([self.datas[index][2]])
            # print(index)
            # print(sumy)
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'dimension':
            data_encoding = self.tokenizer(
                self.datas[index][0],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
            # aaa = self.tokenizer.mask_token_id
            # mask_ids = torch.nonzero(data_encoding["input_ids"][0]==mask_id).squeeze()
            mask_ids = torch.nonzero(data_encoding["input_ids"][0] == self.tokenizer.mask_token_id).squeeze()
            label = torch.LongTensor([self.datas[index][2]])
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label,mask_ids

class Log_Dataset_v3(Dataset):

    def __init__(self, tokenizer, datas, max_length,data_name='num'):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_name = data_name


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        if self.data_name == 'num':
            data_encoding = self.tokenizer(
                self.datas[index][0],self.datas[index][1],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # label_n = self.label_dict[self.datas[index][2]]
            label = torch.LongTensor([self.datas[index][3]])
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'code2log':
            data_encoding = self.tokenizer(
                self.datas[index][0], self.datas[index][1],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            label = torch.LongTensor([self.datas[index][2]])
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'anoma':
            data_encoding = self.tokenizer(
                self.datas[index][0],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            label = torch.LongTensor([self.datas[index][1]])
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'log2sumy':
            logs = self.datas[index][0]
            if len(logs)>=10:
                logs = logs[:10]
            logs_clean = [i.strip('\n') for i in logs]
            log = ' '.join(logs_clean)
            assert len(self.datas[index][1]) == 1
            sumy = self.datas[index][1][0].strip('\n')
            data_encoding = self.tokenizer(
                log,sumy,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            label = torch.LongTensor([self.datas[index][2]])
            # print(index)
            # print(sumy)
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label
        elif self.data_name == 'dimension':
            data_encoding = self.tokenizer(
                self.datas[index][0],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
            # aaa = self.tokenizer.mask_token_id
            # mask_ids = torch.nonzero(data_encoding["input_ids"][0]==mask_id).squeeze()
            mask_ids = torch.nonzero(data_encoding["input_ids"][0] == self.tokenizer.mask_token_id).squeeze()
            label = torch.LongTensor([self.datas[index][2]])
            return data_encoding["input_ids"][0], data_encoding["attention_mask"][0], label,mask_ids

if __name__ == '__main__':
    num_file = 'data/num.json'
    code2log_file = 'data/code2log.json'
    log2summ_file = 'data/logsummary.json'
    multiAD_file = 'data/multiAD.json'
    lianggang_file= 'data/lianggang.json'
    num_data = read_json(num_file)
    numLableDict = num_label_dict(num_data)
    code2log_data = read_json(code2log_file)
    log2summ_data = read_json(log2summ_file)
    multiAD_data = read_json(multiAD_file)
    lianggang_data = read_json(lianggang_file)
    _ = pro_summ(log2summ_data)
    t,v,s = proc_num(multiAD_data)
    # t,v,s = proc_cd2log(code2log_data,langu='all')
    tokenizer = BertTokenizer.from_pretrained('/home/zmj/localmodel/bert-base-uncased')
    dataset = LogDataset(tokenizer,t,max_length=512,data_name='num')
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )
    for batch in dataloader:
        input_ids = batch[0]
        atten_mask = batch[1]
        labels = batch[2].squeeze()
        print(input_ids.shape)