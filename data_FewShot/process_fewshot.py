import json
import random

def save_json(data, file):
    dict_json = json.dumps(data,indent=1)
    with open(file,'w+',newline='\n') as file:
        file.write(dict_json)

def read_json(file):
    with open(file,'r+') as file:
        content = file.read()
        content = json.loads(content)
        return content

def select_Kshot_MulClass(data,K:int=4,label_dict=None):
    if label_dict==None:
        label_set = [0,1]
    else:
        label_set = list(range(len(label_dict)))
    random.shuffle(data)
    new_train_data = []
    new_valid_data = []
    train_data_split = {}
    for i in data:
        label = i[-1]
        if label in train_data_split:
            train_data_split[label].append(i)
        else:
            train_data_split[label] = [i]
    for key in train_data_split:
        print(key,': ',len(train_data_split[key]))
    for key in train_data_split:
        sample_K = train_data_split[key][:K]
        # sample_K_v = train_data_split[key][K:2*K]
        new_train_data.extend(sample_K)
        # new_valid_data.extend(sample_K_v)
    return new_train_data,new_valid_data

def select_kshot_TwoClass(data,K=16):
    label_set = [0,1]
    random.shuffle(data)
    new_train_data = []
    new_valid_data = []
    train_data_split = {}
    for i in data:
        label = i[1]
        if label in train_data_split:
            train_data_split[label].append(i)
        else:
            train_data_split[label] = [i]
    # for key in train_data_split:
    #     print(key,': ',len(train_data_split[key]))
    for key in train_data_split:
        sample_K = train_data_split[key][:K]
        sample_K_v = train_data_split[key][K:2*K]
        new_train_data.extend(sample_K)
        new_valid_data.extend(sample_K_v)
    return new_train_data,new_valid_data

def select_kshot_TwoClass_match(data,K=16):
    label_set = [0,1]
    random.shuffle(data)
    new_train_data = []
    new_valid_data = []
    train_data_split = {}
    for i in data:
        label = i[2]
        if label in train_data_split:
            train_data_split[label].append(i)
        else:
            train_data_split[label] = [i]
    for key in train_data_split:
        print(key,': ',len(train_data_split[key]))
    for key in train_data_split:
        sample_K_t = train_data_split[key][:K]
        sample_K_v = train_data_split[key][K:2*K]
        new_train_data.extend(sample_K_t)
        new_valid_data.extend(sample_K_v)
    return new_train_data,new_valid_data

def select_thunder(data,K=16):
    label_set = [0,1]
    random.shuffle(data)
    new_train_data = []
    new_valid_data = []
    train_data_split = {0:[],1:[]}
    for i in data:
        label = i[-1]
        if label == 0:
            train_data_split[0].append(i)
        else:
            train_data_split[1].append([i[0],1])
    for key in train_data_split:
        print(key,': ',len(train_data_split[key]))
    for key in train_data_split:
        sample_K = train_data_split[key][:K]
        sample_K_v = train_data_split[key][K:2*K]
        new_train_data.extend(sample_K)
        new_valid_data.extend(sample_K_v)
    return new_train_data,new_valid_data

def token_cls():
    token_file = '../data_zmj/tokenclf/altokclf/'
    train = read_json(token_file + 'train.json')
    valid = read_json(token_file + 'valid.json')
    test = read_json(token_file + 'test.json')
    label_dict = read_json(token_file + 'label_dict.json')
    new_train, new_valid = select_Kshot_MulClass(train, K=6, label_dict=label_dict)
    save_json(new_train, '/home/zmj/task/LogCL/data_FewShot/tokenclf/altokclf/train_6.json')
    save_json(new_valid, '/home/zmj/task/LogCL/data_FewShot/tokenclf/altokclf/valid_6.json')

def dimen_cls():
    dimen_file = '../data_zmj/dimenclf/aldimclf/'
    train = read_json(dimen_file + 'train.json')
    valid = read_json(dimen_file + 'valid.json')
    test = read_json(dimen_file + 'test.json')
    label_dict = read_json(dimen_file + 'label_dict.json')
    new_train, new_valid = select_Kshot_MulClass(train, K=6, label_dict=label_dict)
    save_json(new_train,'/home/zmj/task/LogCL/data_FewShot/dimenclf/aldimclf/train_6.json')


def csharp2log():
    csharp_file = '../data_zmj/code2log/csha2log/'
    train = read_json(csharp_file + 'train.json')
    valid = read_json(csharp_file + 'valid.json')
    test = read_json(csharp_file + 'test.json')
    new_train, new_valid = select_kshot_TwoClass_match(train, K=32)
    save_json(new_train,'/home/zmj/task/LogCL/data_FewShot/code2log/csha2log/train_64Sample.json')
    save_json(new_valid, '/home/zmj/task/LogCL/data_FewShot/code2log/csha2log/valid_64Sample.json')


def select_Kshot_MulClass_IF(data, K: int = 4, label_dict=None):
    if label_dict == None:
        label_set = [0, 1]
    else:
        label_set = list(range(len(label_dict)))
    random.shuffle(data)
    new_train_data = []
    new_valid_data = []
    train_data_split = {}
    for i in data:
        label = i[-1]
        if label in train_data_split:
            train_data_split[label].append(i)
        else:
            train_data_split[label] = [i]
    for key in train_data_split:
        print(key, ': ', len(train_data_split[key]))

    for key in train_data_split:
        if len(train_data_split[key]) < K:
            pass
        else:
            sample_K = train_data_split[key][:K]
            random.shuffle(train_data_split[key])
            sample_K_v = train_data_split[key][:K]
            new_train_data.extend(sample_K)
            new_valid_data.extend(sample_K_v)
    return new_train_data, new_valid_data

def thunderbirdAD():
    token_file = '/home/zmj/task/LogCL/Thunderbird_ori/thunderbird_AD_200k/'
    train_data = read_json(token_file + 'train.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')

    new_train, new_valid = select_kshot_TwoClass(train_data, K=32)
    # new_train, new_valid = select_thunder(train, K=16)

    save_json(new_train, '/home/zmj/task/LogCL/data_FewShot/anomadet/thunderAD/train_64sample.json')
    save_json(new_valid, '/home/zmj/task/LogCL/data_FewShot/anomadet/thunderAD/valid_64sample.json')


def thunderbirdFI():
    thunderfile = '/home/zmj/task/LogCL/Thunderbird_ori/thunderbird_1m/'
    train = read_json(thunderfile + 'train.json')
    valid = read_json(thunderfile + 'valid.json')
    test = read_json(thunderfile + 'test.json')
    label_dict = read_json(thunderfile + 'label_dict.json')
    new_train, new_valid = select_Kshot_MulClass_IF(train, K=3, label_dict=label_dict)
    # new_train, new_valid = select_thunder(train, K=16)

    save_json(new_train, '/home/zmj/task/LogCL/data_FewShot/failure/thunderFI/train_3.json')
    save_json(new_valid, '/home/zmj/task/LogCL/data_FewShot/failure/thunderFI/valid_3.json')


def bglAD():
    token_file = '../data_zmj/anomadet/bglanoma/'
    train_data = read_json(token_file + 'train.json')
    valid_data = read_json(token_file + 'valid.json')
    test_data = read_json(token_file + 'test.json')

    new_train, new_valid = select_kshot_TwoClass(train_data, K=32)
    # new_train, new_valid = select_thunder(train, K=16)

    save_json(new_train, '/home/zmj/task/LogCL/data_FewShot/anomadet/BGLAD/train_64sample.json')
    save_json(new_valid, '/home/zmj/task/LogCL/data_FewShot/anomadet/BGLAD/valid_64sample.json')


def bglFI():
    file = '../data_zmj/anomadet/bglanoma_mul/'
    train_data = read_json(file+ 'train.json')
    valid_data = read_json(file + 'valid.json')
    test_data = read_json(file + 'test.json')
    label_dict = read_json(file + 'label_dict.json')
    new_train, new_valid = select_Kshot_MulClass_IF(train_data, K=6, label_dict=label_dict)

    save_json(new_train, '/home/zmj/task/LogCL/data_FewShot/failure/BGLFI/train_6.json')
    save_json(new_valid, '/home/zmj/task/LogCL/data_FewShot/failure/BGLFI/valid_6.json')


def log2sumy():
    file = '../data_zmj/log2sumy/allg2smy/'
    train_data = read_json(file + 'train.json')
    valid_data = read_json(file + 'valid.json')
    test_data = read_json(file + 'test.json')
    new_train, new_valid = select_kshot_TwoClass_match(train_data, K=32)
    save_json(new_train, '/home/zmj/task/LogCL/data_FewShot/log2sumy/allg2smy/train_64sample.json')
    save_json(new_valid, '/home/zmj/task/LogCL/data_FewShot/log2sumy/allg2smy/valid_64sample.json')

def java2log():
    csharp_file = '../data_zmj/code2log/java2log/'
    train = read_json(csharp_file + 'train.json')
    valid = read_json(csharp_file + 'valid.json')
    test = read_json(csharp_file + 'test.json')
    new_train, new_valid = select_kshot_TwoClass_match(train, K=32)
    save_json(new_train, '/home/zmj/task/LogCL/data_FewShot/code2log/java2log/train_64Sample.json')
    save_json(new_valid, '/home/zmj/task/LogCL/data_FewShot/code2log/java2log/valid_64Sample.json')



if __name__ == '__main__':
    # token_cls()
    # dimen_cls()
    csharp2log()
    # thunderbirdAD()
    # thunderbirdFI()
    # bglAD()
    # bglFI()
    # log2sumy()
    # java2log()