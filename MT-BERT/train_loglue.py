import datetime
import gc
import hashlib
import math
import operator
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
from random import sample
import pandas as pd
# import pytorch_warmup as warmup
import scipy
import torch
from torch import optim
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss

from tqdm import tqdm
from transformers import BertTokenizer,BertModel,get_scheduler
import argparse
import os
from model import MT_BERT, MT_PROMPT,MT_PREFIX, MT_PREFIX_WO_BGLF,MT_PREFIX_WO_BGLC
# from task import Task, define_dataset_config, define_tasks_config
from task_few import Task, define_dataset_config, define_tasks_config
from utils import stream_redirect_tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")



def train_minibatch(input_data, task, label, model, task_criterion, **kwargs):
    output = model(input_data, task)
    loss = task_criterion(output, label)
    loss.backward()
    del output
    return loss



def main_sp_v3(task_name):

    NUM_EPOCHS = int(100)
    datasets_config = define_dataset_config()
    tasks_config = define_tasks_config(datasets_config)
    task_actions = []
    task_list = [
                        Task.Token_C,
                        Task.BGL_C,
                        Task.Thund_C,
                        Task.BGL_F,
                        Task.Thund_F,
                        Task.Dimen_C,
                        Task.Log2Sum,
                        Task.Java2Log,
                        Task.Csharp2Log,
                        ]

    if task_name == 'Token_C':
        task_list = [Task.Token_C]
    elif task_name == 'BGL_C':
        task_list = [Task.BGL_C]
    elif task_name == 'Thund_C':
        task_list = [Task.Thund_C]
    elif task_name == 'BGL_F':
        task_list = [Task.BGL_F]
    elif task_name == 'Thund_F':
        task_list = [Task.Thund_F]
    elif task_name == 'Dimen_C':
        task_list = [Task.Dimen_C]
    elif task_name == 'Log2Sum':
        task_list = [Task.Log2Sum]
    elif task_name == 'Java2Log':
        task_list = [Task.Java2Log]
    elif task_name == 'Csharp2Log':
        task_list = [Task.Csharp2Log]



    for task in iter(Task):
        if task not in task_list:
            train_loader = tasks_config[task]["train_loader"]
            task_actions.extend([task] * len(train_loader))
    epoch_steps = len(task_actions)
    # model_path = r'D:\ZMJ\Local-model\bert-base-uncased'
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    model = MT_PREFIX_WO_BGLC(model_path)
    # print(model.state_dict().keys())
    model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=5e-2)
    initial_epoch = 0
    training_start = datetime.datetime.now().isoformat()


    print("Starting training from scratch")
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    # warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=(epoch_steps * NUM_EPOCHS) // 10)
    num_train_step = NUM_EPOCHS * epoch_steps
    lr_scheduler = get_scheduler(name='linear',optimizer=optimizer,num_warmup_steps=num_train_step//10,num_training_steps=num_train_step)


    print(f"------------------ training-start:  {training_start} --------------------------)")

    losses = {'BCELoss': BCELoss(), 'CrossEntropyLoss': CrossEntropyLoss(), 'MSELoss': MSELoss()}
    for name, loss in losses.items():
        losses[name].to(device)

    for epoch in range(initial_epoch, NUM_EPOCHS):
        with stream_redirect_tqdm() as orig_stdout:
            task_sample = sample(task_actions, len(task_actions))
            epoch_bar = tqdm((task_sample), file=orig_stdout)
            #
            model.train()
            # print(epoch_bar)

            train_loaders = { task: iter(tasks_config[task]["train_loader"]) for task in set(task_actions) }

            for task_action in epoch_bar:
                # print(task_action)
                train_loader = tasks_config[task_action]["train_loader"]
                epoch_bar.set_description(f"current task: {task_action.name} in epoch:{epoch}")

                try:
                    data = next(train_loaders[task_action])
                except StopIteration:
                    print(f"Iterator ended early on task {task_action}")
                    continue

                optimizer.zero_grad(set_to_none=True)
                label = data[2].squeeze(1).to(device)
                task_criterion = losses[MT_PREFIX_WO_BGLF.loss_for_task(task_action)]

                loss = train_minibatch(input_data=data, task=task_action, label=label, model=model,
                                task_criterion=task_criterion, optimizer=optimizer)
                # print(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                lr_scheduler.step()


            if (epoch+1) % 5 ==0:
                output_dir = f"save_models_fewshot_wo_{task_name}/"

                for name,para in model.named_parameters():
                    if name ==  'prefix_encoder_TokenC.embedding.weight':
                        prefix_encoder_TokenC_weights_params = para
                        torch.save(prefix_encoder_TokenC_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_TokenC_embedding_weights_params.pt"))
                    if name ==  'prefix_encoder_BGLC.embedding.weight':
                        prefix_encoder_BGLC_weights_params = para
                        torch.save(prefix_encoder_BGLC_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_BGLC_embedding_weights_params.pt"))
                    if name ==  'prefix_encoder_BGLF.embedding.weight':
                        prefix_encoder_BGLF_weights_params = para
                        torch.save(prefix_encoder_BGLF_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_BGLF_embedding_weights_params.pt"))
                    if name ==  'prefix_encoder_ThunC.embedding.weight':
                        prefix_encoder_ThunC_weights_params = para
                        torch.save(prefix_encoder_ThunC_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_ThunC_embedding_weights_params.pt"))
                    if name ==  'prefix_encoder_ThunF.embedding.weight':
                        prefix_encoder_ThunF_weights_params = para
                        torch.save(prefix_encoder_ThunF_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_ThunF_embedding_weights_params.pt"))
                    if name ==  'prefix_encoder_DimC.embedding.weight':
                        prefix_encoder_DimC_weights_params = para
                        torch.save(prefix_encoder_DimC_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_DimC_embedding_weights_params.pt"))
                    if name == 'prefix_encoder_Log2Sum.embedding.weight':
                        prefix_encoder_Log2Sum_weights_params = para
                        torch.save(prefix_encoder_Log2Sum_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_Log2Sum_embedding_weights_params.pt"))
                    if name == 'prefix_encoder_Java2Log.embedding.weight':
                        prefix_encoder_Java2Log_weights_params = para
                        torch.save(prefix_encoder_Java2Log_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_Java2Log_embedding_weights_params.pt"))
                    if name == 'prefix_encoder_Csharp2Log.embedding.weight':
                        prefix_encoder_Csharp2Log_weights_params = para
                        torch.save(prefix_encoder_Csharp2Log_weights_params,
                                   (output_dir + f"epoch_{epoch}_prefix_encoder_Csharp2Log_embedding_weights_params.pt"))

                print('save models')
                model.eval()
                val_results = {}
                with torch.no_grad():
                    task_bar = tqdm([task for task in Task if task not in task_list], file=orig_stdout)
                    for task in task_bar:
                        task_bar.set_description(task.name)
                        # val_loader = tasks_config[task]["val_loader"]
                        test_loader = tasks_config[task]["test_loader"]
                        task_predicted_labels = torch.empty(0, device=device)
                        task_labels = torch.empty(0, device=device)

                        for test_data in test_loader:
                            label = test_data[2].squeeze().to(device)
                            model_output = model(test_data, task)

                            if task.num_classes() > 1:
                                predict_y = model_output.softmax(dim=-1)
                                predicted_label = torch.argmax(predict_y, -1)
                            else:
                                predicted_label = model_output

                            task_predicted_labels = torch.hstack((task_predicted_labels, predicted_label.view(-1)))
                            task_labels = torch.hstack((task_labels, label))

                        metrics = datasets_config[task].metrics
                        for metric in metrics:
                            if metric.__name__ == 'accuracy_score':
                                metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu())
                            else:
                                metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu(),average='macro')
                            if type(metric_result) == tuple or type(metric_result) == scipy.stats.stats.SpearmanrResult:
                                metric_result = metric_result[0]
                            val_results[task.name, metric.__name__] = metric_result
                            print(
                                f"val_results[{task.name}, {metric.__name__}] = {val_results[task.name, metric.__name__]}")


    #test
    print('-----------------------------------------------------------')
    print('Start test')
    model.eval()
    test_results = {}
    output_file = open('test_result.txt','w',encoding='utf-8')

    with torch.no_grad():
        task_bar = tqdm([task for task in Task if task not in task_list], file=orig_stdout)
        for task in task_bar:
            task_bar.set_description(task.name)
            test_loader = tasks_config[task]["test_loader"]

            task_predicted_labels = torch.empty(0, device=device)
            task_labels = torch.empty(0, device=device)
            for test_data in test_loader:

                label = test_data[2].squeeze().to(device)

                model_output = model(test_data, task)

                if task.num_classes() > 1:
                    predict_y = model_output.softmax(dim=-1)
                    predicted_label = torch.argmax(predict_y, -1)
                else:
                    predicted_label = model_output

                task_predicted_labels = torch.hstack((task_predicted_labels, predicted_label.view(-1)))
                task_labels = torch.hstack((task_labels, label))

            metrics = datasets_config[task].metrics
            for metric in metrics:
                if metric.__name__ == 'accuracy_score':
                    metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu())
                else:
                    metric_result = metric(task_labels.cpu(), task_predicted_labels.cpu(), average='macro')
                if type(metric_result) == tuple or type(metric_result) == scipy.stats.stats.SpearmanrResult:
                    metric_result = metric_result[0]
                test_results[task.name, metric.__name__] = metric_result
                print(
                    f"val_results[{task.name}, {metric.__name__}] = {test_results[task.name, metric.__name__]}")
                output_file.write(f"val_results[{task.name}, {metric.__name__}] = {test_results[task.name, metric.__name__]}"+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Token_C")
    parser.add_argument("--OutputPath", type=str, default="Token_C")
    args = parser.parse_args()
    main_sp_v3(args.task)
