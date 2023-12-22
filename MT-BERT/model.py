import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizer, BertModel
# from models.prefix_encoder import PrefixEncoder_clean
# from task import Task
from task_few import Task

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SSCModule(nn.Module):  # Single sentence classification
    def __init__(self, hidden_size, dropout_prob=0.1, output_classes=2):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_classes))

    def forward(self, x):
        return self.output_layer(x)

# class MPCModule(nn.Module): # mask prediction classification
#     def __init__(self,hidden_size,dropout_prob=0.1,output_classes=2):
#         super().__init__()
#
#     def forward(self,x,):



class PrefixEncoder_clean(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, pre_seq_len,hidden_size,prefix_hidden_size,num_hidden_layers,prefix_projection=False):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size,prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values



# class PTSModule(nn.Module):  # Pairwise text similarity
#     def __init__(self, hidden_size, dropout_prob=0.1):
#         super().__init__()
#         self.output_layer = nn.Sequential(
#             nn.Dropout(dropout_prob),
#             nn.Linear(hidden_size, 1),
#         )
#
#     def forward(self, x):
#         return self.output_layer(x).view(-1)


# class PTCModule(nn.Module):  # Pariwise text classification
#     def __init__(self, hidden_size, k_steps, output_classes, dropout_prob=0.1, stochastic_prediction_dropout_prob=0.1):
#         super().__init__()
#         self.stochastic_prediction_dropout = stochastic_prediction_dropout_prob
#         self.k_steps = k_steps
#         self.hidden_size = hidden_size
#         self.output_classes = output_classes
#
#         self.GRU = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
#
#         self.W1 = nn.Sequential(
#             nn.Dropout(dropout_prob),
#             nn.Linear(hidden_size, 1),
#             )
#
#         self.W2 = nn.Sequential(
#             nn.Dropout(dropout_prob),
#             nn.Linear(hidden_size, hidden_size))
#
#         self.W3 = nn.Sequential(
#             nn.Dropout(dropout_prob),
#             nn.Linear(4 * hidden_size, output_classes),
#         )
#
#     def forward(self, premises: torch.Tensor, hypotheses: torch.Tensor):
#         batch_size = premises.size(0)
#
#         output_probabilities = torch.zeros(batch_size, self.output_classes).to(device)
#
#         flatten_hypotheses = hypotheses.reshape(-1, self.hidden_size)
#         flatten_premise = premises.reshape(-1, self.hidden_size)
#
#         alfas = F.softmax(self.W1(flatten_hypotheses).view(batch_size, - 1), -1)
#         s_state = (alfas.unsqueeze(1) @ hypotheses)  # (Bs,1,hidden)
#
#         layer_output = self.W2(flatten_premise).view(batch_size, -1, self.hidden_size)
#         layer_output_transpose = torch.transpose(layer_output, 1, 2)
#
#         actual_k = 0
#         for k in range(self.k_steps):
#             betas = F.softmax(s_state @ layer_output_transpose, -1)
#             x_input = betas @ premises
#             _, s_state = self.GRU(x_input, s_state.transpose(0, 1))
#             s_state = s_state.transpose(0, 1).to(device)
#             concatenated_features = torch.cat([s_state, x_input, (s_state - x_input).abs(), x_input * s_state],
#                                               -1).to(device)
#             if torch.rand(()) > self.stochastic_prediction_dropout or (not self.training):
#                 output_probabilities += self.W3(concatenated_features).squeeze()
#                 actual_k += 1
#
#         return output_probabilities / actual_k


# class PRModule(nn.Module):  # Pairwise ranking module
#     def __init__(self, hidden_size, dropout_prob=0.1):
#         super().__init__()
#         self.output_layer = nn.Sequential(
#             nn.Dropout(dropout_prob),
#             nn.Linear(hidden_size, 1),
#         )
#
#     def forward(self, x):
#         return torch.sigmoid(self.output_layer(x)).view(x.size(0))


class MT_BERT(nn.Module):
    def __init__(self, bert_pretrained_model="bert-base-uncased"):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        self.bert = BertModel.from_pretrained(bert_pretrained_model)
        self.hidden_size = self.bert.config.hidden_size


        # Single-Sentence Classification modules
        self.Token_C = SSCModule(self.hidden_size,output_classes=9)
        self.Anoma_C = SSCModule(self.hidden_size,output_classes=10)


        # Mask Classification modules
        self.Dimen_C = SSCModule(self.hidden_size,output_classes=13)

        # Pairwise sentence Classification modules
        self.Log2Sum = SSCModule(self.hidden_size)
        self.Code2Log = SSCModule(self.hidden_size)



    def forward(self, x, task: Task):

        input_ids = x[0].to(device)
        atten_mask = x[1].to(device)
        # labels = x[2].squeeze().to(device)
        out_put = self.bert(input_ids, atten_mask)
        cls_embedding = out_put[0][:, 0, :]


        if task == Task.Token_C:
            return self.Token_C(cls_embedding)
        elif task == Task.Anoma_C:
            return self.Anoma_C(cls_embedding)
        elif task == Task.Dimen_C:
            mask_ids = x[3]
            batch_ids = torch.arange(mask_ids.shape[0]).to(device)
            mask_embedding = out_put[0][batch_ids, mask_ids]
            return self.Dimen_C(mask_embedding)
        elif task == Task.Log2Sum:
            return self.Log2Sum(cls_embedding)
        elif task == Task.Csharp2Log:
            return self.Code2Log(cls_embedding)



    @staticmethod
    def loss_for_task(t: Task):
        losses = {

            Task.Token_C: "CrossEntropyLoss",
            Task.Num_C: "CrossEntropyLoss",
            Task.SpToken_C: "CrossEntropyLoss",
            Task.Anoma_C: "CrossEntropyLoss",
            Task.Dimen_C: "CrossEntropyLoss",
            Task.Log2Sum: "CrossEntropyLoss",
            Task.Java2Log: "CrossEntropyLoss",
            Task.Csharp2Log: "CrossEntropyLoss"

        }

        return losses[t]

class MT_PROMPT(nn.Module):
    def __init__(self, model_path,pre_seq_len=20):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size
        self.embeddings = self.bert.embeddings

        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = pre_seq_len
        self.n_layer = self.bert.config.num_hidden_layers
        self.n_head = self.bert.config.num_attention_heads
        self.n_embd = self.bert.config.hidden_size // self.bert.config.num_attention_heads



        # Single-Sentence Classification modules
        # self.Token_C = SSCModule(self.hidden_size,output_classes=9)
        # self.Num_C = SSCModule(self.hidden_size,output_classes=4)
        # self.SpToken_C = SSCModule(self.hidden_size,output_classes=5)

        self.Anoma_C = SSCModule(self.hidden_size,output_classes=10)

        # Mask Classification modules
        # self.Dimen_C = SSCModule(self.hidden_size,output_classes=13)

        # Pairwise sentence Classification modules
        self.Log2Sum = SSCModule(self.hidden_size)
        # self.Csharp2Log = SSCModule(self.hidden_size)
        self.Java2Log = SSCModule(self.hidden_size)

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder_generate = torch.nn.Embedding(self.pre_seq_len, self.bert.config.hidden_size)
        self.prefix_encoder_Anoma_C = torch.nn.Embedding(self.pre_seq_len, self.bert.config.hidden_size)
        self.prefix_encoder_Log2Sum = torch.nn.Embedding(self.pre_seq_len, self.bert.config.hidden_size)
        self.prefix_encoder_Java2Log = torch.nn.Embedding(self.pre_seq_len, self.bert.config.hidden_size)

    def get_prompt(self, batch_size,task: Task=None):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        if task == Task.Anoma_C:
            prompts = self.prefix_encoder_Anoma_C(prefix_tokens)
        elif task == Task.Log2Sum:
            prompts = self.prefix_encoder_Log2Sum(prefix_tokens)
        elif task == Task.Csharp2Log:
            prompts = self.prefix_encoder_Java2Log(prefix_tokens)
        else:
            prompts = self.prefix_encoder_generate(prefix_tokens)
        return prompts


    def forward(self, x, task: Task):

        input_ids = x[0].to(device)
        atten_mask = x[1].to(device)
        # labels = x[2].squeeze().to(device)
        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids=input_ids
        )
        prompts_G = self.get_prompt(batch_size=batch_size,task=None)

        if task == Task.Anoma_C:

            for param_A in self.prefix_encoder_Anoma_C.parameters():
                param_A.requires_grad = True
            for param_L in self.prefix_encoder_Log2Sum.parameters():
                param_L.requires_grad = False
            for param_J in self.prefix_encoder_Java2Log.parameters():
                param_J.requires_grad = False

            prompts_E = self.get_prompt(batch_size=batch_size, task=task)
            inputs_embeds = torch.cat((prompts_G,prompts_E,raw_embedding), dim=1)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len*2).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)
            outputs = self.bert(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            sequence_output = outputs[0]
            sequence_output = sequence_output[:, self.pre_seq_len*2:, :].contiguous()
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.bert.pooler.dense(first_token_tensor)
            pooled_output = self.bert.pooler.activation(pooled_output)
            logits = self.Anoma_C(pooled_output)
            return logits
        elif task == Task.Log2Sum:
            for param_A in self.prefix_encoder_Anoma_C.parameters():
                param_A.requires_grad = False
            for param_L in self.prefix_encoder_Log2Sum.parameters():
                param_L.requires_grad = True
            for param_J in self.prefix_encoder_Java2Log.parameters():
                param_J.requires_grad = False

            prompts_E = self.get_prompt(batch_size=batch_size, task=task)
            inputs_embeds = torch.cat((prompts_G,prompts_E,raw_embedding), dim=1)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len*2).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)
            outputs = self.bert(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            sequence_output = outputs[0]
            sequence_output = sequence_output[:, self.pre_seq_len*2:, :].contiguous()
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.bert.pooler.dense(first_token_tensor)
            pooled_output = self.bert.pooler.activation(pooled_output)
            logits = self.Log2Sum(pooled_output)
            return logits
        elif task == Task.Java2Log:
            for param_A in self.prefix_encoder_Anoma_C.parameters():
                param_A.requires_grad = False
            for param_L in self.prefix_encoder_Log2Sum.parameters():
                param_L.requires_grad = False
            for param_J in self.prefix_encoder_Java2Log.parameters():
                param_J.requires_grad = True

            prompts_E = self.get_prompt(batch_size=batch_size, task=task)
            inputs_embeds = torch.cat((prompts_G, prompts_E, raw_embedding), dim=1)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len * 2).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)
            outputs = self.bert(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            sequence_output = outputs[0]
            sequence_output = sequence_output[:, self.pre_seq_len * 2:, :].contiguous()
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.bert.pooler.dense(first_token_tensor)
            pooled_output = self.bert.pooler.activation(pooled_output)
            logits = self.Java2Log(pooled_output)
            return logits



    @staticmethod
    def loss_for_task(t: Task):
        losses = {

            Task.Token_C: "CrossEntropyLoss",
            Task.Num_C: "CrossEntropyLoss",
            Task.SpToken_C: "CrossEntropyLoss",
            Task.Anoma_C: "CrossEntropyLoss",
            Task.Dimen_C: "CrossEntropyLoss",
            Task.Log2Sum: "CrossEntropyLoss",
            Task.Java2Log: "CrossEntropyLoss",
            Task.Csharp2Log: "CrossEntropyLoss"

        }

        return losses[t]


class MT_PREFIX(nn.Module):
    def __init__(self, model_path, pre_seq_len=20):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size
        self.embeddings = self.bert.embeddings
        self.dropout = torch.nn.Dropout(0.1)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = pre_seq_len
        self.n_layer = self.bert.config.num_hidden_layers
        self.n_head = self.bert.config.num_attention_heads
        self.n_embd = self.bert.config.hidden_size // self.bert.config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        # if self.general_prefix==True:
        #     self.prefix_encoder_general = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512, self.n_layer)
        self.prefix_encoder_Anoma_C = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512, self.n_layer)
        self.prefix_encoder_Log2Sum = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512, self.n_layer)
        self.prefix_encoder_Java2Log = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512, self.n_layer)


        # Single-Sentence Classification modules
        # self.Token_C = SSCModule(self.hidden_size,output_classes=9)
        # self.Num_C = SSCModule(self.hidden_size,output_classes=4)
        # self.SpToken_C = SSCModule(self.hidden_size,output_classes=5)

        self.Anoma_C = SSCModule(self.hidden_size, output_classes=10)

        # Mask Classification modules
        # self.Dimen_C = SSCModule(self.hidden_size,output_classes=13)

        # Pairwise sentence Classification modules
        self.Log2Sum = SSCModule(self.hidden_size)
        # self.Csharp2Log = SSCModule(self.hidden_size)
        self.Java2Log = SSCModule(self.hidden_size)


        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size, task: Task = None):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        if task == Task.Anoma_C:
            past_key_values = self.prefix_encoder_Anoma_C(prefix_tokens)
        elif task == Task.Log2Sum:
            past_key_values = self.prefix_encoder_Log2Sum(prefix_tokens)
        elif task == Task.Java2Log:
            past_key_values = self.prefix_encoder_Java2Log(prefix_tokens)
        else:
            past_key_values=None
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values


    def forward(self, x, task: Task):

        input_ids = x[0].to(device)
        atten_mask = x[1].to(device)
        batch_size = input_ids.shape[0]
        # if self.general_prefix:
        #     past_key_values_g = self.get_prompt(batch_size=batch_size,task=None)


        if task == Task.Anoma_C:
            for param_A in self.prefix_encoder_Anoma_C.parameters():
                param_A.requires_grad = True
            for param_L in self.prefix_encoder_Log2Sum.parameters():
                param_L.requires_grad = False
            for param_J in self.prefix_encoder_Java2Log.parameters():
                param_J.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Anoma_C(pooled_output)
            return logits

        elif task == Task.Log2Sum:
            for param_A in self.prefix_encoder_Anoma_C.parameters():
                param_A.requires_grad = False
            for param_L in self.prefix_encoder_Log2Sum.parameters():
                param_L.requires_grad = True
            for param_J in self.prefix_encoder_Java2Log.parameters():
                param_J.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size, task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Log2Sum(pooled_output)
            return logits

        elif task == Task.Java2Log:
            for param_A in self.prefix_encoder_Anoma_C.parameters():
                param_A.requires_grad = False
            for param_L in self.prefix_encoder_Log2Sum.parameters():
                param_L.requires_grad = False
            for param_J in self.prefix_encoder_Java2Log.parameters():
                param_J.requires_grad = True

            past_key_values = self.get_prompt(batch_size=batch_size, task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Java2Log(pooled_output)
            return logits


    @staticmethod
    def loss_for_task(t: Task):
        losses = {

            Task.Token_C: "CrossEntropyLoss",
            Task.Num_C: "CrossEntropyLoss",
            Task.SpToken_C: "CrossEntropyLoss",
            Task.Anoma_C: "CrossEntropyLoss",
            Task.Thund_C:"CrossEntropyLoss",
            Task.Dimen_C: "CrossEntropyLoss",
            Task.Log2Sum: "CrossEntropyLoss",
            Task.Java2Log: "CrossEntropyLoss",
            Task.Csharp2Log: "CrossEntropyLoss"

        }

        return losses[t]


class MT_PREFIX_WO_BGLF(nn.Module):
    def __init__(self, model_path, pre_seq_len=20):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size
        self.embeddings = self.bert.embeddings
        self.dropout = torch.nn.Dropout(0.1)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = pre_seq_len
        self.n_layer = self.bert.config.num_hidden_layers
        self.n_head = self.bert.config.num_attention_heads
        self.n_embd = self.bert.config.hidden_size // self.bert.config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()



        self.prefix_encoder_TokenC = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                         self.n_layer)
        self.prefix_encoder_BGLC = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                       self.n_layer)
        # self.prefix_encoder_BGLF = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
        #                                                self.n_layer)
        self.prefix_encoder_ThunC = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                        self.n_layer)
        self.prefix_encoder_ThunF = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                        self.n_layer)
        self.prefix_encoder_DimC = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                       self.n_layer)
        self.prefix_encoder_Log2Sum = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                          self.n_layer)
        self.prefix_encoder_Java2Log = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_Csharp2Log = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                             self.n_layer)


        self.TokenC = SSCModule(self.hidden_size,output_classes=9)
        self.BGLC = SSCModule(self.hidden_size)
        # self.BGLF = SSCModule(self.hidden_size,output_classes=10)
        self.ThundC = SSCModule(self.hidden_size)
        self.ThundF = SSCModule(self.hidden_size, output_classes=5)
        self.DimenC = SSCModule(self.hidden_size,output_classes=13)
        self.Log2Sum = SSCModule(self.hidden_size)
        self.Csharp2Log = SSCModule(self.hidden_size)
        self.Java2Log = SSCModule(self.hidden_size)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size, task: Task = None):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        if task == Task.Token_C:
            past_key_values = self.prefix_encoder_TokenC(prefix_tokens)
        elif task == Task.BGL_C:
            past_key_values = self.prefix_encoder_BGLC(prefix_tokens)
        elif task == Task.BGL_F:
            past_key_values = self.prefix_encoder_BGLF(prefix_tokens)
        elif task == Task.Thund_C:
            past_key_values = self.prefix_encoder_ThunC(prefix_tokens)
        elif task == Task.Thund_F:
            past_key_values = self.prefix_encoder_ThunF(prefix_tokens)
        elif task == Task.Dimen_C:
            past_key_values = self.prefix_encoder_DimC(prefix_tokens)
        elif task == Task.Log2Sum:
            past_key_values = self.prefix_encoder_Log2Sum(prefix_tokens)
        elif task == Task.Csharp2Log:
            past_key_values = self.prefix_encoder_Csharp2Log(prefix_tokens)
        elif task == Task.Java2Log:
            past_key_values = self.prefix_encoder_Java2Log(prefix_tokens)
        else:
            past_key_values=None
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values


    def forward(self, x, task: Task):

        input_ids = x[0].to(device)
        atten_mask = x[1].to(device)
        batch_size = input_ids.shape[0]

        mod = [
            self.prefix_encoder_TokenC,
            self.prefix_encoder_BGLC,
            # self.prefix_encoder_BGLF,
            self.prefix_encoder_ThunC,
            self.prefix_encoder_ThunF,
            self.prefix_encoder_DimC,
            self.prefix_encoder_Log2Sum,
            self.prefix_encoder_Java2Log,
            self.prefix_encoder_Csharp2Log,
            self.TokenC,
            self.BGLC,
            # self.BGLF,
            self.ThundC,
            self.ThundF,
            self.DimenC,
            self.Log2Sum,
            self.Java2Log,
            self.Csharp2Log
               ]
        if task == Task.Token_C:
            for s in mod:
                if s == self.prefix_encoder_TokenC or s == self.TokenC:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.TokenC(pooled_output)
            return logits
        elif task == Task.BGL_C:
            for s in mod:
                if s == self.prefix_encoder_BGLC or s == self.BGLC:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False


            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.BGLC(pooled_output)
            return logits

        # elif task == Task.BGL_F:
        #     for s in mod:
        #         if s == self.prefix_encoder_BGLF or s == self.BGLF:
        #             for para in s.parameters():
        #                 para.requires_grad = True
        #         else:
        #             for para in s.parameters():
        #                 para.requires_grad = False
        #
        #     past_key_values = self.get_prompt(batch_size=batch_size,task=task)
        #     prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        #     attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)
        #
        #     outputs = self.bert(
        #         input_ids,
        #         attention_mask=attention_mask,
        #         past_key_values=past_key_values,
        #     )
        #     pooled_output = outputs[1]
        #     pooled_output = self.dropout(pooled_output)
        #     logits = self.BGLF(pooled_output)
        #     return logits

        elif task == Task.Thund_C:
            for s in mod:
                if s == self.prefix_encoder_ThunC or s == self.ThundC:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False


            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.ThundC(pooled_output)
            return logits
        elif task == Task.Thund_F:
            for s in mod:
                if s == self.prefix_encoder_ThunF or s == self.ThundF:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False


            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.ThundF(pooled_output)
            return logits
        elif task == Task.Dimen_C:
            for s in mod:
                if s == self.prefix_encoder_DimC or s == self.DimenC:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            mask_ids = x[-1]
            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            all_output = outputs[0]
            batch_ids = torch.arange(mask_ids.shape[0])
            first_token_tensor = all_output[batch_ids, mask_ids]
            pooled_output = self.dropout(first_token_tensor)
            logits = self.DimenC(pooled_output)
            return logits
        elif task == Task.Log2Sum:
            for s in mod:
                if s == self.prefix_encoder_Log2Sum or s == self.Log2Sum:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size, task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Log2Sum(pooled_output)
            return logits
        elif task == Task.Java2Log:
            for s in mod:
                if s == self.prefix_encoder_Java2Log or s == self.Java2Log:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size, task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Java2Log(pooled_output)
            return logits
        elif task == Task.Csharp2Log:
            for s in mod:
                if s == self.prefix_encoder_Csharp2Log or s == self.Csharp2Log:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size, task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Csharp2Log(pooled_output)
            return logits

    @staticmethod
    def loss_for_task(t: Task):
        losses = {

            Task.Token_C: "CrossEntropyLoss",
            Task.BGL_C: "CrossEntropyLoss",
            Task.BGL_F: "CrossEntropyLoss",
            Task.Thund_C:"CrossEntropyLoss",
            Task.Thund_F: "CrossEntropyLoss",
            Task.Dimen_C: "CrossEntropyLoss",
            Task.Log2Sum: "CrossEntropyLoss",
            Task.Java2Log: "CrossEntropyLoss",
            Task.Csharp2Log: "CrossEntropyLoss"

        }

        return losses[t]

class MT_PREFIX_WO_BGLC(nn.Module):
    def __init__(self, model_path, pre_seq_len=20):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size
        self.embeddings = self.bert.embeddings
        self.dropout = torch.nn.Dropout(0.1)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = pre_seq_len
        self.n_layer = self.bert.config.num_hidden_layers
        self.n_head = self.bert.config.num_attention_heads
        self.n_embd = self.bert.config.hidden_size // self.bert.config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()



        self.prefix_encoder_TokenC = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                         self.n_layer)
        # self.prefix_encoder_BGLC = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
        #                                                self.n_layer)
        self.prefix_encoder_BGLF = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                       self.n_layer)
        self.prefix_encoder_ThunC = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                        self.n_layer)
        self.prefix_encoder_ThunF = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                        self.n_layer)
        self.prefix_encoder_DimC = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                       self.n_layer)
        self.prefix_encoder_Log2Sum = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                          self.n_layer)
        self.prefix_encoder_Java2Log = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_Csharp2Log = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                             self.n_layer)


        self.TokenC = SSCModule(self.hidden_size,output_classes=9)
        # self.BGLC = SSCModule(self.hidden_size)
        self.BGLF = SSCModule(self.hidden_size,output_classes=10)
        self.ThundC = SSCModule(self.hidden_size)
        self.ThundF = SSCModule(self.hidden_size, output_classes=5)
        self.DimenC = SSCModule(self.hidden_size,output_classes=13)
        self.Log2Sum = SSCModule(self.hidden_size)
        self.Csharp2Log = SSCModule(self.hidden_size)
        self.Java2Log = SSCModule(self.hidden_size)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))  # 9860105

    def get_prompt(self, batch_size, task: Task = None):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        if task == Task.Token_C:
            past_key_values = self.prefix_encoder_TokenC(prefix_tokens)
        elif task == Task.BGL_C:
            past_key_values = self.prefix_encoder_BGLC(prefix_tokens)
        elif task == Task.BGL_F:
            past_key_values = self.prefix_encoder_BGLF(prefix_tokens)
        elif task == Task.Thund_C:
            past_key_values = self.prefix_encoder_ThunC(prefix_tokens)
        elif task == Task.Thund_F:
            past_key_values = self.prefix_encoder_ThunF(prefix_tokens)
        elif task == Task.Dimen_C:
            past_key_values = self.prefix_encoder_DimC(prefix_tokens)
        elif task == Task.Log2Sum:
            past_key_values = self.prefix_encoder_Log2Sum(prefix_tokens)
        elif task == Task.Csharp2Log:
            past_key_values = self.prefix_encoder_Csharp2Log(prefix_tokens)
        elif task == Task.Java2Log:
            past_key_values = self.prefix_encoder_Java2Log(prefix_tokens)
        else:
            past_key_values=None
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return past_key_values


    def forward(self, x, task: Task):

        input_ids = x[0].to(device)
        atten_mask = x[1].to(device)
        batch_size = input_ids.shape[0]

        mod = [
            self.prefix_encoder_TokenC,
            # self.prefix_encoder_BGLC,
            self.prefix_encoder_BGLF,
            self.prefix_encoder_ThunC,
            self.prefix_encoder_ThunF,
            self.prefix_encoder_DimC,
            self.prefix_encoder_Log2Sum,
            self.prefix_encoder_Java2Log,
            self.prefix_encoder_Csharp2Log,
            self.TokenC,
            # self.BGLC,
            self.BGLF,
            self.ThundC,
            self.ThundF,
            self.DimenC,
            self.Log2Sum,
            self.Java2Log,
            self.Csharp2Log
               ]
        if task == Task.Token_C:
            for s in mod:
                if s == self.prefix_encoder_TokenC or s == self.TokenC:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.TokenC(pooled_output)
            return logits

        # elif task == Task.BGL_C:
        #     for s in mod:
        #         if s == self.prefix_encoder_BGLC or s == self.BGLC:
        #             for para in s.parameters():
        #                 para.requires_grad = True
        #         else:
        #             for para in s.parameters():
        #                 para.requires_grad = False
        #
        #
        #     past_key_values = self.get_prompt(batch_size=batch_size,task=task)
        #     prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        #     attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)
        #
        #     outputs = self.bert(
        #         input_ids,
        #         attention_mask=attention_mask,
        #         past_key_values=past_key_values,
        #     )
        #     pooled_output = outputs[1]
        #     pooled_output = self.dropout(pooled_output)
        #     logits = self.BGLC(pooled_output)
        #     return logits

        elif task == Task.BGL_F:
            for s in mod:
                if s == self.prefix_encoder_BGLF or s == self.BGLF:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.BGLF(pooled_output)
            return logits

        elif task == Task.Thund_C:
            for s in mod:
                if s == self.prefix_encoder_ThunC or s == self.ThundC:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False


            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.ThundC(pooled_output)
            return logits
        elif task == Task.Thund_F:
            for s in mod:
                if s == self.prefix_encoder_ThunF or s == self.ThundF:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False


            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.ThundF(pooled_output)
            return logits
        elif task == Task.Dimen_C:
            for s in mod:
                if s == self.prefix_encoder_DimC or s == self.DimenC:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            mask_ids = x[-1]
            past_key_values = self.get_prompt(batch_size=batch_size,task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            all_output = outputs[0]
            batch_ids = torch.arange(mask_ids.shape[0])
            first_token_tensor = all_output[batch_ids, mask_ids]
            pooled_output = self.dropout(first_token_tensor)
            logits = self.DimenC(pooled_output)
            return logits
        elif task == Task.Log2Sum:
            for s in mod:
                if s == self.prefix_encoder_Log2Sum or s == self.Log2Sum:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size, task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Log2Sum(pooled_output)
            return logits
        elif task == Task.Java2Log:
            for s in mod:
                if s == self.prefix_encoder_Java2Log or s == self.Java2Log:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size, task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Java2Log(pooled_output)
            return logits
        elif task == Task.Csharp2Log:
            for s in mod:
                if s == self.prefix_encoder_Csharp2Log or s == self.Csharp2Log:
                    for para in s.parameters():
                        para.requires_grad = True
                else:
                    for para in s.parameters():
                        para.requires_grad = False

            past_key_values = self.get_prompt(batch_size=batch_size, task=task)
            prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
            attention_mask = torch.cat((prefix_attention_mask, atten_mask), dim=1)

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.Csharp2Log(pooled_output)
            return logits

    @staticmethod
    def loss_for_task(t: Task):
        losses = {

            Task.Token_C: "CrossEntropyLoss",
            Task.BGL_C: "CrossEntropyLoss",
            Task.BGL_F: "CrossEntropyLoss",
            Task.Thund_C:"CrossEntropyLoss",
            Task.Thund_F: "CrossEntropyLoss",
            Task.Dimen_C: "CrossEntropyLoss",
            Task.Log2Sum: "CrossEntropyLoss",
            Task.Java2Log: "CrossEntropyLoss",
            Task.Csharp2Log: "CrossEntropyLoss"

        }

        return losses[t]

if __name__ == '__main__':
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    model = MT_PREFIX(model_path)
    print(model)
    # print(model.state_dict())
    print(model.state_dict().keys())