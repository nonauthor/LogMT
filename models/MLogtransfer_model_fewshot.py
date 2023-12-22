import torch
import os
from torch import nn
import torch.nn.functional as F

# from model_VisualPT.transformers.modeling_outputs import TokenClassifierOutput
# from model_VisualPT.transformers.models.bert import BertLayer
from models.modeling_bert import BertModel

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


class ThreeExpertsModel(nn.Module):
    def __init__(self, num_labels,model_path,pre_seq_len):
        super(ThreeExpertsModel, self).__init__()

        # Set the length of the prompt to the length of the main image plus the length of the secondary image

        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert_config = self.bert.config
        self.n_layer = self.bert_config.num_hidden_layers
        self.n_head = self.bert_config.num_attention_heads
        self.n_embd = self.bert_config.hidden_size // self.bert_config.num_attention_heads

        self.pre_seq_len = pre_seq_len

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder_Target = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                         self.n_layer)
        self.prefix_encoder_source_1 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                          self.n_layer)
        self.prefix_encoder_source_2 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_3 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_4 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_5 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_6 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_7 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_8 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)


        # Determine the contribution of each image representation in each layer
        self.gates = nn.ModuleList([nn.Linear(20 * 768, 9) for i in range(12)])
        self.gates_projection = nn.ModuleList([nn.Linear(768, 768) for i in range(12)])

        self.num_labels = num_labels
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

        source_1_param = 0
        for name, param in self.prefix_encoder_source_1.named_parameters():
            source_1_param += param.numel()
        source_2_param = 0
        for name, param in self.prefix_encoder_source_2.named_parameters():
            source_2_param += param.numel()
        source_3_param = 0
        for name, param in self.prefix_encoder_source_3.named_parameters():
            source_3_param += param.numel()
        source_4_param = 0
        for name, param in self.prefix_encoder_source_4.named_parameters():
            source_4_param += param.numel()
        source_5_param = 0
        for name, param in self.prefix_encoder_source_5.named_parameters():
            source_5_param += param.numel()
        source_6_param = 0
        for name, param in self.prefix_encoder_source_6.named_parameters():
            source_6_param += param.numel()
        source_7_param = 0
        for name, param in self.prefix_encoder_source_7.named_parameters():
            source_7_param += param.numel()
        source_8_param = 0
        for name, param in self.prefix_encoder_source_8.named_parameters():
            source_8_param += param.numel()
        total_param = source_1_param + source_2_param + source_3_param + source_4_param + source_5_param + \
                      source_6_param + source_7_param + source_8_param
        print('total param is {}'.format(total_param))


    def forward(self, input_ids=None, attention_mask=None):
        batch_size = input_ids.shape[0]
        past_key_values, sum_key_values = self.get_prompt(batch_size)

        prompt_guids_length = past_key_values[0][0].shape[3]

        # attention_mask: bsz, seq_len
        # prompt attention， attention mask
        bsz = attention_mask.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.bert.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                # token_type_ids=token_type_ids,
                                past_key_values=past_key_values,
                                sum_key_values=sum_key_values,
                                gates=self.gates,
                                gates_projection=self.gates_projection,
                                return_dict=False)
        sequence_output = bert_output[1]  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        logits = self.fc(sequence_output)  # bsz, len, labels
        moe_loss = bert_output[-1]
        return logits,moe_loss


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values_source_1 = self.prefix_encoder_source_1(prefix_tokens)
        past_key_values_source_1 = past_key_values_source_1.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_2 = self.prefix_encoder_source_2(prefix_tokens)
        past_key_values_source_2 = past_key_values_source_2.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_3 = self.prefix_encoder_source_3(prefix_tokens)
        past_key_values_source_3 = past_key_values_source_3.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_4 = self.prefix_encoder_source_4(prefix_tokens)
        past_key_values_source_4 = past_key_values_source_4.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_4 = self.prefix_encoder_source_4(prefix_tokens)
        past_key_values_source_4 = past_key_values_source_4.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_5 = self.prefix_encoder_source_5(prefix_tokens)
        past_key_values_source_5 = past_key_values_source_5.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_6 = self.prefix_encoder_source_6(prefix_tokens)
        past_key_values_source_6 = past_key_values_source_6.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_7 = self.prefix_encoder_source_7(prefix_tokens)
        past_key_values_source_7 = past_key_values_source_7.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_8 = self.prefix_encoder_source_8(prefix_tokens)
        past_key_values_source_8 = past_key_values_source_8.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )


        past_key_values_Target = self.prefix_encoder_Target(prefix_tokens)
        past_key_values_Target = past_key_values_Target.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        total_key_values = torch.stack(
            [past_key_values_source_1,
             past_key_values_source_2,
             past_key_values_source_3,
             past_key_values_source_4,
             past_key_values_source_5,
             past_key_values_source_6,
             past_key_values_source_7,
             past_key_values_source_8,
             past_key_values_Target])
        sum_key_values = total_key_values.sum(0)
        total_key_values = total_key_values.permute([3, 0, 1, 4, 2, 5]).split(2)
        sum_key_values = sum_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return total_key_values, sum_key_values

class ThreeExpertsModel_ms(nn.Module):
    def __init__(self, num_labels, model_path, pre_seq_len):
        super(ThreeExpertsModel_ms, self).__init__()

        # Set the length of the prompt to the length of the main image plus the length of the secondary image

        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert_config = self.bert.config
        self.n_layer = self.bert_config.num_hidden_layers
        self.n_head = self.bert_config.num_attention_heads
        self.n_embd = self.bert_config.hidden_size // self.bert_config.num_attention_heads

        self.pre_seq_len = pre_seq_len

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder_Target = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                         self.n_layer)
        self.prefix_encoder_source_1 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_2 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_3 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_4 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_5 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_6 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_7 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)
        self.prefix_encoder_source_8 = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                           self.n_layer)

        # Determine the contribution of each image representation in each layer
        self.gates = nn.ModuleList([nn.Linear(20 * 768, 9) for i in range(12)])
        self.gates_projection = nn.ModuleList([nn.Linear(768, 768) for i in range(12)])

        self.num_labels = num_labels
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

        source_1_param = 0
        for name, param in self.prefix_encoder_source_1.named_parameters():
            source_1_param += param.numel()
        source_2_param = 0
        for name, param in self.prefix_encoder_source_2.named_parameters():
            source_2_param += param.numel()
        source_3_param = 0
        for name, param in self.prefix_encoder_source_3.named_parameters():
            source_3_param += param.numel()
        source_4_param = 0
        for name, param in self.prefix_encoder_source_4.named_parameters():
            source_4_param += param.numel()
        source_5_param = 0
        for name, param in self.prefix_encoder_source_5.named_parameters():
            source_5_param += param.numel()
        source_6_param = 0
        for name, param in self.prefix_encoder_source_6.named_parameters():
            source_6_param += param.numel()
        source_7_param = 0
        for name, param in self.prefix_encoder_source_7.named_parameters():
            source_7_param += param.numel()
        source_8_param = 0
        for name, param in self.prefix_encoder_source_8.named_parameters():
            source_8_param += param.numel()
        total_param = source_1_param + source_2_param + source_3_param + source_4_param + source_5_param + \
                      source_6_param + source_7_param + source_8_param
        print('total param is {}'.format(total_param))



    def forward(self, input_ids=None, attention_mask=None,mask_ids=None):
        batch_size = input_ids.shape[0]
        past_key_values, sum_key_values = self.get_prompt(batch_size)

        prompt_guids_length = past_key_values[0][0].shape[3]

        # attention_mask: bsz, seq_len
        # prompt attention， attention mask
        bsz = attention_mask.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.bert.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                # token_type_ids=token_type_ids,
                                past_key_values=past_key_values,
                                sum_key_values=sum_key_values,
                                gates=self.gates,
                                gates_projection=self.gates_projection,
                                return_dict=False)


        sequence_output = bert_output[0]  # bsz, len, hidden
        batch_ids = torch.arange(mask_ids.shape[0])
        first_token_tensor = sequence_output[batch_ids, mask_ids]
        pooled_output = self.bert.pooler.dense(first_token_tensor)
        pooled_output = self.bert.pooler.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.fc(pooled_output)  # bsz, len, labels
        moe_loss = bert_output[-1]
        return logits,moe_loss

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values_source_1 = self.prefix_encoder_source_1(prefix_tokens)
        past_key_values_source_1 = past_key_values_source_1.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_2 = self.prefix_encoder_source_2(prefix_tokens)
        past_key_values_source_2 = past_key_values_source_2.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_3 = self.prefix_encoder_source_3(prefix_tokens)
        past_key_values_source_3 = past_key_values_source_3.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_4 = self.prefix_encoder_source_4(prefix_tokens)
        past_key_values_source_4 = past_key_values_source_4.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_4 = self.prefix_encoder_source_4(prefix_tokens)
        past_key_values_source_4 = past_key_values_source_4.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_5 = self.prefix_encoder_source_5(prefix_tokens)
        past_key_values_source_5 = past_key_values_source_5.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_6 = self.prefix_encoder_source_6(prefix_tokens)
        past_key_values_source_6 = past_key_values_source_6.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_7 = self.prefix_encoder_source_7(prefix_tokens)
        past_key_values_source_7 = past_key_values_source_7.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values_source_8 = self.prefix_encoder_source_8(prefix_tokens)
        past_key_values_source_8 = past_key_values_source_8.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        past_key_values_Target = self.prefix_encoder_Target(prefix_tokens)
        past_key_values_Target = past_key_values_Target.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        total_key_values = torch.stack(
            [past_key_values_source_1,
             past_key_values_source_2,
             past_key_values_source_3,
             past_key_values_source_4,
             past_key_values_source_5,
             past_key_values_source_6,
             past_key_values_source_7,
             past_key_values_source_8,
             past_key_values_Target])
        sum_key_values = total_key_values.sum(0)
        total_key_values = total_key_values.permute([3, 0, 1, 4, 2, 5]).split(2)
        sum_key_values = sum_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return total_key_values, sum_key_values


class ThreeExpertsModel_flexible(nn.Module):
    def __init__(self, num_labels,model_path,pre_seq_len,num_prefix=3,ms=False):
        super(ThreeExpertsModel_flexible, self).__init__()

        # Set the length of the prompt to the length of the main image plus the length of the secondary image
        self.num_prefix_encoders = num_prefix
        self.ms = ms
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert_config = self.bert.config
        self.n_layer = self.bert_config.num_hidden_layers
        self.n_head = self.bert_config.num_attention_heads
        self.n_embd = self.bert_config.hidden_size // self.bert_config.num_attention_heads

        self.pre_seq_len = pre_seq_len

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder_Target = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512,
                                                         self.n_layer)

        self.prefix_encoder_source_list = []
        for i in range(1, self.num_prefix_encoders + 1):
            prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512, self.n_layer)
            setattr(self, f'prefix_encoder_source_{i}', prefix_encoder)
            self.prefix_encoder_source_list.append(prefix_encoder)



        # Determine the contribution of each image representation in each layer
        self.gates = nn.ModuleList([nn.Linear(20 * 768, self.num_prefix_encoders+1) for i in range(12)])
        self.gates_projection = nn.ModuleList([nn.Linear(768, 768) for i in range(12)])

        self.num_labels = num_labels
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

        total_param = 0

        # Use a loop to iterate over the prefix encoders
        for i in range(1, self.num_prefix_encoders + 1):
            source_param = 0
            current_prefix_encoder = getattr(self, f'prefix_encoder_source_{i}')
            # Use another loop to iterate over the named parameters of the current prefix encoder
            for name, param in current_prefix_encoder.named_parameters():
                source_param += param.numel()
            # Add the parameters of the current prefix encoder to the total
            total_param += source_param

        print('total param is {}'.format(total_param))


    def forward(self, input_ids=None, attention_mask=None,mask_ids=None):
        batch_size = input_ids.shape[0]
        past_key_values, sum_key_values = self.get_prompt(batch_size)

        prompt_guids_length = past_key_values[0][0].shape[3]

        # attention_mask: bsz, seq_len
        # prompt attention， attention mask
        bsz = attention_mask.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.bert.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                # token_type_ids=token_type_ids,
                                past_key_values=past_key_values,
                                sum_key_values=sum_key_values,
                                gates=self.gates,
                                gates_projection=self.gates_projection,
                                return_dict=False)

        if self.ms == False:
            sequence_output = bert_output[1]  # bsz, len, hidden
            sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
            logits = self.fc(sequence_output)  # bsz, len, labels
            moe_loss = bert_output[-1]
        else:
            sequence_output = bert_output[0]  # bsz, len, hidden
            batch_ids = torch.arange(mask_ids.shape[0])
            first_token_tensor = sequence_output[batch_ids, mask_ids]
            pooled_output = self.bert.pooler.dense(first_token_tensor)
            pooled_output = self.bert.pooler.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.fc(pooled_output)  # bsz, len, labels
            moe_loss = bert_output[-1]

        return logits,moe_loss


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)

        past_key_values_source = []

        # Use a loop to iterate over the prefix encoders
        for i in range(1, self.num_prefix_encoders + 1):
            current_prefix_encoder = getattr(self, f'prefix_encoder_source_{i}')
            past_key_values_source_i = current_prefix_encoder(prefix_tokens)
            past_key_values_source_i = past_key_values_source_i.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2,
                self.n_head,
                self.n_embd
            )
            past_key_values_source.append(past_key_values_source_i)


        past_key_values_target = self.prefix_encoder_Target(prefix_tokens)
        past_key_values_target = past_key_values_target.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        total_key_values = torch.stack(past_key_values_source + [past_key_values_target])
        sum_key_values = total_key_values.sum(0)
        total_key_values = total_key_values.permute([3, 0, 1, 4, 2, 5]).split(2)
        sum_key_values = sum_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return total_key_values, sum_key_values

class ThreeExpertsModel_flexible_WoTarget(nn.Module):
    def __init__(self, num_labels,model_path,pre_seq_len,num_prefix=3,ms=False):
        super(ThreeExpertsModel_flexible_WoTarget, self).__init__()

        # Set the length of the prompt to the length of the main image plus the length of the secondary image
        self.num_prefix_encoders = num_prefix
        self.ms = ms
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.bert_config = self.bert.config
        self.n_layer = self.bert_config.num_hidden_layers
        self.n_head = self.bert_config.num_attention_heads
        self.n_embd = self.bert_config.hidden_size // self.bert_config.num_attention_heads

        self.pre_seq_len = pre_seq_len

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()

        self.prefix_encoder_source_list = []
        for i in range(1, self.num_prefix_encoders + 1):
            prefix_encoder = PrefixEncoder_clean(self.pre_seq_len, self.bert.config.hidden_size, 512, self.n_layer)
            setattr(self, f'prefix_encoder_source_{i}', prefix_encoder)
            self.prefix_encoder_source_list.append(prefix_encoder)


        # Determine the contribution of each image representation in each layer
        self.gates = nn.ModuleList([nn.Linear(20 * 768, self.num_prefix_encoders) for i in range(12)])
        self.gates_projection = nn.ModuleList([nn.Linear(768, 768) for i in range(12)])

        self.num_labels = num_labels
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

        total_param = 0

        # Use a loop to iterate over the prefix encoders
        for i in range(1, self.num_prefix_encoders + 1):
            source_param = 0
            current_prefix_encoder = getattr(self, f'prefix_encoder_source_{i}')
            # Use another loop to iterate over the named parameters of the current prefix encoder
            for name, param in current_prefix_encoder.named_parameters():
                source_param += param.numel()
            # Add the parameters of the current prefix encoder to the total
            total_param += source_param

        print('total param is {}'.format(total_param))


    def forward(self, input_ids=None, attention_mask=None,mask_ids=None):
        batch_size = input_ids.shape[0]
        past_key_values, sum_key_values = self.get_prompt(batch_size)

        prompt_guids_length = past_key_values[0][0].shape[3]

        # attention_mask: bsz, seq_len
        # prompt attention， attention mask
        bsz = attention_mask.size(0)
        prompt_guids_mask = torch.ones((bsz, prompt_guids_length)).to(self.bert.device)
        prompt_attention_mask = torch.cat((prompt_guids_mask, attention_mask), dim=1)

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                # token_type_ids=token_type_ids,
                                past_key_values=past_key_values,
                                sum_key_values=sum_key_values,
                                gates=self.gates,
                                gates_projection=self.gates_projection,
                                return_dict=False)

        if self.ms == False:
            sequence_output = bert_output[1]  # bsz, len, hidden
            sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
            logits = self.fc(sequence_output)  # bsz, len, labels
            moe_loss = bert_output[-1]
        else:
            sequence_output = bert_output[0]  # bsz, len, hidden
            batch_ids = torch.arange(mask_ids.shape[0])
            first_token_tensor = sequence_output[batch_ids, mask_ids]
            pooled_output = self.bert.pooler.dense(first_token_tensor)
            pooled_output = self.bert.pooler.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.fc(pooled_output)  # bsz, len, labels
            moe_loss = bert_output[-1]

        return logits,moe_loss


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)

        past_key_values_source = []

        # Use a loop to iterate over the prefix encoders
        for i in range(1, self.num_prefix_encoders + 1):
            current_prefix_encoder = getattr(self, f'prefix_encoder_source_{i}')
            past_key_values_source_i = current_prefix_encoder(prefix_tokens)
            past_key_values_source_i = past_key_values_source_i.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2,
                self.n_head,
                self.n_embd
            )
            past_key_values_source.append(past_key_values_source_i)



        total_key_values = torch.stack(past_key_values_source)
        sum_key_values = total_key_values.sum(0)
        total_key_values = total_key_values.permute([3, 0, 1, 4, 2, 5]).split(2)
        sum_key_values = sum_key_values.permute([2, 0, 3, 1, 4]).split(2)

        return total_key_values, sum_key_values

if __name__ == '__main__':
    model_path = '/home/zmj/localmodel/bert-base-uncased'
    model = ThreeExpertsModel(2, model_path, pre_seq_len=20)