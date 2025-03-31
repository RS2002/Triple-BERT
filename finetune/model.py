import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, layer_sizes=[64,64,64,1], arl=False, dropout=0.0, bias = True):
        super().__init__()
        self.arl = arl
        if self.arl:
            self.attention = nn.Sequential(
                nn.Linear(layer_sizes[0],layer_sizes[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(layer_sizes[0],layer_sizes[0])
            )

        self.layer_sizes = layer_sizes
        if len(layer_sizes) < 2:
            raise ValueError()
        self.layers = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias = bias))

    def forward(self, x):
        if self.arl:
            x = x * self.attention(x)
        for layer in self.layers[:-1]:
            x = self.dropout(self.act(layer(x)))
        x = self.layers[-1](x)
        return x

class LSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = MLP([hidden_size, hidden_size, output_size], dropout=dropout)

    def forward(self,x,valid_index=None):
        self.lstm.flatten_parameters()
        x,_ = self.lstm(x)
        x = torch.concat([torch.zeros(x.shape[0],1,x.shape[2]).to(x.device),x],dim=1)
        if valid_index is not None:
            x = x[torch.arange(x.size(0)),valid_index]
        else:
            x = x[:, -1, :]
        x = self.fc(x)
        return x

class BiLSTM(nn.Module):
    def __init__(self,input_size, output_size, hidden_size=64, dropout=0.0):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.lstm2 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = MLP([hidden_size * 2, hidden_size, output_size], dropout=dropout)


    def forward(self,x,valid_index=None):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x1,_ = self.lstm1(x)
        x1 = torch.concat([torch.zeros(x1.shape[0],1,x1.shape[2]).to(x1.device),x1],dim=1)
        if valid_index is not None:
            x1 = x1[torch.arange(x1.size(0)),valid_index]
            x2 = x.clone()
            for i in range(x2.shape[0]):
                end_index = valid_index[i].item()
                x2[i, :end_index] = x[i, :end_index].flip(dims=(0,))
        else:
            x1 = x1[:, -1, :]
            x2 = torch.flip(x, [1])
        x2, _ = self.lstm2(x2)
        x2 = torch.concat([torch.zeros(x2.shape[0],1,x2.shape[2]).to(x2.device),x2],dim=1)
        if valid_index is not None:
            x2 = x2[torch.arange(x2.size(0)),valid_index]
        else:
            x2 = x2[:, -1, :]
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc(x)
        return x


class Worker_Net(nn.Module):
    def __init__(self, state_size=7, order_size=4, output_dim=32, bi_direction=False, dropout=0.0):
        super().__init__()
        if bi_direction:
            self.lstm = BiLSTM(order_size, output_dim, dropout=dropout)
        else:
            self.lstm = LSTM(order_size, output_dim, dropout=dropout)
        self.encode = MLP([state_size, output_dim, output_dim], arl=True, dropout=dropout)
        self.mlp = MLP([output_dim * 2, output_dim, output_dim], dropout=dropout)

    def forward(self,x_state,x_order,order_num=None):
        x_order = self.lstm(x_order,order_num)
        x_state = self.encode(x_state)
        y = self.mlp(torch.concat([x_state,x_order],dim=-1))
        return y

class Order_Net(nn.Module):
    def __init__(self, state_size, output_size=64, dropout=0.0):
        super().__init__()
        self.model = MLP([state_size, output_size//2, output_size], arl=True, dropout=dropout)

    def forward(self,x):
        y = self.model(x)
        return y

class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        super().__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat


class Sequence_Classifier(nn.Module):
    def __init__(self, class_num=1, hs=512, da=512, r=8):
        super().__init__()
        self.attention = SelfAttention(hs, da, r)
        self.classifier = MLP([hs * r, (hs * r + class_num)// 2, class_num])

    def forward(self, x):
        attn_mat = self.attention(x)
        m = torch.bmm(attn_mat, x)
        flatten = m.view(m.size()[0], -1)
        res = self.classifier(flatten)
        return res

class QK_Attention(nn.Module):
    def __init__(self, input_dims=64, hidden_dims=64, head=1, dropout=0.0, method="mean"):
        super().__init__()
        # self.q_emb = MLP([input_dims,hidden_dims,1])
        # self.k_emb = MLP([input_dims,hidden_dims,1])
        self.q_linear = nn.ModuleList()
        self.k_linear = nn.ModuleList()
        for i in range(int(head)):
            self.q_linear.append(MLP([input_dims,hidden_dims*2,hidden_dims*4], dropout=dropout))
            self.k_linear.append(MLP([input_dims,hidden_dims*2,hidden_dims*4], dropout=dropout))
        self.head = head
        self.method = method
        # self.softplus = nn.Softplus()
        # self.relu = nn.ReLU()
        if self.method != "mean":
            self.fuse_layer = MLP([head,head,1], dropout=dropout)

    def forward(self,q,k):
        attn_matrix = None

        # q_result = self.q_emb(q)
        # k_result = self.k_emb(k)
        # k_result = k_result.T
        # q_result = q_result.expand(-1,k.shape[0])
        # k_result = k_result.expand(q.shape[0],-1)

        for i in range(self.head):
            query=self.q_linear[i](q)
            key=self.k_linear[i](k)

            # key = self.softplus(key)
            key = key ** 2
            # key = self.relu(key)
            norms = torch.norm(key, dim=1, keepdim=True) + 1e-8
            key = key / norms
            # key = torch.abs(key)

            attn = torch.mm(query,key.T)
            if self.head == 1:
                return attn
                # return attn + q_result + k_result

            attn = attn.unsqueeze(-1)
            if attn_matrix is None:
                attn_matrix = attn
            else:
                attn_matrix = torch.concat([attn_matrix, attn],dim=-1)

        if self.method == "mean":
            attn_matrix = torch.mean(attn_matrix,dim=-1)
        else:
            attn_matrix = self.fuse_layer(attn_matrix)
            attn_matrix = attn_matrix.squeeze(-1)
        return attn_matrix
        # return attn_matrix + q_result + k_result


class Assignment_Net(nn.Module):
    def __init__(self, state_size=7, history_order_size=4, current_order_size=5, hidden_dim=64, bi_direction=False, dropout=0.0):
        super().__init__()
        self.worker_net = Worker_Net(state_size=state_size, order_size=history_order_size, output_dim=hidden_dim, bi_direction=bi_direction, dropout=dropout)
        self.order_net = Order_Net(state_size=current_order_size, output_size=hidden_dim, dropout=dropout)
        self.attention = QK_Attention(input_dims=hidden_dim, hidden_dims=hidden_dim, head=1, dropout=dropout)

    def forward(self, order, x_state, x_order, order_num=None):
        worker, order = self.encode(order, x_state, x_order, order_num)
        q_matrix = self.attention(worker,order)
        return q_matrix

    def encode(self, order, x_state, x_order, order_num=None):
        if order_num is not None:
            order_num = order_num.int()
        order = order.float()
        x_state = x_state.float()
        x_order = x_order.float()
        order = self.order_net(order)
        worker = self.worker_net(x_state,x_order,order_num)
        return worker, order

class AC_BERT(nn.Module):
    def __init__(self, bertconfig_actor, bertconfig_critic, state_size=7, history_order_size=4, current_order_size=5, hidden_dim=64, agent_num=1000, bi_direction=False, dropout=0.0):
        super().__init__()

        self.assignment_net = Assignment_Net(state_size, history_order_size, current_order_size, hidden_dim, bi_direction, dropout)
        self.attention = self.assignment_net.attention

        self.bert_actor = BertModel(bertconfig_actor)

        self.bert_critic1 = BertModel(bertconfig_critic)
        self.critic1 = Sequence_Classifier(class_num=1, hs=hidden_dim*2, da=hidden_dim*2, r=4)

        self.bert_critic2 = BertModel(bertconfig_critic)
        self.critic2 = Sequence_Classifier(class_num=1, hs=hidden_dim*2, da=hidden_dim*2, r=4)

        self.softmax = nn.Softmax(dim=-1)

    def pretrain(self, order, x_state, x_order, order_num=None):
        return self.assignment_net(order, x_state, x_order, order_num)

    def encode(self, order, x_state, x_order, order_num=None):
        worker, order = self.assignment_net.encode(order, x_state, x_order, order_num)
        x = torch.concat([worker, order], dim=0).unsqueeze(0)

        # x = x.detach()

        x_emb = self.bert_actor(inputs_embeds=x, attention_mask=None, output_hidden_states=False)
        x_emb = x_emb.last_hidden_state
        return x_emb

    def act(self, order, x_state, x_order, order_num=None, output_prob=False):
        x_emb = self.encode(order, x_state, x_order, order_num)

        # x_emb2 = x_emb.clone().detach()
        # worker = x_emb2[0, :x_state.shape[0], :]
        # order = x_emb2[0, x_state.shape[0]:, :]

        worker = x_emb[0, :x_state.shape[0], :]
        order = x_emb[0, x_state.shape[0]:, :]

        p_matrix = self.attention(worker,order)
        p_matrix = self.softmax(p_matrix)

        if output_prob:
            return torch.log(p_matrix), x_emb, p_matrix
        else:
            return torch.log(p_matrix), x_emb


    def criticize(self, x_emb, action):
        worker, order = x_emb[0, :action.shape[0], :], x_emb[0, action.shape[0]:, :]
        valid_indices = (action != -1)
        worker = worker[valid_indices]
        action = action[valid_indices]
        order = order[action]
        x = torch.concat([worker,order],dim=-1).unsqueeze(0)

        # x = x.detach()

        x_emb2_1 = self.bert_critic1(inputs_embeds=x, attention_mask=None, output_hidden_states=False)
        x_emb2_1 = x_emb2_1.last_hidden_state
        q_value_1 = self.critic1(x_emb2_1)

        x_emb2_2 = self.bert_critic2(inputs_embeds=x, attention_mask=None, output_hidden_states=False)
        x_emb2_2 = x_emb2_2.last_hidden_state
        q_value_2 = self.critic2(x_emb2_2)

        return [q_value_1,q_value_2]

    def forward(self,order,x_state,x_order,action,order_num=None):
        p_matrix, x_emb = self.act(order, x_state, x_order, order_num)
        valid_indices = (action != -1)
        selected_elements = p_matrix[valid_indices, action[valid_indices]]

        log_prob = selected_elements.sum()
        # log_prob = selected_elements.mean()

        q_value = self.criticize(x_emb,action)

        return p_matrix, log_prob, q_value, x_emb
