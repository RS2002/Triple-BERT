import numpy as np
import torch
from model import AC_BERT
from joblib import Parallel, delayed
import torch.nn as nn
import tqdm
import pickle
from transformers import BertConfig
from Platform import assign

INF = 1e8

class Buffer():
    def __init__(self,capacity = 1e5, episode_capacity = 10):
        super().__init__()
        self.reset(capacity, episode_capacity)

    def reset(self, capacity = None, episode_capacity = None):
        if capacity is not None:
            self.capacity = capacity
            self.episode_capacity = episode_capacity

        self.num = 0

        self.worker_state = []
        self.order_state = []
        self.order_num = []
        self.pooling_order = []
        self.action = []

        self.worker_state_next = []
        self.order_state_next = []
        self.order_num_next = []
        self.pooling_order_next = []
        self.action_next = []

        self.reward = []

        self.episode = []

    def append(self, experience, episode=0):
        if self.num > 0 and self.episode[0]<episode-self.episode_capacity:
            episode_np = np.array(self.episode)
            old_record_num = len(episode_np[episode_np<(episode-self.episode_capacity)])
            self.num -= old_record_num
            self.worker_state = self.worker_state[old_record_num:]
            self.order_state = self.order_state[old_record_num:]
            self.order_num = self.order_num[old_record_num:]
            self.pooling_order = self.pooling_order[old_record_num:]
            self.action = self.action[old_record_num:]
            self.worker_state_next = self.worker_state_next[old_record_num:]
            self.order_state_next = self.order_state_next[old_record_num:]
            self.order_num_next = self.order_num_next[old_record_num:]
            self.pooling_order_next = self.pooling_order_next[old_record_num:]
            self.action_next = self.action_next[old_record_num:]
            self.reward = self.reward[old_record_num:]
            self.episode = self.episode[old_record_num:]
            if self.episode[0]<episode-self.episode_capacity:
                print("Buffer Error!")
                exit(-1)
        if self.num == self.capacity:
            self.worker_state = self.worker_state[1:]
            self.order_state = self.order_state[1:]
            self.order_num = self.order_num[1:]
            self.pooling_order = self.pooling_order[1:]
            self.action = self.action[1:]
            self.worker_state_next = self.worker_state_next[1:]
            self.order_state_next = self.order_state_next[1:]
            self.order_num_next = self.order_num_next[1:]
            self.pooling_order_next = self.pooling_order_next[1:]
            self.action_next = self.action_next[1:]
            self.reward = self.reward[1:]
            self.episode = self.episode[1:]
        else:
            self.num+=1

        state, action, reward, state_next, action_next = experience

        self.worker_state.append(torch.from_numpy(state[0]))
        self.order_state.append(torch.from_numpy(state[1]))
        self.order_num.append(torch.from_numpy(state[2]))
        self.pooling_order.append(torch.from_numpy(state[3]))
        self.action.append(torch.tensor(action))
        if action_next is not None:
            self.worker_state_next.append(torch.from_numpy(state_next[0]))
            self.order_state_next.append(torch.from_numpy(state_next[1]))
            self.order_num_next.append(torch.from_numpy(state_next[2]))
            self.pooling_order_next.append(torch.from_numpy(state_next[3]))
            self.action_next.append(torch.tensor(action_next))
        else:
            self.worker_state_next.append(state_next[0])
            self.order_state_next.append(state_next[1])
            self.order_num_next.append(state_next[2])
            self.pooling_order_next.append(state_next[3])
            self.action_next.append(action_next)
        self.reward.append(reward)
        self.episode.append(episode)


    def sample(self,size):
        if size>self.num:
            size = self.num

        indices = np.random.randint(0, self.num, size=size)
        # priority = np.array(self.episode)
        # priority = priority - np.min(priority) + 1
        # probabilities = np.array(priority) / np.sum(priority)
        # indices = np.random.choice(self.num, size, p=probabilities)

        worker_state = [self.worker_state[i] for i in indices]
        order_state = [self.order_state[i] for i in indices]
        order_num = [self.order_num[i] for i in indices]
        pool_order = [self.pooling_order[i] for i in indices]
        action = [self.action[i] for i in indices]
        worker_state_next = [self.worker_state[i] for i in indices]
        order_state_next = [self.order_state[i] for i in indices]
        order_num_next = [self.order_num[i] for i in indices]
        pool_order_next = [self.pooling_order_next[i] for i in indices]
        action_next = [self.action_next[i] for i in indices]
        reward = [self.reward[i] for i in indices]

        return worker_state, order_state, order_num, pool_order, action, reward, worker_state_next, order_state_next, order_num_next, pool_order_next, action_next

def norm(order_state, worker_state, history_order_state, lat_min = 40.68878421555262, lat_max = 40.875967791801536, lon_min = -74.04528828347375, lon_max = -73.91037864632285, simulation_time = 60, max_capacity = 3):
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    if isinstance(order_state, torch.Tensor):
        worker_state, history_order_state, order_state = worker_state.clone(), history_order_state.clone(), order_state.clone()
    else:
        worker_state, history_order_state, order_state = worker_state.copy(), history_order_state.copy(), order_state.copy()

    # 1. lat & lon
    order_state[:,0] = (order_state[:,0] - lat_min) / lat_range
    order_state[:,2] = (order_state[:,2] - lat_min) / lat_range
    order_state[:,1] = (order_state[:,1] - lon_min) / lon_range
    order_state[:,3] = (order_state[:,3] - lon_min) / lon_range

    worker_state[:,0] = (worker_state[:,0] - lat_min) / lat_range
    worker_state[:,1] = (worker_state[:,1] - lon_min) / lon_range


    history_order_state[:,:,0] = (history_order_state[:,:,0] - lat_min) / lat_range * (history_order_state[:,:,0] != 0)
    history_order_state[:,:,1] = (history_order_state[:,:,1] - lon_range) / lon_range * (history_order_state[:,:,1] != 0)

    # 2. time
    worker_state[:, 3] = worker_state[:, 3] / simulation_time
    worker_state[:, 5] = worker_state[:, 5] / simulation_time
    order_state[:,4] = order_state[:,4] / simulation_time

    history_order_state[:,:,2] = history_order_state[:,:,2] / simulation_time
    history_order_state[:,:,3] = history_order_state[:,:,3] / simulation_time
    history_order_state[:,:,4] = history_order_state[:,:,4] / simulation_time

    # 3. capacity
    worker_state[:, 2] = worker_state[:, 2] / max_capacity

    return order_state, worker_state, history_order_state

class Worker():
    def __init__(self, buffer, lr=0.0001, gamma=0.99, max_step=60, num=1000, device=None, zone_table_path = "./data/Manhattan_dic.pkl", model_path = None, njobs = 24, bi_direction = True, dropout = 0.0, compression = False):
        super().__init__()
        self.buffer = buffer

        self.gamma = gamma
        self.device = device
        self.max_step = max_step
        self.num = num

        with open(zone_table_path, 'rb') as f:
            self.zone_dic = pickle.load(f)
        # self.zone_lookup = self.zone_dic["zone_num"]
        self.coordinate_lookup_lat = np.array(self.zone_dic["centroid_lat"])
        self.coordinate_lookup_lon = np.array(self.zone_dic["centroid_lon"])
        self.zone_map = np.array(self.zone_dic["map"])

        if compression:
            max_len = 2200
        else:
            max_len = 1500

        bertconfig_actor = BertConfig(max_position_embeddings=max_len, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, position_embedding_type="none")
        bertconfig_critic = BertConfig(max_position_embeddings=1000, hidden_size=128, num_hidden_layers=4, num_attention_heads=4, position_embedding_type="none")

        self.AC_training = AC_BERT(bertconfig_actor, bertconfig_critic, state_size=6, history_order_size=5, current_order_size=5, hidden_dim=64, agent_num=1000, bi_direction=bi_direction, dropout=dropout).to(device)
        self.AC_target = AC_BERT(bertconfig_actor, bertconfig_critic, state_size=6, history_order_size=5, current_order_size=5, hidden_dim=64, agent_num=1000, bi_direction=bi_direction, dropout=dropout).to(device)

        self.load(model_path,self.device)
        for param in self.AC_target.parameters():
            param.requires_grad = False
        self.AC_target.eval()
        print('Platform total parameters:', sum(p.numel() for p in self.AC_training.parameters() if p.requires_grad))
        self.update_target(tau=1.0)

        self.optim = torch.optim.Adam(self.AC_training.parameters(), lr=lr, weight_decay=0)
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)
        self.njobs = njobs

        self.reset()

    def save(self, path):
        torch.save(self.AC_training.state_dict(), path)

    def load(self, path1=None, device=torch.device("cpu")):
        if device == torch.device("cpu"):
            if path1 is not None:
                self.AC_target.load_state_dict(torch.load(path1, map_location=torch.device('cpu')))
                self.AC_training.load_state_dict(torch.load(path1, map_location=torch.device('cpu')))
        else:
            if path1 is not None:
                self.AC_target.load_state_dict(torch.load(path1))
                self.AC_training.load_state_dict(torch.load(path1))

    def update_target(self, tau=0.005):
        for target_param, train_param in zip(self.AC_target.parameters(), self.AC_training.parameters()):
            target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)

    def reset(self, capacity = 3, train=True):
        if train:
            self.AC_training.train()
        else:
            self.AC_training.eval()
        torch.set_grad_enabled(False)

        self.is_train = train

        '''
        observation space
        0,1: current lat,lon (required to be normalized before inputting to the network, following lat and lon remain same)
        2: remaining order place
        3: remaining picking time
        4: state -- 0 allows to pick up new orders, 1 does not (because picking up the order that doesn't allow pooling or the capacity is full)
        5: current time
        '''
        self.observe_space = np.zeros([self.num, 6])
        self.observe_space[:,2] = capacity

        '''
        current orders
        0,1: drop-off lat,lon
        2: remaining transportation time (approximated)
        3: total transportation time (approximated)
        4: detour time (current)
        '''
        self.current_orders = np.zeros([self.num, capacity, 5])
        self.current_order_num = np.zeros([self.num])

        # allocate a initial location randomly from valid zone
        random_integers = np.random.randint(0, len(self.coordinate_lookup_lat), size=(self.num))
        self.observe_space[:, 0] = self.coordinate_lookup_lat[random_integers]
        self.observe_space[:, 1] = self.coordinate_lookup_lon[random_integers]

        # some records for simulation
        self.travel_route = [[] for _ in range(self.num)]
        self.travel_time = [[] for _ in range(self.num)]
        self.experience = []
        self.Pass_Travel_Time = []
        self.Detour_Time = []


    def observe(self, order, current_time, exploration_rate=0):
        pid = order['PULocationID']
        did = order['DOLocationID']
        pid = self.zone_map[pid - 1]
        did = self.zone_map[did - 1]
        minute = order['minute']
        plat, plon = self.coordinate_lookup_lat[pid], self.coordinate_lookup_lon[pid]
        dlat, dlon = self.coordinate_lookup_lat[did], self.coordinate_lookup_lon[did]
        minute = np.array(minute).reshape(-1, 1)
        plat = np.array(plat).reshape(-1, 1)
        plon = np.array(plon).reshape(-1, 1)
        dlat = np.array(dlat).reshape(-1, 1)
        dlon = np.array(dlon).reshape(-1, 1)
        order = np.concatenate([plat, plon, dlat, dlon, minute], axis=-1)

        self.observe_space[:,5] = current_time

        # 1. calculate q-value
        x1, x2, x3 = norm(order, self.observe_space, self.current_orders)
        x1, x2, x3 = torch.tensor(x1).to(self.device), torch.tensor(x2).to(self.device), torch.tensor(x3).to(self.device)
        q_value, _ = self.AC_training.act(x1, x2, x3, torch.from_numpy(self.current_order_num).to(self.device))
        # 2. epsilon-greedy explore
        exploration_matrix = torch.rand_like(q_value)
        q_value[exploration_matrix < exploration_rate] = INF
        q_value[self.observe_space[:, 4] == 1] = -INF

        return q_value.cpu().detach().numpy(), order

    # def train(self,batch_size=16, train_times=10, show_pbar=True):
    #     torch.set_grad_enabled(True)
    #     self.AC_training.train()
    #     if show_pbar:
    #         pbar = tqdm.tqdm(range(train_times))
    #     else:
    #         pbar = range(train_times)
    #
    #     loss_list = []
    #     for _ in pbar:
    #         loss = None
    #         worker_state, order_state, order_num, pool_order, action, reward, worker_state_next, order_state_next, order_num_next, pool_order_next, action_next = self.buffer.sample(batch_size)
    #         for i in range(len(worker_state)):
    #             worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp, reward_temp, worker_state_next_temp, order_state_next_temp, order_num_next_temp, pool_order_next_temp, action_next_temp = worker_state[i], order_state[i], order_num[i], pool_order[i], action[i], reward[i], worker_state_next[i], order_state_next[i], order_num_next[i], pool_order_next[i], action_next[i]
    #             if torch.all(action_temp == -1):
    #                 continue
    #             if action_next_temp is not None:
    #                 if torch.all(action_next_temp == -1):
    #                     continue
    #                 worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp, worker_state_next_temp, order_state_next_temp, order_num_next_temp, pool_order_next_temp, action_next_temp = worker_state_temp.to(self.device), order_state_temp.to(self.device), order_num_temp.to(self.device), pool_order_temp.to(self.device), action_temp.to(self.device), worker_state_next_temp.to(self.device), order_state_next_temp.to(self.device), order_num_next_temp.to(self.device), pool_order_next_temp.to(self.device), action_next_temp.to(self.device)
    #                 x1, x2, x3 = norm(pool_order_temp, worker_state_temp, order_state_temp)
    #                 x1_next, x2_next, x3_next = norm(pool_order_next_temp, worker_state_next_temp, order_state_next_temp)
    #
    #                 log_prob, q_value = self.AC_training(x1, x2, x3, action_temp, order_num_temp)
    #                 # print(action_next_temp)
    #                 log_prob_next, q_value_next1 = self.AC_training(x1_next, x2_next, x3_next, action_next_temp, order_num_next_temp)
    #                 _, q_value_next2 = self.AC_target(x1_next, x2_next, x3_next, action_next_temp, order_num_next_temp)
    #                 q_value_next = torch.min(q_value_next1,q_value_next2).detach()
    #                 target = self.gamma * q_value_next + reward_temp
    #
    #                 loss_critic = (target - q_value)**2
    #                 # loss_actor = (-torch.log(prob)*q_value.detach()-torch.log(prob_next)*q_value_next1.detach())/2
    #                 loss_actor = (-log_prob*q_value.detach()-log_prob_next*q_value_next1.detach())/2
    #                 if loss is None:
    #                     loss = loss_actor + loss_critic
    #                 else:
    #                      loss += loss_actor + loss_critic
    #             else:
    #                 worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp= worker_state_temp.to(self.device), order_state_temp.to(self.device), order_num_temp.to(self.device), pool_order_temp.to(self.device), action_temp.to(self.device)
    #                 x1, x2, x3 = norm(pool_order_temp, worker_state_temp, order_state_temp)
    #                 log_prob, q_value = self.AC_training(x1, x2, x3, action_temp, order_num_temp)
    #                 target = reward_temp
    #                 loss_critic = (target - q_value)**2
    #                 # loss_actor = -torch.log(prob)*q_value.detach()
    #                 loss_actor = -log_prob*q_value.detach()
    #                 if loss is None:
    #                     loss = loss_actor + loss_critic
    #                 else:
    #                      loss += loss_actor + loss_critic
    #         if loss is not None:
    #             loss /= batch_size
    #             self.optim.zero_grad()
    #             loss.backward()
    #
    #             has_nan = False
    #             for name, param in self.AC_training.named_parameters():
    #                 if param.grad is not None:
    #                     if torch.isnan(param.grad).any():
    #                         has_nan = True
    #                         break
    #             if has_nan:
    #                 # print("NAN Gradient->Skip")
    #                 continue
    #             torch.nn.utils.clip_grad_norm_(self.AC_training.parameters(), 1.0)  # avoid gradient explosion
    #
    #             self.optim.step()
    #             loss_list.append(loss.item())
    #
    #     self.update_target()
    #     return np.mean(loss_list)


    def train(self, batch_size=8, train_times=1, show_pbar=False, train_actor=False, train_critic=True):
        actor_rate = 1.0

        torch.set_grad_enabled(True)
        self.AC_training.train()
        if show_pbar:
            pbar = tqdm.tqdm(range(train_times))
        else:
            pbar = range(train_times)

        loss_list = []
        for _ in pbar:
            loss = None
            worker_state, order_state, order_num, pool_order, action, reward, worker_state_next, order_state_next, order_num_next, pool_order_next, action_next = self.buffer.sample(batch_size)
            for i in range(len(worker_state)):
                worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp, reward_temp, worker_state_next_temp, order_state_next_temp, order_num_next_temp, pool_order_next_temp, action_next_temp = worker_state[i], order_state[i], order_num[i], pool_order[i], action[i], reward[i], worker_state_next[i], order_state_next[i], order_num_next[i], pool_order_next[i], action_next[i]
                if torch.all(action_temp == -1):
                    continue
                if action_next_temp is not None:
                    if train_critic and torch.all(action_next_temp == -1):
                        continue
                    worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp, worker_state_next_temp, order_state_next_temp, order_num_next_temp, pool_order_next_temp, action_next_temp = worker_state_temp.to(self.device), order_state_temp.to(self.device), order_num_temp.to(self.device), pool_order_temp.to(self.device), action_temp.to(self.device), worker_state_next_temp.to(self.device), order_state_next_temp.to(self.device), order_num_next_temp.to(self.device), pool_order_next_temp.to(self.device), action_next_temp.to(self.device)
                    x1, x2, x3 = norm(pool_order_temp, worker_state_temp, order_state_temp)

                    p_matrix, _, q_value, x_emb = self.AC_training(x1, x2, x3, action_temp, order_num_temp)

                    if train_critic:
                        # _, _, q_value_next1, _ = self.AC_training(x1_next, x2_next, x3_next, action_next_temp, order_num_next_temp)
                        # _, _, q_value_next2, _ = self.AC_target(x1_next, x2_next, x3_next, action_next_temp, order_num_next_temp)
                        # q_value_next = torch.min(q_value_next1,q_value_next2).detach()
                        # target = self.gamma * q_value_next + reward_temp
                        # loss_critic = (target - q_value)**2

                        x1_next, x2_next, x3_next = norm(pool_order_next_temp, worker_state_next_temp, order_state_next_temp)
                        p_matrix1, _, _, x_emb1 = self.AC_training(x1_next, x2_next, x3_next, action_next_temp, order_num_next_temp)
                        p_matrix2, _, _, x_emb2 = self.AC_target(x1_next, x2_next, x3_next, action_next_temp, order_num_next_temp)

                        # action_new, _ = assign(p_matrix1.cpu().detach().numpy())
                        # action_new = [-1 if x is None else x for x in action_new]
                        # action_new = torch.tensor(action_new).to(self.device)
                        # q_value_next1 = self.AC_training.criticize(x_emb1, action_new)
                        # log_prob1 = self.AC_training.act_emb(x_emb1, action_new)
                        # action_new, _ = assign(p_matrix2.cpu().detach().numpy())
                        # action_new = [-1 if x is None else x for x in action_new]
                        # action_new = torch.tensor(action_new).to(self.device)
                        # q_value_next2 = self.AC_target.criticize(x_emb2, action_new)

                        action_new1, _ = assign(p_matrix1.cpu().detach().numpy())
                        action_new1 = [-1 if x is None else x for x in action_new1]
                        action_new1 = torch.tensor(action_new1).to(self.device)
                        action_new2, _ = assign(p_matrix2.cpu().detach().numpy())
                        action_new2 = [-1 if x is None else x for x in action_new2]
                        action_new2 = torch.tensor(action_new2).to(self.device)
                        if torch.all(action_new1 == -1) and torch.all(action_new2 == -1):
                            continue
                        q_value_next1 = self.AC_training.criticize(x_emb1, action_new1)
                        q_value_next2 = self.AC_target.criticize(x_emb2, action_new2)


                        q_value_next = torch.min(q_value_next1,q_value_next2).detach()
                        target = self.gamma * q_value_next + reward_temp
                        loss_critic = (target - q_value)**2
                    else:
                        loss_critic = 0

                    if train_actor:
                        action_new, _ = assign(p_matrix.cpu().detach().numpy())
                        action_new = [-1 if x is None else x for x in action_new]
                        action_new = torch.tensor(action_new).to(self.device)
                        valid_indices = (action_new != -1)
                        selected_elements = p_matrix[valid_indices, action_new[valid_indices]]

                        # log_prob = selected_elements.sum()
                        log_prob = selected_elements.mean()

                        q_new = self.AC_training.criticize(x_emb, action_new)
                        # loss_actor = - log_prob * (q_new.detach() - q_value.detach())
                        loss_actor = - log_prob * q_new.detach()
                        # loss_actor += -log_prob1 * q_value_next1.detach()
                    else:
                        loss_actor = 0

                else:
                    worker_state_temp, order_state_temp, order_num_temp, pool_order_temp, action_temp= worker_state_temp.to(self.device), order_state_temp.to(self.device), order_num_temp.to(self.device), pool_order_temp.to(self.device), action_temp.to(self.device)
                    x1, x2, x3 = norm(pool_order_temp, worker_state_temp, order_state_temp)
                    p_matrix, _, q_value, x_emb = self.AC_training(x1, x2, x3, action_temp, order_num_temp)

                    if train_critic:
                        target = reward_temp
                        loss_critic = (target - q_value)**2
                    else:
                        loss_critic = 0

                    if train_actor:
                        action_new, _ = assign(p_matrix.cpu().detach().numpy())
                        action_new = [-1 if x is None else x for x in action_new]
                        action_new = torch.tensor(action_new).to(self.device)
                        valid_indices = (action_new != -1)
                        selected_elements = p_matrix[valid_indices, action_new[valid_indices]]

                        # log_prob = selected_elements.sum()
                        log_prob = selected_elements.mean()

                        q_new = self.AC_training.criticize(x_emb, action_new)
                        # loss_actor = - log_prob * (q_new.detach() - q_value.detach())
                        loss_actor = - log_prob * q_new.detach()
                    else:
                        loss_actor = 0

                if loss is None:
                    loss = loss_actor * actor_rate + loss_critic
                else:
                    loss += loss_actor * actor_rate+ loss_critic

            if loss is not None:
                loss /= batch_size
                self.optim.zero_grad()
                loss.backward()

                has_nan = False
                for name, param in self.AC_training.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            has_nan = True
                            break
                if has_nan:
                    # print("NAN Gradient->Skip")
                    continue
                torch.nn.utils.clip_grad_norm_(self.AC_training.parameters(), 1.0)  # avoid gradient explosion

                self.optim.step()
                loss_list.append(loss.item())

        if train_actor:
            self.update_target()
        # self.schedule.step()
        torch.set_grad_enabled(False)
        return np.mean(loss_list)

    def update(self, feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, new_detour_table, reward, assignment_table, assignment_state, final_step=False, episode=1):
        # update each worker state parallely
        results = Parallel(n_jobs=self.njobs)(
            delayed(single_update)(self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.travel_route[i], self.travel_time[i], feedback_table[i], new_route_table[i], new_route_time_table[i], new_remaining_time_table[i], new_total_travel_time_table[i], new_detour_table[i])
            for i in range(self.num))
        if self.train:
            assignment_table = [-1 if x is None else x for x in assignment_table]
            state = [self.observe_space.copy(),self.current_orders.copy(),self.current_order_num.copy(),assignment_state]
            action = assignment_table
            self.experience.append(state)
            self.experience.append(action)
            if len(self.experience) == 5:
                self.buffer.append(self.experience,episode)
                self.experience = [state, action, reward]
            else:
                self.experience.append(reward)
            if final_step:
                self.experience.append([None,None,None,None])
                self.experience.append(None)
                self.buffer.append(self.experience,episode)

        for i in range(len(results)):
            self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.travel_route[i], \
            self.travel_time[i] = results[i][0], results[i][1], results[i][2], results[i][3], results[i][4]
            if results[i][5] is not None:
                self.Pass_Travel_Time.extend(results[i][5].tolist())
                self.Detour_Time.extend(results[i][6].tolist())


def single_update(observe_space, current_orders, current_orders_num, current_travel_route, current_travel_time, feedback, new_route ,new_route_time, new_remaining_time, new_total_travel_time, new_detour_table):
    finished_order_time = None
    finished_order_detour = None
    current_orders_num = int(current_orders_num)

    if feedback is not None:
        new_order_state = feedback[0][3]
        pickup_time = feedback[2]

        # update state
        observe_space[0] = new_order_state[0]  # plat
        observe_space[1] = new_order_state[1]  # plon
        observe_space[2] -= 1  # remaining seat
        observe_space[3] = pickup_time  # pickup time
        observe_space[4] = 1  # update to picking up state
        current_travel_route, current_travel_time = new_route, new_route_time
        current_orders[:current_orders_num + 1, 2], current_orders[:current_orders_num + 1, 3], current_orders[:current_orders_num + 1, 4] = new_remaining_time, new_total_travel_time, new_detour_table
        current_orders[current_orders_num, 0], current_orders[current_orders_num, 1] = new_order_state[2], new_order_state[3]  # dlat,dlon (new orders)
        current_orders_num += 1

    # simulate 1 min
    step = 1  # 1min
    if observe_space[3] != 0:  # pick up
        if observe_space[3] > step:
            observe_space[3] -= step
        else:  # finish picking up
            step -= observe_space[3]
            observe_space[3] = 0
            if observe_space[2] != 0:  # have available seat
                observe_space[4] = 0 # update state to available

    if step > 0 and current_orders_num != 0:
        step_minute = step
        step = step * 60
        for i in range(len(current_travel_time)):
            if step >= current_travel_time[i]:
                step -= current_travel_time[i]
            else:
                current_travel_time[i] -= step
                current_travel_time = current_travel_time[i:]
                current_travel_route = current_travel_route[i:]
                observe_space[0], observe_space[1] = current_travel_route[0][1], current_travel_route[0][0]  # lat, lon
                break
            if i == len(current_travel_time) - 1:  # finish all orders
                observe_space[0], observe_space[1] = current_travel_route[-1][1], current_travel_route[-1][0]  # lat, lon
                current_travel_time = []
                current_travel_route = []
        current_orders[:current_orders_num, 2] -= step_minute  # update remaining time

        # delete finished orders
        drop_index = np.zeros(current_orders.shape[0])
        drop_index[:current_orders_num] = (current_orders[:current_orders_num, 2] <= 0)
        drop_num = np.sum(drop_index)
        if drop_num > 0:
            current_orders_num -= drop_num
            observe_space[2] += drop_num
            observe_space[4] = 0
            drop_index = drop_index.astype(bool)
            finished_orders = current_orders[drop_index]
            current_orders = current_orders[~drop_index]
            fill_matrix = np.zeros_like(finished_orders)
            current_orders = np.concatenate([current_orders, fill_matrix], axis=0)
            finished_order_time = finished_orders[:, 3]
            finished_order_detour = finished_orders[:, 4]

    return observe_space, current_orders, current_orders_num, current_travel_route, current_travel_time, finished_order_time, finished_order_detour
