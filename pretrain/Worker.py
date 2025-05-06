import numpy as np
import torch
from model import Assignment_Net
from joblib import Parallel, delayed
import torch.nn as nn
import tqdm
import pickle
from torch.cuda.amp import GradScaler

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

        self.action = [] # order state

        self.delta_t = []

        self.worker_state_next = []
        self.order_state_next = []
        self.order_num_next = []

        self.action_next = [] # next order state

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
            self.action = self.action[old_record_num:]
            self.delta_t = self.delta_t[old_record_num:]
            self.worker_state_next = self.worker_state_next[old_record_num:]
            self.order_state_next = self.order_state_next[old_record_num:]
            self.order_num_next = self.order_num_next[old_record_num:]
            self.action_next = self.action_next[old_record_num:]
            self.reward = self.reward[old_record_num:]
            self.episode = self.episode[old_record_num:]
            if self.episode[0]<episode-self.episode_capacity:
                print("Buffer Error!")
                exit(-1)


        state, reward, delta_t, state_next = experience
        if self.num == self.capacity:
            self.worker_state = self.worker_state[1:]
            self.order_state = self.order_state[1:]
            self.order_num = self.order_num[1:]
            self.action = self.action[1:]
            self.delta_t = self.delta_t[1:]
            self.worker_state_next = self.worker_state_next[1:]
            self.order_state_next = self.order_state_next[1:]
            self.order_num_next = self.order_num_next[1:]
            self.action_next = self.action_next[1:]
            self.reward = self.reward[1:]
            self.episode = self.episode[1:]
        else:
            self.num+=1

        self.worker_state.append(state[0].tolist())
        self.order_state.append(state[1].tolist())
        self.order_num.append(state[2])
        self.action.append(state[3].tolist())
        self.delta_t.append(delta_t)
        self.worker_state_next.append(state_next[0].tolist())
        self.order_state_next.append(state_next[1].tolist())
        self.order_num_next.append(state_next[2])
        self.action_next.append(state_next[3].tolist())

        # self.reward.append(reward)
        # print(reward)
        self.reward.append(reward / 10)

        self.episode.append(episode)

    def sample(self,size,device):
        if size>self.num:
            size = self.num

        indices = np.random.randint(0, self.num, size=size)
        # priority = np.array(self.episode)
        # priority = priority - np.min(priority) + 1
        # probabilities = np.array(priority) / np.sum(priority)
        # indices = np.random.choice(self.num, size, p=probabilities)

        worker_state = torch.tensor([self.worker_state[i] for i in indices]).to(device)
        order_state = torch.tensor([self.order_state[i] for i in indices]).to(device)
        order_num = torch.tensor([self.order_num[i] for i in indices]).to(device)
        action = torch.tensor([self.action[i] for i in indices]).to(device)
        delta_t = torch.tensor([self.delta_t[i] for i in indices]).to(device)
        worker_state_next = torch.tensor([self.worker_state_next[i] for i in indices]).to(device)
        order_state_next = torch.tensor([self.order_state_next[i] for i in indices]).to(device)
        order_num_next = torch.tensor([self.order_num_next[i] for i in indices]).to(device)
        action_next = torch.tensor([self.action_next[i] for i in indices]).to(device)
        reward = torch.tensor([self.reward[i] for i in indices]).to(device)

        return worker_state, order_state, order_num, action, delta_t, reward, worker_state_next, order_state_next, order_num_next, action_next

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

        self.Q_training = Assignment_Net(state_size=6, history_order_size=5, current_order_size=5, hidden_dim=64, bi_direction=bi_direction, dropout=dropout).to(device)
        self.Q_target = Assignment_Net(state_size=6, history_order_size=5, current_order_size=5, hidden_dim=64, bi_direction=bi_direction, dropout=dropout).to(device)

        self.load(model_path,self.device)
        for param in self.Q_target.parameters():
            param.requires_grad = False
        self.Q_target.eval()
        print('Platform total parameters:', sum(p.numel() for p in self.Q_training.parameters() if p.requires_grad))
        self.update_target(tau=1.0)

        self.optim = torch.optim.Adam(self.Q_training.parameters(), lr=lr)
        # if compression:
        #     self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=np.sqrt(0.99))
        # else:
        #     self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)
        self.schedule = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma=0.99)

        self.njobs = njobs
        self.loss_func = nn.MSELoss()
        # self.scaler = GradScaler()


        self.reset()

    def save(self, path):
        torch.save(self.Q_training.state_dict(), path)

    def load(self, path1=None, device=torch.device("cpu")):
        if path1 is not None:
            self.Q_target.load_state_dict(torch.load(path1), map_location=device, strict=False)
            self.Q_training.load_state_dict(torch.load(path1), map_location=device, strict=False)

    def update_target(self, tau=0.005):
        for target_param, train_param in zip(self.Q_target.parameters(), self.Q_training.parameters()):
            target_param.data.copy_(tau * train_param.data + (1.0 - tau) * target_param.data)

    def reset(self, capacity = 3, train=True):
        if train:
            self.Q_training.train()
        else:
            self.Q_training.eval()
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
        self.experience = [[] for _ in range(self.num)]
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
        q_value = self.Q_training(x1, x2, x3, torch.from_numpy(self.current_order_num).to(self.device))
        # 2. epsilon-greedy explore
        exploration_matrix = torch.rand_like(q_value)
        q_value[exploration_matrix < exploration_rate] = INF
        q_value[self.observe_space[:, 4] == 1] = -INF

        return q_value.cpu().detach().numpy(), order

    def train(self, batch_size=256, train_times=1, show_pbar=False):
        torch.set_grad_enabled(True)
        self.Q_training.train()
        if show_pbar:
            pbar = tqdm.tqdm(range(train_times))
        else:
            pbar = range(train_times)

        loss_list = []
        for _ in pbar:
            worker_state, order_state, order_num, action, delta_t, reward, worker_state_next, order_state_next, order_num_next, action_next = self.buffer.sample(batch_size,self.device)
            x1, x2, x3 = norm(action, worker_state, order_state)
            x1_next, x2_next, x3_next = norm(action_next, worker_state_next, order_state_next)

            current_q = self.Q_training(x1,x2,x3, order_num)
            current_q = torch.diag(current_q)

            next_q1 = self.Q_target(x1_next,x2_next,x3_next, order_num_next)
            next_q1 = torch.diag(next_q1)
            # next_q2 = self.Q_training(x1_next,x2_next,x3_next, order_num_next)
            # next_q2 = torch.diag(next_q2)
            # next_q = torch.min(next_q1,next_q2)
            next_q = next_q1

            is_done = (delta_t == -1).float()
            target = reward + (self.gamma ** delta_t * next_q) * (1 - is_done)
            loss = self.loss_func(current_q,target.float().detach())
            loss_list.append(loss.item())

            self.optim.zero_grad()
            loss.backward()
            # self.scaler.scale(loss).backward()  # 缩放损失并反向传播

            torch.nn.utils.clip_grad_norm_(self.Q_training.parameters(), 1.0)  # avoid gradient explosion
            has_nan = False
            for name, param in self.Q_training.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan = True
                        break
            if has_nan:
                continue

            self.optim.step()
            # self.scaler.step(self.optim)  # 先unscale梯度，再更新参数（跳过溢出情况）
            # self.scaler.update()  # 调整缩放因子


        self.update_target()
        # self.schedule.step()
        torch.set_grad_enabled(False)
        return np.mean(loss_list)

    def update(self, feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, new_detour_table, reward, assignment_table, assignment_state, final_step=False, episode=1):
        # update each worker state parallely
        results = Parallel(n_jobs=self.njobs)(
            delayed(single_update)(self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.travel_route[i], self.travel_time[i], feedback_table[i], new_route_table[i], new_route_time_table[i], new_remaining_time_table[i], new_total_travel_time_table[i], new_detour_table[i], self.experience[i])
            for i in range(self.num))

        for i in range(len(results)):
            self.observe_space[i], self.current_orders[i], self.current_order_num[i], self.travel_route[i], \
            self.travel_time[i], self.experience[i] = results[i][0], results[i][1], results[i][2], results[i][3], results[i][4], results[i][-2]
            if results[i][5] is not None:
                self.Pass_Travel_Time.extend(results[i][5].tolist())
                self.Detour_Time.extend(results[i][6].tolist())
            if self.is_train:
                if results[i][-1] is not None:
                    self.buffer.append(results[i][-1], episode)
                if final_step and len(self.experience[i])>0:
                    self.experience[i].append(-1)  # △t: -1 represents done
                    self.experience[i].append(self.experience[i][0])  # meaningless: only used to keep a same dimension
                    self.buffer.append(self.experience[i], episode)


def single_update(observe_space, current_orders, current_orders_num, current_travel_route, current_travel_time, feedback, new_route ,new_route_time, new_remaining_time, new_total_travel_time, new_detour_table, experience):
    finished_order_time = None
    finished_order_detour = None
    full_experience = None
    current_orders_num = int(current_orders_num)

    if feedback is not None:
        new_order_state = feedback[0][3]
        pickup_time = feedback[2]

        # write experience
        if len(experience) > 0:
            experience.append(observe_space[5] - experience[0][0][5])  # △t
            experience.append(feedback[0])  # s_next+a_next
            full_experience = experience
            experience = []
        experience.append(feedback[0])  # s_current+a_current
        experience.append(feedback[1])  # r


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
            step = 0
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

    return observe_space, current_orders, current_orders_num, current_travel_route, current_travel_time, finished_order_time, finished_order_detour, experience, full_experience
