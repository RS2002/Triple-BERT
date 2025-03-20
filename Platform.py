from osrm import TSP_route
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
import numpy as np

def assign(q_matrix,pad=True):
    # threshold = -5.0
    threshold = -10.0

    # Solve Bipartite Match Process with ILP
    num_vehicles, num_demands = q_matrix.shape
    if pad:
        Value_Matrix = np.concatenate((q_matrix,np.zeros_like(q_matrix)+threshold),axis=1)
    else:
        Value_Matrix = q_matrix
    cost_matrix = -Value_Matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    # 创建一个列表来保存每个车辆被分配的订单
    assignment = [None] * len(Value_Matrix)
    # 获取每个车辆被分配的订单
    for i in range(len(row_indices)):
        if col_indices[i] >= num_demands:
            assignment[row_indices[i]] = None
        else:
            assignment[row_indices[i]] = col_indices[i]
    # 计算最大化的值
    max_value = -1 * cost_matrix[row_indices, col_indices].sum()
    # 返回分配结果和最大值
    return assignment, max_value


'''
beta_list:
beta_list[0]: reward of taking new order
beta_list[1]: reward of user paying (proportion to distance : km)
beta_list[2]: punishment of picking up time (user waiting time : hour)
beta_list[3]: punishment of timeout orders
beta_list[4]: punishment of added time
beta_list[5]: bigger punishment of added time (over threshold part)
'''
def reward_func_generator(beta_list, threshold):
    def reward_func(time_add,time_out,pickup_time,direct_distance):
        # print(time_add,time_out,pickup_time,direct_distance)
        if time_add <= threshold:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * pickup_time / 60 - beta_list[3] * time_out - beta_list[4] * time_add / 60
        else:
            r = beta_list[0] + beta_list[1] * direct_distance / 1000  - beta_list[2] * pickup_time / 60 - beta_list[3] * time_out - beta_list[4] * time_add / 60 - beta_list[5] * (time_add-threshold) / 60
        return r
    return reward_func

class Platform():
    def __init__(self,discount_factor=0.99, njobs=24):
        super().__init__()
        self.reset(discount_factor)
        self.njobs = njobs

    def reset(self,discount_factor=0.99):
        self.discount_factor = discount_factor

        self.Total_Reward = 0
        self.Pickup_Num = 0
        self.Pickup_Time = 0
        self.Confirmation_Time = 0

    def feedback(self, observe, current_order_state, current_order_num, assignment, new_orders_state, time_threshold, reward_func, current_time):
        feedback_table = []
        new_route_table = []
        new_route_time_table = []
        new_remaining_time_table = []
        new_total_travel_time_table = []
        new_detour_table = []
        reward = 0

        results = Parallel(n_jobs=self.njobs)(
            delayed(execute)(observe[i], current_order_state[i], current_order_num[i], assignment[i], new_orders_state, time_threshold, reward_func, current_time)
            for i
            in range(observe.shape[0]))


        for i in range(len(results)):
            result = results[i]
            feedback_table.append(result[0])
            new_route_table.append(result[1])
            new_route_time_table.append(result[2])
            new_remaining_time_table.append(result[3])
            new_total_travel_time_table.append(result[4])
            new_detour_table.append(result[5])

            if result[0] is None:
                assignment[i] = None
            else:
                log = result[6]
                self.Pickup_Num += 1
                self.Confirmation_Time += log[0]
                self.Pickup_Time += log[1]
                reward += result[0][1]
        self.Total_Reward += self.discount_factor ** current_time * reward
        return feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, new_detour_table, assignment, reward



def execute(observe, current_order_state, current_order_num, assignment, new_orders_state, time_threshold, reward_func, current_time):
    if assignment is not None and observe[4] == 0:
        curr_lat, curr_lon = observe[0], observe[1]
        plat, plon, dlat, dlon, appear_time = new_orders_state[assignment]
        confirmation_time = current_time - appear_time
        current_order_num = int(current_order_num)

        # 1. compute direct distance & pickup time
        pickup_route, pickup_route_t, pickup_time, _ = TSP_route((curr_lat, curr_lon), [(plat, plon)])
        pickup_time = pickup_time[0]
        direct_route, direct_route_time, direct_time, direct_distance = TSP_route((plat, plon), [(dlat, dlon)])
        direct_time = direct_time[0]

        # 2. schedule route
        destination_points = []
        for i in range(current_order_num):
            destination_points.append((current_order_state[i, 0], current_order_state[i, 1]))
        destination_points.append((dlat, dlon))
        new_route, new_route_time, new_time, _ = TSP_route((plat, plon), destination_points)

        # 3. update total travel time
        new_total_travel_time = np.array(new_time)
        new_total_travel_time[:-1] = new_total_travel_time[:-1] + current_order_state[:current_order_num, 3] - current_order_state[:current_order_num, 2]  # add the time already cost for each old order
        new_total_travel_time = new_total_travel_time + pickup_time

        # 4. calculate reward
        original_total_travel_time = np.sum(current_order_state[:, 3])
        time_add = np.sum(new_total_travel_time) - original_total_travel_time  # total added time of all orders
        timeout = np.sum(new_total_travel_time > time_threshold)  # how many orders will be over time
        reward = reward_func(time_add,timeout,pickup_time,direct_distance)

        # 5. calculate detour (!= workload add of driver)
        # detour_current = new_total_travel_time[-1] - direct_time - pickup_time
        # detour_previous = np.sum(new_total_travel_time[:-1] - current_order_state[:current_order_num, 3])
        # detour = detour_current + detour_previous
        detour = np.zeros_like(new_time)
        detour[:current_order_num] = current_order_state[:current_order_num, 4] + new_total_travel_time[:-1] - current_order_state[:current_order_num, 3]
        detour[current_order_num] = new_total_travel_time[-1] - direct_time - pickup_time

        # 6. log: detour, confirmation time, pickup time
        log = [confirmation_time, pickup_time]
        feedback = [ [observe, current_order_state, current_order_num, new_orders_state[assignment]] ,
                    reward, pickup_time]
        return feedback, new_route, new_route_time, new_time, new_total_travel_time, detour, log

    return None, None, None, None, None, None, None
