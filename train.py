from Worker import Buffer, Worker
from Platform import Platform, reward_func_generator, assign
from Order_Env import Demand
import argparse
import tqdm
import torch
import pickle
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_times', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--converge_epoch', type=int, default=10)
    parser.add_argument('--minimum_episode', type=int, default=1000)
    parser.add_argument('--worker_num', type=int, default=1000)
    parser.add_argument('--buffer_capacity', type=int, default=1500)
    parser.add_argument('--buffer_episode', type=int, default=20)
    parser.add_argument('--demand_sample_rate', type=float, default=0.99)
    parser.add_argument('--order_max_wait_time', type=float, default=5.0)
    parser.add_argument('--order_threshold', type=float, default=40.0)
    parser.add_argument('--reward_parameter', type=float, nargs='+', default=[3.0,1.0,3.0,1.0,3.0,5.0])

    parser.add_argument("--day", type=int, default=17)
    parser.add_argument("--hour", type=int, default=18)
    parser.add_argument("--compression", action="store_true",default=False)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument("--bi_direction", action="store_true",default=False)
    parser.add_argument('--eval_episode', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_decay_rate', type=float, default=0.99)
    parser.add_argument('--epsilon_final', type=float, default=0.0005)

    parser.add_argument("--cpu", action="store_true",default=False)
    parser.add_argument("--cuda", type=str, default='0')

    parser.add_argument('--init_episode', type=int, default=0)
    parser.add_argument('--njobs', type=int, default=24)

    parser.add_argument("--model_path",type=str,default=None)

    parser.add_argument("--demand_path",type=str,default="./data/yellow_tripdata_2024-07.parquet")
    parser.add_argument("--zone_dic_path",type=str,default="./data/Manhattan_dic.pkl")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    device_name = "cuda:" + args.cuda
    device = torch.device(device_name if torch.cuda.is_available() and not args.cpu else 'cpu')

    compression = args.compression
    if compression:
        args.max_step = args.max_step // 2
        args.epsilon_decay_rate = np.sqrt(args.epsilon_decay_rate)


    day = args.day
    hour = args.hour
    exploration_rate = args.epsilon
    epsilon_decay_rate = args.epsilon_decay_rate
    epsilon_final = args.epsilon_final


    critic_train = 4
    actor_train = 12
    counter = 0
    cycle = 12


    with open(args.zone_dic_path, 'rb') as f:
        zone_dic = pickle.load(f)
    zone_table = zone_dic["zone_num"]

    platform = Platform(discount_factor=args.gamma, njobs=args.njobs)
    demand = Demand(demand_path=args.demand_path, zone_table=zone_table)
    buffer = Buffer(capacity=args.buffer_capacity, episode_capacity=args.buffer_episode)
    worker = Worker(buffer, lr=args.lr, gamma=args.gamma, max_step=args.max_step, num=args.worker_num, device=device, zone_table_path = args.zone_dic_path, model_path = args.model_path, njobs = args.njobs, bi_direction = args.bi_direction, dropout = args.dropout, compression = compression)
    reward_func = reward_func_generator(args.reward_parameter, args.order_threshold)

    best_reward = -1e8
    best_epoch = 0
    j = args.init_episode
    exploration_rate = max(exploration_rate * (epsilon_decay_rate**j), epsilon_final)


    while True:
        j+=1
        exploration_rate = max(exploration_rate * epsilon_decay_rate, epsilon_final)
        print(f"Exploration Rate {exploration_rate}")

        worker.reset(train=True)
        platform.reset(discount_factor=args.gamma)
        demand.reset(day=day, hour=hour, wait_time=args.order_max_wait_time, compression=compression)

        loss_list = []
        pbar = tqdm.tqdm(range(args.max_step))
        for t in pbar:
            q_value, order_state = worker.observe(demand.current_demand, t, exploration_rate)
            assignment, _ = assign(q_value)
            feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, new_detour_table, assignment, reward = \
                platform.feedback(worker.observe_space, worker.current_orders, worker.current_order_num, assignment, order_state, args.order_threshold, reward_func, t)
            worker.update(feedback_table, new_route_table ,new_route_time_table ,new_remaining_time_table ,new_total_travel_time_table, new_detour_table, reward, assignment, order_state, (t==args.max_step - 1), j)
            assignment = [x for x in assignment if x is not None]
            demand.pickup(assignment)
            demand.update()

            if (counter+1) % critic_train == 0:
                train_actor = bool((counter+1)%actor_train==0)
                train_critic = not train_actor
                loss = worker.train(args.batch_size, 1, False, train_actor, train_critic)
                loss_list.append(loss)
            counter = (counter + 1) % cycle

        # loss = worker.train(args.batch_size,args.train_times)
        loss = np.mean(loss_list)
        worker.schedule.step()

        Pickup_Num = platform.Pickup_Num
        Detour = np.mean(worker.Detour_Time)
        Confirmation =  platform.Confirmation_Time / platform.Pickup_Num
        Pickup = platform.Pickup_Time / platform.Pickup_Num
        Delivery = np.mean(worker.Pass_Travel_Time)
        reward = platform.Total_Reward

        log = f"Epoch {j} | Reward: {reward} , Loss: {loss} , Served Order: {Pickup_Num} , Delivery Time: {Delivery} , Detour Time: {Detour} , Pickup Time: {Pickup} , Confirmation Time: {Confirmation}"
        print(log)
        with open("train.txt", 'a') as file:
            file.write(log + "\n")
        worker.save("latest.pth")

        if j % args.eval_episode == 0 :
            worker.reset(train=False)
            platform.reset(discount_factor=args.gamma)
            demand.reset(day=day, hour=hour, wait_time=args.order_max_wait_time, compression=compression)

            pbar = tqdm.tqdm(range(args.max_step))
            for t in pbar:
                q_value, order_state = worker.observe(demand.current_demand, t)
                assignment, _ = assign(q_value)
                feedback_table, new_route_table, new_route_time_table, new_remaining_time_table, new_total_travel_time_table, new_detour_table, assignment, reward = \
                    platform.feedback(worker.observe_space, worker.current_orders, worker.current_order_num, assignment,
                                      order_state, args.order_threshold, reward_func, t)
                worker.update(feedback_table, new_route_table, new_route_time_table, new_remaining_time_table,
                              new_total_travel_time_table, new_detour_table, reward, assignment, order_state, (t == args.max_step - 1), j)
                assignment = [x for x in assignment if x is not None]
                demand.pickup(assignment)
                demand.update()

            Pickup_Num = platform.Pickup_Num
            Detour = np.mean(worker.Detour_Time)
            Confirmation = platform.Confirmation_Time / platform.Pickup_Num
            Pickup = platform.Pickup_Time / platform.Pickup_Num
            Delivery = np.mean(worker.Pass_Travel_Time)
            reward = platform.Total_Reward

            log = f"Epoch {j} | Reward: {reward}, Served Order: {Pickup_Num} , Delivery Time: {Delivery} , Detour Time: {Detour} , Pickup Time: {Pickup} , Confirmation Time: {Confirmation}"
            print(log)
            with open("eval.txt", 'a') as file:
                file.write(log + "\n")
            if reward > best_reward:
                best_epoch = 0
                best_reward = reward
                worker.save("best.pth")
            else:
                best_epoch += 1

            if j == args.minimum_episode:
                best_epoch = 0
            if j >= args.minimum_episode:
                print("Converge Step: ", best_epoch)
                if best_epoch >= args.converge_epoch:
                    break

if __name__ == '__main__':
    main()