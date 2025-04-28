import pandas as pd
import pickle

zone_table_path = "./Manhattan_dic.pkl"
with open(zone_table_path, 'rb') as f:
    zone_dic = pickle.load(f)
zone_table = zone_dic["zone_num"]

demand_path = "./yellow_tripdata_2024-07.parquet"
demand = pd.read_parquet(demand_path)
demand['tpep_pickup_datetime'] = pd.to_datetime(demand['tpep_pickup_datetime'])
demand['day'] = demand['tpep_pickup_datetime'].dt.day
demand['hour'] = demand['tpep_pickup_datetime'].dt.hour
demand = demand[
    (demand["PULocationID"].isin(zone_table)) & (demand["DOLocationID"].isin(zone_table))]

for i in range(1,32):
    for j in range(24):
        filtered_demand = demand[(demand["day"] == i) & (demand["hour"] == j)]
        print(f"Day {i} Hour {j} Order {len(filtered_demand)}")