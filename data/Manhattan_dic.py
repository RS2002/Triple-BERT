import geopandas as gpd
from pyproj import CRS, Transformer
import pickle

# transfer New York coordinate to lat&lon
with open("./taxi_zones/taxi_zones.prj", 'r') as file:
    prj_content = file.read()
input_crs = CRS.from_wkt(prj_content)
output_crs = CRS.from_epsg(4326)
transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)


taxi_zones = gpd.read_file('./taxi_zones/taxi_zones.shp')

zone_num = []
Manhattan_num = []
centroid_lat = []
centroid_lon = []
map = [-1]*taxi_zones.shape[0]

j = 0
for i in range(taxi_zones.shape[0]):
    if taxi_zones["borough"][i] == "Manhattan":
        map[i] = j
        zone_num.append(i+1)
        Manhattan_num.append(j)
        j+=1

        # calculate the region centroid
        polygon = taxi_zones['geometry'][i]
        centroid = polygon.centroid
        longitude, latitude = transformer.transform(centroid.x, centroid.y)
        centroid_lat.append(latitude)
        centroid_lon.append(longitude)

Manhattan_dic = {
    "zone_num": zone_num,
    # "Manhattan_num": Manhattan_num,
    "centroid_lat": centroid_lat,
    "centroid_lon": centroid_lon,
    "map": map
}

with open("Manhattan_dic.pkl",'wb') as f:
    pickle.dump(Manhattan_dic, f)



