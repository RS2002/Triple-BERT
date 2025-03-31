import requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import folium

session = requests.Session()

retry = Retry(connect=500000000000000000, backoff_factor=0.2)

adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# docker run -t -i -p 5000:5000 -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/new-york-latest.osrm

def TSP_route(origin_point, destination_points):

    url = "http://localhost:6000/trip/v1/driving/" + str(origin_point[1]) + "," + str(origin_point[0]) + ";"

    for i in range(len(destination_points)):
        if i < len(destination_points) - 1:
            url += str(destination_points[i][1]) + "," + str(destination_points[i][0]) + ";"
        else:
            url += str(destination_points[i][1]) + "," + str(
                destination_points[i][0]) + "?roundtrip=false&source=first&annotations=true&geometries=geojson&overview=full"

    # session = requests.Session()
    #
    # retry = Retry(connect=500000000000000000, backoff_factor=0.2)
    # # retry = Retry(connect=500000000000000000, backoff_factor=0)
    #
    # adapter = HTTPAdapter(max_retries=retry)
    # session.mount('http://', adapter)
    # session.mount('https://', adapter)

    route = []
    route_time = []
    time = []
    time_permutation = []
    time_order = []
    distance = 0

    r = session.get(url)
    if not r.text:
        route_time = [0]
        time_order = [0]
    else:
        res = r.json()

        # print(res['code'])
        if res['code'] == 'Ok':
            route = res["trips"][0]["geometry"]["coordinates"]

            distance = res['trips'][0]['distance']

            waypoints = res['waypoints']
            for i in range(len(waypoints)):
                time_permutation.append(waypoints[i]['waypoint_index'])

            legs = res['trips'][0]['legs']
            for i in range(len(legs)):
                route_time.extend(legs[i]['annotation']["duration"])
                if i == 0:
                    time.append(int(legs[i]['duration'] / 60))
                else:
                    time.append(int(legs[i]['duration'] / 60) + time[i - 1])

            for i in range(len(time)):
                time_order.append(time[time_permutation[i + 1] - 1])

        else:
            route_time = [0]
            time_order = [0]

    return (route,route_time,time_order, distance) # newly added distance
    ''' 
    route: 规划路径上的节点; route_time：各节点间所需时间; time_order：到达每个目的地需要的时间; distance：总移动距离
    route中第一个元素是起点位置，可能由于精度等问题，数值上存在一些误差
    route_time元素数量正好比route中元素数少1
    '''

def update_loc(step,route,route_t):
    for i in range(len(route_t)):
        if step < sum(route_t[:i]):
            route = route[i:]
            route_t = route_t[i:]
            break

    loc = (route[0][1],route[0][0])

    return loc,route,route_t


def get_map(route,originpoint, destinationpoints,m):#real_dests
    #inverse route to get right lon&lat

    # print(destinationpoints)

    if len(destinationpoints) > 0:

        route_map = []
        for i in range(len(route)):
            route_map.append([route[i][1], route[i][0]])

        folium.PolyLine(
            route_map,
            weight=8,
            color='blue',
            opacity=0.6
        ).add_to(m)

        folium.Marker(
            location=originpoint,
            icon=folium.Icon(icon='play', color='green')
        ).add_to(m)

        for i in range(len(destinationpoints)):
            folium.Marker(
                location=destinationpoints[i],
                icon=folium.Icon(icon='stop', color='red'),
                popup='drop_'+str(i),
                popout=True
            ).add_to(m)

            # print(destinationpoints[i])
            # print(real_dests[i])
            # folium.PolyLine(
            #     [destinationpoints[i],real_dests[i]],
            #     weight=8,
            #     color='purple',
            #     opacity=0.6
            # ).add_to(m)
            #
            # folium.Marker(
            #     location=real_dests[i],
            #     icon=folium.Icon(icon='home', color='purple')
            #     , popup='real_'+str(i), popout=True
            # ).add_to(m)

    else:

        folium.Marker(
            location=originpoint,
            icon=folium.Icon(icon='play', color='green')
        ).add_to(m)

    return m


if __name__ == '__main__':
    # test
    origin_point = (40.77876573980772, -73.9510100659439)
    destination_points = [(40.72375208451233, -73.97696827424141), (40.804333857858566,  -73.95129204385638), (40.804333857858566,  -73.99129204385638)]
    route, route_t, t, dis = TSP_route(origin_point, destination_points)

    print(len(route))
    print(len(route_t))
    print(route)
    print(route_t)
    print(t)
    print(dis)
    print(np.sum(route_t))
    #
    #
    # print(np.sum(route_t)/60)
    #
    # print(len(route))
    # print(len(route_t))

    # print(geodistance())

    # route, route_t, t = TSP_route(origin_point,destination_points)[:-1]
    #
    # # Do some visualization if you want
    # # m = folium.Map(location=[22.34022923928267, 114.26263474465348],
    # #                  zoom_start=13)
    # #
    # # map = get_map(route, origin_point, destination_points, destination_points, m)
    # # map.save("test.html")
    #
    # print("route total time for each destination is", t)