# coding=utf-8
import socket  # socket模块
import json
import random
import numpy as np
from collections import deque
from math import floor, sqrt
import subprocess
import time
import psutil
import pyautogui
import os
from multiprocessing import Pool
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def connect(ser):
    conn, addr = ser.accept()  # 接受TCP连接，并返回新的套接字与IP地址
    print('Connected by', addr)  # 输出客户端的IP地址
    return conn


def open_ter(loc):
    print("gnome-terminal -e 'bash -c \"cd " + loc + " && ./path_planning; exec bash\"'")
    os.system("gnome-terminal -e 'bash -c \"cd " + loc + " && ./path_planning; exec bash\"'")
    time.sleep(1)
    return sim


def kill_terminal():
    pids = psutil.pids()
    for pid in pids:
        p = psutil.Process(pid)
        if p.name() == "gnome-terminal-server":
            os.kill(pid, 9)


def close_all(sim):
    if sim.poll() is None:
        sim.terminate()
        sim.wait()
    time.sleep(2)
    kill_terminal()


EPISODES = 20
location = "/mnt/d/code/algorithm/Fighting/decision-making-CarND/CarND-test/build"

HOST = '127.0.0.1'
PORT = 1234
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
server.bind((HOST, PORT))  # 套接字绑定的IP与端口
server.listen(1)  # 开始TCP监听

state_height = 45
state_width = 3
action_size = 3

batch_size = 32
episode = 1


while episode <= EPISODES:
    pool = Pool(processes=2)
    result = []
    result.append(pool.apply_async(connect, (server,)))
    pool.apply_async(open_ter, (location,))
    pool.close()
    pool.join()
    conn = result[0].get()
    print(1)
    sim = subprocess.Popen('../../../term3_sim_linux/term3_sim.x86_64')
    time.sleep(2)
    pyautogui.click(x=1164, y=864, button='left')
    time.sleep(6)
    pyautogui.click(x=465, y=535, button='left')
    try:
        data = conn.recv(2000)  # 把接收的数据实例化
    except Exception as e:
        # close_all(sim)
        continue
    while not data:
        try:
            data = conn.recv(2000)  # 把接收的数据实例化
        except Exception as e:
            # close_all(sim)
            continue
    data = bytes.decode(data)
    # print(data)
    j = json.loads(data)
    # Main car's localization Data
    car_x = j[1]['x']
    car_y = j[1]['y']
    car_s = j[1]['s']
    car_d = j[1]['d']
    car_yaw = j[1]['yaw']
    car_speed = j[1]['speed']
    # Sensor Fusion Data, a list of all other cars on the same side of the road.\
    sensor_fusion = j[1]['sensor_fusion']
    grid = np.ones((51, 3))
    ego_car_lane = int(floor(car_d/4))
    grid[31:35, ego_car_lane] = car_speed / 100.0

    # sensor_fusion_array = np.array(sensor_fusion)
    for i in range(len(sensor_fusion)):
        vx = sensor_fusion[i][3]
        vy = sensor_fusion[i][4]
        s = sensor_fusion[i][5]
        d = sensor_fusion[i][6]
        check_speed = sqrt(vx * vx + vy * vy)
        car_lane = int(floor(d / 4))
        if 0 <= car_lane < 3:
            s_dis = s - car_s
            if -36 < s_dis < 66:
                pers = - int(floor(s_dis / 2.0)) + 30
                grid[pers:pers + 4, car_lane] = - check_speed / 100.0 * 2.237

    state = np.zeros((state_height, state_width))
    state[:, :] = grid[3:48, :]
    state = np.reshape(state, [-1, 1, state_height, state_width])
    pos = [car_speed / 50, 0, 0]
    if ego_car_lane == 0:
        pos = [car_speed / 50, 0, 1]
    elif ego_car_lane == 1:
        pos = [car_speed / 50, 1, 1]
    elif ego_car_lane == 2:
        pos = [car_speed / 50, 1, 0]
    pos = np.reshape(pos, [1, 3])
    # print(state)
    action = 0
    mess_out = str(action)
    mess_out = str.encode(mess_out)
    conn.sendall(mess_out)
    count = 0
    start = time.time()
    while True:
        now = time.time()
        if (now - start) / 60 > 15:
            # close_all(sim)
            break
        try:
            data = conn.recv(2000)  # 把接收的数据实例化
        except Exception as e:
            pass
        while not data:
            try:
                data = conn.recv(2000)  # 把接收的数据实例化
            except Exception as e:
                pass
        data = bytes.decode(data)
        if data == "over":
            print("episode:{}".format(episode))
            # close_all(sim)
            conn.close()  # 关闭连接
            episode = episode + 1
            break
        j = json.loads(data)
        last_state = state
        last_pos = pos
        last_act = action
        last_lane = ego_car_lane
        # Main car's localization Data
        car_x = j[1]['x']
        car_y = j[1]['y']
        car_s = j[1]['s']
        car_d = j[1]['d']
        car_yaw = j[1]['yaw']
        car_speed = j[1]['speed']
        print(car_s)
        if car_speed == 0:
            mess_out = str(0)
            mess_out = str.encode(mess_out)
            conn.sendall(mess_out)
            continue

        # Sensor Fusion Data, a list of all other cars on the same side of the road.
        sensor_fusion = j[1]['sensor_fusion']
        ego_car_lane = int(floor(car_d / 4))
        if last_act == 0:
            last_reward = 2 * (2 * ((j[3] - 35.0) / 5.0))  # - abs(ego_car_lane - 1))
        else:
            last_reward = 2 * (2 * ((j[3] - 35.0) / 5.0)) - 6.0
        if grid[3:31, last_lane].sum() > 27 and last_act != 0:
            last_reward = -60.0

        grid = np.ones((51, 3))
        grid[31:35, ego_car_lane] = car_speed / 100.0
        # sensor_fusion_array = np.array(sensor_fusion)
        for i in range(len(sensor_fusion)):
            vx = sensor_fusion[i][3]
            vy = sensor_fusion[i][4]
            s = sensor_fusion[i][5]
            d = sensor_fusion[i][6]
            check_speed = sqrt(vx * vx + vy * vy)
            car_lane = int(floor(d / 4))
            if 0 <= car_lane < 3:
                s_dis = s - car_s
                if -36 < s_dis < 66:
                    pers = - int(floor(s_dis / 2.0)) + 30
                    grid[pers:pers + 4, car_lane] = - check_speed / 100.0 * 2.237
            if j[2] < -10:
                last_reward = float(j[2])  # reward -50, -100
        last_reward = last_reward / 20.0

        state = np.zeros((state_height, state_width))
        state[:, :] = grid[3:48, :]
        state = np.reshape(state, [-1, 1, state_height, state_width])
        # print(state)
        pos = [car_speed / 50, 0, 0]
        if ego_car_lane == 0:
            pos = [car_speed / 50, 0, 1]
        elif ego_car_lane == 1:
            pos = [car_speed / 50, 1, 1]
        elif ego_car_lane == 2:
            pos = [car_speed / 50, 1, 0]
        pos = np.reshape(pos, [1, 3])
        # print(state)
        action = random.randint(0, 2)
        # action = agent.act([state, pos])
        if action != 0:
            print(action)

        mess_out = str(action)
        mess_out = str.encode(mess_out)
        conn.sendall(mess_out)
