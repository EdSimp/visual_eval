import json

import numpy as np


def coordinateTrans(raw, a, t=None):
    '''

    :param raw: 某点在原坐标系下坐标 (n,2)
    :param a: 新坐标系相对于原坐标系，逆时针旋转角度 弧度制
    :param t: 原坐标系下，新坐标系原点的坐标 (n,2) or (2,)
    :return: 某点在新坐标系下坐标 (n,2)
    '''
    # a = np.pi*a/180
    if t is None:
        vector = raw
    else:
        vector = raw - t
    newx = np.cos(a) * vector[:, 0] + np.sin(a) * vector[:, 1]
    newy = -np.sin(a) * vector[:, 0] + np.cos(a) * vector[:, 1]
    new = np.concatenate([newx.reshape(-1, 1), newy.reshape(-1, 1)], 1)
    return new


def parse_json(json_path):
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def substring(now_list):
    now_list_back = now_list
    all_list = np.arange(now_list[0], now_list[-1] + 1)
    idx = [0]
    i_reset_times = 0
    for i in range(len(all_list)):
        if i_reset_times:
            i = i - idx[-1] - i_reset_times
        if i == len(now_list):
            break
        if all_list[i] != now_list[i]:
            idx.append(i + idx[-1])
            now_list = now_list[i:]
            all_list = np.arange(now_list[0], now_list[-1] + 1)
            i_reset_times += 1
    idx.append(len(now_list_back))

    continuous_list = []
    idx_list = []
    for i in range(len(idx) - 1):
        continuous_list.append(now_list_back[idx[i]:idx[i + 1]])
        idx_list.append([idx[i], idx[i + 1]])
    return continuous_list, idx_list


if __name__ == '__main__':
    json_root = "/home/SENSETIME/chenjinsheng/Downloads/txt_hz_raw/2018-10-30-06-33-51/infos.json"
    car_json = parse_json(json_root)
    x = car_json[str(1)]['car_pose']
    print(x)
