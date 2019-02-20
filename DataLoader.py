import os
import sys

import numpy as np
import pandas as pd


class DataLoader(object):
    def __init__(self):
        self._data = None
        self._folder_name = None
        self._skip = None
        self._actor = None
        self._input_base = None
        # self.operate(input_base, path, actor)
        self._key = None
        self._data_dict = None

    def load_filename(self, path):
        """"Load filename from paths

        Args:
            path: A string recording path

        Return:
        """
        folders_name = np.array([line.rstrip('\n').split(' ')
                                 for line in open(path, 'r')]).astype(np.str)
        self._folder_name = folders_name[:, 0]
        if folders_name.shape[1] > 2:
            self._skip = folders_name[:, 1].astype(np.int)
            # self._actor = folders_name[:, 2]

    def load_data(self, name):
        """"Load data from name

        Args:
            name: A string recording path name

        Returns:
            data: A list recording data
        """

        data = np.array([line.rstrip('\n').split(' ')
                         for line in open(name, 'r')]).astype(np.float32)
        return data

    def file_load_data(self, key_1=0, key_2=1):
        """"Load whole data from folder
        """

        def make_key(x, y):
            return self._key + '_' + str(int(x)) + '_' + str(int(y))

        for folder in self._folder_name:
            file_list = self.file_name(folder)
            for file in file_list:
                data = pd.DataFrame(self.load_data(file))
                self._key = str(folder) + '_' + str(self._actor)
                data.index = data.apply(lambda row:
                                        make_key(row.iloc[key_1], row.iloc[key_2]), axis=1)

                if self._data is None:
                    self._data = data
                else:
                    self._data = pd.concat([self._data, data], axis=0)

    def file_name(self, file_dir):
        """Get file_name of directory

        Args:
            file_dir: directory name

        Returns:
            file_list: file name list
        """
        file_list = []
        for file in os.listdir(os.path.join(self._input_base, file_dir)):
            if file.split('.')[1] != 'txt':
                continue
            if file.split('.')[0] == 'res' or file.split('.')[0] == self._actor:
                file_list.append(os.path.join(self._input_base, file_dir, file))
        return file_list

    def operate(self, input_base, path, actor, foldername=None, key1=0, key2=1):
        """Operate here
        """
        if foldername is None:
            self.load_filename(path)
        else:
            self._folder_name = foldername
        self._actor = actor
        self._input_base = input_base
        self.file_load_data(key1, key2)

    def get_data(self):
        return self._data

    def get_folder_name(self):
        return self._folder_name

    def get_data_with_window(self, start_frame, each_len=None,
                             stride=None, times=None, data=None):
        """"Get data with slide window

        Args:
            start_frame: A int recording start_frame position
            each_len: A int recording each window size
            stride: A int recording each stride
            times: A int recording sliding times

        Return:
            data_dict: A pandas
        """
        if data is None:
            data = self._data.iloc[:, start_frame:]
        else:
            data = data.iloc[:, start_frame:]

        if each_len is None:
            return data
        if stride is None:
            # return data.iloc[:, :each_len * 2]
            stride = 2
        if times is None:
            times = sys.maxsize

        end_pos = int(min((times - 1) * stride + each_len, data.shape[1]))

        data_result = data.iloc[:, 0: each_len]
        i = stride
        while i < end_pos:
            data_piece = data.iloc[:, i:i + each_len]
            data_result = pd.concat([data_result, data_piece], axis=1)
            i += stride

        return data_result

    def variable_operate(self, input_base, path, actor, key1=0, key2=1):
        self._actor = actor
        self._input_base = input_base
        self.load_filename(path)
        self.variable_file_load_data(actor, key1, key2)

    def variable_load_data(self, name):
        """"Load data from name

        Args:
            name: A string recording path name

        Returns:
            data: A list recording data
        """
        data = []
        data.append([line.rstrip('\n').split(' ') for line in open(name, 'r')])
        data = data[0]
        self._data = data
        return data

    def variable_file_load_data(self, actor, key_1=0, key_2=1):
        """"Load whole data from folder
        """
        data_dict = {}
        for i in range(0, len(self._folder_name)):
            file_list = self.file_name(self._folder_name[i])
            for file in file_list:
                self.variable_load_data(file)
                # actor = file.split('/')[-1].split('.')[0]
                # if actor == 'res':
                #    actor = self._actor[0]
                dict_tmp = self.variable_make_unique_idx(self._folder_name[i], actor, key_1, key_2)

                if len(data_dict) == 0:
                    data_dict = dict_tmp
                else:
                    data_dict = data_dict.copy()
                    data_dict.update(dict_tmp)
        self._data_dict = data_dict

    def variable_get_data(self):
        return self._data_dict

    def variable_make_unique_idx(self, folder_name, actor, pid_pos, start_frame_index_pos):
        """Make unique idx

        Args:
            pid_pos: A int recording pid position
            start_frame_index_pos: A int recording start_frameIndex position

        Return:
        """
        dict_tmp = {str(folder_name) + '_' + str(actor) +
                    '_' + str(int(float(self._data[n][pid_pos]))) + "_"
                    + str(int(float(self._data[n][start_frame_index_pos]))):
                        self._data[n][:]
                    for n in range(0, len(self._data))}
        return dict_tmp

    def variable_get_data_with_window(self, start_frame, each_len=None,
                                      stride=None, times=None):
        """"Get data with slide window

        Args:
            pid_pos: A int recording pid position
            start_frame: A int recording start_frame position
            each_len: A int recording each window size
            stride: A int recording each stride
            times: A int recording sliding times

        Return:
            data_dict: A dict mapping unique id to window data
        """
        if stride is None:
            stride = [each_len if times != 1 else sys.maxsize][0]

        def end_func(x, y):
            return min((start_frame + x * stride),
                       len(y)) if x is not None else len(y)

        xy_data_dict = {}
        for key in self._data_dict:
            # actor = key.split('_')[1]
            # if actor == 'veh':
            #     stride += 1
            key_value = self._data_dict[key]
            end_pos = end_func(times, key_value)
            for t in range(0, (end_pos - start_frame) // stride + 1):
                if each_len is None:
                    xy_data_dict[key] = np.array(key_value[start_frame:]).astype(np.float32)
                    break
                if t == 0:
                    xy_data_dict[key] = np.array(key_value[(start_frame + stride * t):
                                                           (start_frame + each_len + stride * t)]) \
                        .astype(np.float32)
                else:
                    value = np.array(key_value[(start_frame + stride * t):
                                               (start_frame + each_len + stride * t)]) \
                        .astype(np.float32)
                    xy_data_dict[key] = np.hstack((xy_data_dict[key], value))

            # if actor == 'veh':
            #     stride -= 1
        return xy_data_dict

    def variable_convert_data_to_dict(self, data, stride):
        new_dict = {}
        for idx in data:
            folder, actor, pid, start_fid = idx.split('_')[:4]
            real_num = (len(data[idx]) - 2) // stride
            for i in range(0, real_num):
                this_id = i + int(start_fid)
                new_dict[folder + '_' + actor + '_' + pid + '_' + str(this_id)]\
                    = data[idx][(2 + i * stride):(
                            2 + (i + 1) * stride)]
        return new_dict


if __name__ == '__main__':
    from config import args
    origin = DataLoader()
    input_base = args.origin_input_base
    origin_info_path = os.path.join(input_base, 'test.txt')
    actor = args.actor
    origin.variable_operate(args.origin_input_base, origin_info_path, actor)
    origin.variable_convert_data_to_dict(origin.variable_get_data(), stride=12)
    origin_x_y = origin.variable_get_data_with_window(2, each_len=2, stride=13)

    result = DataLoader()
    input_base = args.result_input_base
    result_info_path = os.path.join(input_base, 'data_out.txt')
    result.operate(input_base, result_info_path, actor)
    result_x_y = result.get_data_with_window(4, each_len=2)
