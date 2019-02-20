import os
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataloader import DataLoader
from evaluation import Evaluation
from util import substring


class Attributes(object):
    def __init__(self, origin_data, origin_xy_data,
                 result_data, result_xy_data, config, orign_single_dict, frame_list):
        """Init data here
        Args:
            self._origin_data: A dict recording gt data
            self._origin_xy_data: A dict recording gt xy data
            self._result_data: A DataFrame recording pred data
            self._result_xy_data: A DataFrame recording pred xy data
            self._caculate_result: A DataFrame recording result data
            self._cfg: A Config Class recording config
            self._index: A Series recording index
            self._orign_single_dict: A DataFrame recording first pred data(used for making idx)
            self._frame_list: A list recording frame list
        """
        self._origin_data = origin_data
        self._origin_xy_data = origin_xy_data
        self._result_data = result_data
        self._result_xy_data = result_xy_data
        self._caculate_result = None
        self._cfg = config
        self._index = self._result_data.index
        self._orign_single_dict = orign_single_dict
        self._frame_list = frame_list

    def operate(self):
        """Operate Caculating here
        """
        self.velocity_mean()
        self.velocity_std()
        self.distance_mean()
        self.distance_std()
        self.direction()
        self.shake()
        self.crowd()

    def paint_operate(self, eva, eval_save_path, save_path):
        """Operate Painting here
        """
        self.paint_velocity_mean(eva, eval_save_path, save_path=save_path)
        self.paint_velocity_std(eva, eval_save_path, save_path=save_path)
        self.paint_direction(eva, eval_save_path, save_path=save_path)
        self.paint_shake(eva, eval_save_path, save_path=save_path)
        self.paint_crowd(eva, eval_save_path, save_path=save_path)

    def velocity_mean(self):
        """Caculate velocity mean
        """
        print("===== starting calculate velocity mean =====")
        vm = []
        result_data = self._result_data.values

        for i, each_result_data in enumerate(result_data):
            start_fid = int(each_result_data[1])
            pred_fid = int(each_result_data[2])

            each_gt_data = self._origin_xy_data[self._index[i]].reshape(-1, 2)
            now_idx = pred_fid - start_fid
            obs_len = int(each_result_data[3])
            each_gt_data = each_gt_data[(now_idx - obs_len * self._cfg['skip']):
                                        (now_idx + min((len(each_gt_data) - now_idx)
                                                       // self._cfg['skip'],
                                                       self._cfg['seq']) * self._cfg['skip']):
                                        self._cfg['skip']]
            x, y = each_gt_data[:, 0], each_gt_data[:, 1]
            d = self.caculate_distance(x, y)
            v = d / (0.1 * self._cfg['skip'])
            vm.append(v.mean())
        vm = pd.DataFrame(np.array(vm), index=self._index, columns=['velocity_mean'])
        self._caculate_result = vm if self._caculate_result is None \
            else pd.concat([self._caculate_result, vm])
        return vm

    def velocity_std(self):
        """Caculate velocity std
        """
        print("===== starting calculate velocity std =====")
        vs = []
        result_data = self._result_data.values
        for i, each_result_data in enumerate(result_data):
            start_fid = int(each_result_data[1])
            pred_fid = int(each_result_data[2])
            each_gt_data = self._origin_xy_data[self._index[i]].reshape(-1, 2)
            now_idx = pred_fid - start_fid
            obs_len = int(each_result_data[3])
            each_gt_data = each_gt_data[(now_idx - obs_len * self._cfg['skip']):
                                        (now_idx + min((len(each_gt_data) - now_idx)
                                                       // self._cfg['skip'],
                                                       self._cfg['seq']) * self._cfg['skip']):
                                        self._cfg['skip']]
            x, y = each_gt_data[:, 0], each_gt_data[:, 1]
            d = self.caculate_distance(x, y)
            v = d / (0.1 * self._cfg['skip'])
            vs.append(v.std())
        vs = pd.DataFrame(np.array(vs), index=self._index, columns=['velocity_std'])
        self._caculate_result = vs if self._caculate_result is None \
            else pd.concat([self._caculate_result, vs], axis=1)
        return vs

    def distance_mean(self):
        """Caculate distance mean
        """
        print("===== starting calculate distance mean =====")
        dm = []
        result_data = self._result_data.values
        for i, each_result_data in enumerate(result_data):
            start_fid = int(each_result_data[1])
            pred_fid = int(each_result_data[2])
            each_gt_data = self._origin_xy_data[self._index[i]].reshape(-1, 2)
            now_idx = pred_fid - start_fid
            obs_len = int(each_result_data[3])
            each_gt_data = each_gt_data[(now_idx - obs_len * self._cfg['skip']):
                                        (now_idx + min((len(each_gt_data) - now_idx)
                                                       // self._cfg['skip'],
                                                       self._cfg['seq']) * self._cfg['skip']):
                                        self._cfg['skip']]
            x, y = each_gt_data[:, 0], each_gt_data[:, 1]
            d = self.caculate_distance(x, y)
            dm.append(d.mean())
        dm = pd.DataFrame(np.array(dm), index=self._index, columns=['distance_mean'])
        self._caculate_result = dm if self._caculate_result is None else \
            pd.concat([self._caculate_result, dm], axis=1)
        return dm

    def distance_std(self):
        """Caculate distance std
        """
        print("===== starting calculate velocity std =====")
        ds = []
        result_data = self._result_data.values
        for i, each_result_data in enumerate(result_data):
            start_fid = int(each_result_data[1])
            pred_fid = int(each_result_data[2])
            each_gt_data = self._origin_xy_data[self._index[i]].reshape(-1, 2)
            now_idx = pred_fid - start_fid
            obs_len = int(each_result_data[3])
            each_gt_data = each_gt_data[(now_idx - obs_len * self._cfg['skip']):
                                        (now_idx + min((len(each_gt_data) - now_idx)
                                                       // self._cfg['skip'],
                                                       self._cfg['seq']) * self._cfg['skip']):
                                        self._cfg['skip']]
            x, y = each_gt_data[:, 0], each_gt_data[:, 1]
            d = self.caculate_distance(x, y)
            ds.append(d.std())
        ds = pd.DataFrame(np.array(ds), index=self._index, columns=['distance_std'])
        self._caculate_result = ds if self._caculate_result is None else \
            pd.concat([self._caculate_result, ds], axis=1)
        return ds

    def direction(self):
        """Caculate direction
        """
        print("===== starting calculate direction =====")
        directions = []
        result_data = self._result_data.values
        for i, each_result_data in enumerate(result_data):
            start_fid = int(each_result_data[1])
            pred_fid = int(each_result_data[2])
            each_gt_data = self._origin_xy_data[self._index[i]].reshape(-1, 2)
            now_idx = pred_fid - start_fid
            obs_len = int(each_result_data[3])
            each_gt_data = each_gt_data[(now_idx - obs_len * self._cfg['skip']):
                                        (now_idx + min((len(each_gt_data) - now_idx)
                                                       // self._cfg['skip'],
                                                       self._cfg['seq']) * self._cfg['skip']):
                                        self._cfg['skip']]
            x, y = each_gt_data[:, 0], each_gt_data[:, 1]
            vector = np.array([x[1:] - x[:-1], y[1:] - y[:-1]]).T
            base = np.array([x[-1] - x[0], y[-1] - y[0]])
            angle = self.caculate_angle_relative(base, vector)
            dangle = angle[1:] - angle[:-1]

            isleft = False
            left_sum_angle = 0.
            left_idx = np.where(dangle < -self._cfg['direction_monotonic_point_deg'])[0]
            if len(left_idx) > 0:
                left_idx, _ = substring(left_idx)
                left_num = np.array([len(item) for item in left_idx])
                max_idx = np.argmax(left_num)
                max_num = left_num[max_idx]
                left_sum_angle = abs(dangle[left_idx[max_idx][0]:left_idx[max_idx][-1] + 1].sum())
                isleft = True if max_num >= self._cfg['direction_monotonic_point_num'] else False

            isright = False
            right_sum_angle = 0.
            right_idx = np.where(dangle > self._cfg['direction_monotonic_point_deg'])[0]
            if len(right_idx) > 0:
                right_idx, _ = substring(right_idx)
                right_num = np.array([len(item) for item in right_idx])
                max_idx = np.argmax(right_num)
                max_num = right_num[max_idx]
                right_sum_angle = \
                    abs(dangle[right_idx[max_idx][0]:right_idx[max_idx][-1] + 1].sum())
                isright = True if max_num >= self._cfg['direction_monotonic_point_num'] else False

            d_threshold = self._cfg['direction_dist'] * (len(each_gt_data) / (self._cfg['seq'] * 2))
            if (x.max() - x.min()) < d_threshold and (y.max() - y.min()) < d_threshold:
                direction = 'static'
            elif isleft and left_sum_angle > self._cfg['direction_deg']:
                direction = 'left'
            elif isright and right_sum_angle > self._cfg['direction_deg']:
                direction = 'right'
            else:
                direction = 'straight'
            directions.append(direction)
        directions = pd.DataFrame(directions, index=self._index, columns=['directions'])
        self._caculate_result = directions if self._caculate_result is None else pd.concat(
            [self._caculate_result, directions], axis=1)
        return directions

    def shake(self):
        """Caculate shake data here
        """
        print("===== starting calculate shake =====")
        shakes = []
        ds = []
        result_data = self._result_data.values
        for i, each_result_data in enumerate(result_data):
            start_fid = int(each_result_data[1])
            pred_fid = int(each_result_data[2])
            each_gt_data = self._origin_xy_data[self._index[i]].reshape(-1, 2)
            now_idx = pred_fid - start_fid
            obs_len = int(each_result_data[3])
            each_gt_data = each_gt_data[(now_idx - obs_len * self._cfg['skip']):
                                        (now_idx + min((len(each_gt_data) - now_idx)
                                                       // self._cfg['skip'],
                                                       self._cfg['seq']) * self._cfg['skip']):
                                        self._cfg['skip']]
            x, y = each_gt_data[:, 0], each_gt_data[:, 1]
            xy = np.array([[[x[i], y[i]], [x[i + 1], y[i + 1]], [x[i + 2], y[i + 2]]]
                           for i in range(len(x) - 2)])
            a = np.sqrt(((xy[:, 0, :] - xy[:, 1, :]) ** 2).sum(1))
            b = np.sqrt(((xy[:, 1, :] - xy[:, 2, :]) ** 2).sum(1))
            c = np.sqrt(((xy[:, 2, :] - xy[:, 0, :]) ** 2).sum(1))
            p = (a + b + c) / 2
            s = np.sqrt(p * (p - a) * (p - b) * (p - c))
            d = 2 * s / b
            d[np.isnan(d)] = -1
            d = d.max()
            ds.append(d)
            if d < self._cfg['shake_d']:
                shake = 'smooth'
            else:
                shake = 'jitter'
            shakes.append(shake)
        shakes = pd.DataFrame(shakes, index=self._index, columns=['shakes'])
        self._caculate_result = shakes if self._caculate_result is None else pd.concat(
            [self._caculate_result, shakes], axis=1)
        return shakes

    def crowd(self):
        """Caculate crowd data here
        """
        crowds = []
        origin_dict = self._orign_single_dict
        new_dict = defaultdict(lambda: [])
        for idx, _ in origin_dict.items():
            folder, actor, _, start_fid = idx.split('_')[:4]
            new_dict[folder + '_' + actor + '_' + start_fid].append(idx)
        result_data = self._result_data.values
        for i, each_result_data in enumerate(result_data):
            crowd = 0
            idx = self._index[i]
            folder, actor, pid, _ = idx.split('_')[:4]
            pred_fid = int(each_result_data[2])
            this_data = np.array(origin_dict
                                 [folder + '_' + actor + '_' + pid + '_' + str(pred_fid)]
                                 [:2]).astype(np.float32)
            for around_idx in new_dict[folder + '_' + actor + '_' + str(pred_fid)]:
                around_pid = around_idx.split('_')[2]
                if int(around_pid) == int(pid):
                    continue
                around_data = np.array(origin_dict[around_idx][:2]).astype(np.float32)
                dist = np.sqrt(((this_data - around_data) ** 2).sum())
                if dist < self._cfg['crowd_dist']:
                    crowd += 1
            crowds.append(crowd)
        crowds = pd.DataFrame(crowds, index=self._index, columns=['crowds'])
        self._caculate_result = crowds if self._caculate_result is None else pd.concat(
            [self._caculate_result, crowds], axis=1)
        return crowds

    @staticmethod
    def caculate_distance(x, y):
        """Caculate distance in trace
        """
        vector = np.array([x[1:] - x[:-1], y[1:] - y[:-1]]).T
        distance = np.sqrt((vector ** 2).sum(1))
        return distance

    @staticmethod
    def caculate_angle_relative(base, vector):
        """caculate relative angles between base and vector

        Args:
            base: A numpy array with shape (2,)
            vector: A numpy array with shape (n,2)

        Returns:
            A numpy array with shape(n,)
        """

        def caculate_angley(vector):
            # vector np(n,2) 与y轴夹角0-360
            base = np.array([0., 1.])
            numerator = np.sum(vector * base, 1)
            denominator = np.sqrt(np.sum(base ** 2)) * np.sqrt(np.sum(vector ** 2, 1))
            denominator = np.clip(denominator, 1e-10, np.inf)
            cos = numerator / denominator
            angle = np.arccos(cos) * 180 / np.pi
            angle = np.clip(angle, 0., 180.)
            angle[vector[:, 0] < 0] = 360. - angle[vector[:, 0] < 0]
            return angle

        vector = np.concatenate([base.reshape(1, 2), vector], 0)
        angley = caculate_angley(vector)
        angle = (angley - angley[0])[1:]  # -360~360
        angle[angle > 180.] = angle[angle > 180.] - 360
        angle[angle < -180.] = angle[angle < -180.] + 360  # -180~180
        return angle

    def concate_trace_result(self, data, is_result=False):
        """Concate data with self._caculate_result

        return:
            new: A DataFrame recording caculate result
            caculate_size: A int recording numbers of result
        """
        new = pd.concat([data, self._caculate_result], axis=1)
        if is_result:
            self._caculate_result = new
        caculate_size = data.shape[1]
        return new, caculate_size

    def get_data(self):
        return self._result_data

    def get_xy_data(self):
        return self._result_xy_data

    def get_result_data(self):
        return self._caculate_result

    def get_index_data(self):
        return self._index

    def select_discrete_attr(self, index_data, name, attr):
        index = (index_data.ix[:, name] == attr)
        return index

    def select_continuous_attr(self, index_data, name, min_, max_):
        index = (index_data.ix[:, name] >= min_) & (index_data.ix[:, name] <= max_)
        return index

    def paint_direction(self, eva, eval_save_path, figsize=(30, 20),
                        save_path=None):
        attr = ['static', 'left', 'right', 'straight']
        save_path_pie = os.path.join(save_path, 'directions_pie.png')
        self.paint_discrete_pie('directions', attr, figsize, save_path_pie)
        save_path_bl = os.path.join(save_path, 'directions_broken_line.png')
        self.paint_discrete_trace(eva, 'directions', attr, save_path_bl, eval_save_path)

    def paint_shake(self, eva, eval_save_path, figsize=(30, 20),
                    save_path=None):
        attr = ['smooth', 'jitter']
        save_path_pie = os.path.join(save_path, 'shakes_pie.png')
        self.paint_discrete_pie('shakes', attr, figsize, save_path_pie)
        save_path_bl = os.path.join(save_path, 'shakes_broken_line.png')
        self.paint_discrete_trace(eva, 'shakes', attr, save_path_bl, eval_save_path)

    def paint_crowd(self, eva, eval_save_path,
                    figsize=(30, 20), save_path=None, interval=5, interval_part=None,
                    max_=1.):
        save_path_pie = os.path.join(save_path, 'crowd_pie.png')
        self.paint_continuous_pie('crowds', figsize, save_path_pie,
                                  interval, interval_part, max_)
        save_path_bl = os.path.join(save_path, 'crow_broken_line.png')
        self.paint_continuous_trace(eva, 'crowds', save_path_bl,
                                    eval_save_path, interval=interval)

    def paint_velocity_mean(self, eva, eval_save_path, figsize=(30, 20),
                            save_path=None, interval=5, interval_part=None,
                            max_=1.):
        save_path_pie = os.path.join(save_path, 'velocity_mean_pie.png')
        self.paint_continuous_pie('velocity_mean', figsize, save_path_pie,
                                  interval, interval_part, max_)
        save_path_bl = os.path.join(save_path, 'velocity_mean_broken_line.png')
        self.paint_continuous_trace(eva, 'velocity_mean', save_path_bl, eval_save_path, interval=5)

    def paint_velocity_std(self, eva, eval_save_path, figsize=(30, 20),
                           save_path=None, interval=5, interval_part=None,
                           max_=1.):
        save_path_pie = os.path.join(save_path, 'velocity_std_pie.png')
        self.paint_continuous_pie('velocity_std', figsize, save_path_pie,
                                  interval, interval_part, max_)
        save_path_bl = os.path.join(save_path, 'velocity_std_broken_line.png')
        self.paint_continuous_trace(eva, 'velocity_std', save_path_bl, eval_save_path, interval=5)

    def paint_discrete_pie(self, name, attr, figsize=(5, 5), save_path=None, data=None):
        if data is None:
            data = self._caculate_result[name].values
        else:
            data = data[name].values
        plt.figure(figsize=figsize)

        count = []
        for type_ in attr:
            count.append((data == type_).sum())
        count = np.array(count)
        percent = 100. * count / count.sum()
        plt.subplot(1, 1, 1)
        patches, _ = plt.pie(count, shadow=False, startangle=90)
        labels = ['{0}  {1:1.2f}%'.format(i, j) for i, j in zip(attr, percent)]
        plt.axis('equal')
        plt.legend(patches, labels, loc='center right', fontsize=30)
        if name != 'crowd':
            plt.title(name, fontsize='xx-large')
        else:
            pass
        if save_path is not None:
            plt.gcf().savefig(save_path, bbox_inches='tight')
        # plt.show()

    def paint_continuous_pie(self, name, figsize=(20, 20), save_path=None,
                             interval=30, interval_part=None, max_partial=1.,
                             data=None):
        if data is None:
            data = self._caculate_result[name].values
        else:
            data = data[name].values
        plt.figure(figsize=figsize)
        data_ = np.linspace(data.min(), data.max(), interval + 1)
        count = []
        for i in range(interval):
            min_, max_ = data_[i], data_[i + 1]
            num = ((data >= min_) & (data <= max_)).sum()
            count.append(num)
        count = np.array(count)
        percent = 100. * count / count.sum()
        if interval_part is None:
            plt.subplot(1, 1, 1)
        else:
            plt.subplot(1, 2, 1)
        patches, _ = plt.pie(count, shadow=False, startangle=90)
        labels = ['{0:1.2f}~{1:1.2f}   {2:1.3f}%'.format(data_[i],
                                                         data_[i + 1], percent[i])
                  for i in range(interval)]
        plt.axis('equal')
        plt.legend(patches, labels, loc='center left', fontsize=30)
        plt.title(name + ' (m/s)', fontsize=30)

        if interval_part is not None:
            data_ = np.linspace(data.min(), max_partial, interval_part + 1)
            count = []
            for i in range(interval_part):
                min_, max_ = data_[i], data_[i + 1]
                num = ((data >= min_) & (data <= max_)).sum()
                count.append(num)
            count = np.array(count)
            percent = 100. * count / count.sum()

            plt.subplot(1, 2, 2)
            patches, _ = plt.pie(count, shadow=False, startangle=90)
            labels = ['{0:1.2f}~{1:1.2f} m/s  {2:1.3f}%'.format(data_[i],
                                                                data_[i + 1], percent[i]) for i in
                      range(interval_part)]
            plt.axis('equal')
            plt.legend(patches, labels, loc='center right', fontsize=30)
            plt.title(name + '_partial (m/s)', fontsize=30)
        if save_path is not None:
            plt.gcf().savefig(save_path, bbox_inches='tight')
        # plt.show()

    def paint_discrete_trace(self, eva, name, attr, save_path,
                             eval_save_path, figsize=(30, 20), data=None, xy_data=None):
        if data is None and xy_data is None:
            data, data_num = self.concate_trace_result(self._result_data)
            data_xy, data_xy_num = self.concate_trace_result(self._result_xy_data)
        else:
            data, data_num = self.concate_trace_result(data)
            data_xy, data_xy_num = self.concate_trace_result(xy_data)
        plt.figure(figsize=figsize)

        mse_list = []
        mse_only_list = []
        for type_ in attr:
            print("====== evaluating ", str(type_), " mse =====")
            with open(eval_save_path, 'a') as f:
                f.write("====== evaluating " + str(type_) + " mse =====\n")
            this_attr_data = data.ix[(data[name] == type_), :]

            if len(this_attr_data.values) == 0:
                attr.remove(type_)
                continue
            this_data = this_attr_data.ix[:, :data_num]
            this_xy_attr_data = data_xy.ix[(data_xy[name] == type_), :]
            this_xy_data = this_xy_attr_data.ix[:, :data_xy_num]
            mse, mse_only = eva.eval(self._origin_xy_data, this_xy_data,
                                     this_data, eval_save_path, self._cfg['skip'])
            mse_list.append(mse)
            mse_only_list.append(mse_only)
        mse_list = np.array(mse_list)
        mse_only_list = np.array(mse_only_list)
        for i in range(0, mse_list.shape[1]):
            x_axis = [i for i in range(0, len(attr))]
            y_axis = mse_list[:, i]
            plt.plot(x_axis, y_axis, label="mse_" + str(self._frame_list[i]))
            for a, b in zip(x_axis, y_axis):
                plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=40)
            y_axis = mse_only_list[:, i]
            plt.plot(x_axis, y_axis, label="mse_only_" + str(self._frame_list[i]))
            for a, b in zip(x_axis, y_axis):
                plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=40)
            index = attr
            plt.xticks(x_axis, index, fontsize=40)
        plt.title(name + '_mse', fontsize=40)
        plt.yticks(fontsize=40)
        plt.legend(loc='best', fontsize=40)
        if save_path is not None:
            plt.gcf().savefig(save_path, bbox_inches='tight')
        # plt.show()

    def paint_continuous_trace(self, eva, name, save_path,
                               eval_save_path, interval=5,
                               figsize=(30, 20), data=None, xy_data=None):
        if data is None and xy_data is None:
            data, data_num = self.concate_trace_result(self._result_data)
            data_xy, data_xy_num = self.concate_trace_result(self._result_xy_data)
        else:
            data, data_num = self.concate_trace_result(data)
            data_xy, data_xy_num = self.concate_trace_result(xy_data)
        plt.figure(figsize=figsize)
        res = data[name].values
        data_ = np.linspace(res.min(), res.max(), interval + 1)
        mse_list = []
        mse_only_list = []
        index = []

        for i in range(interval):
            print("====== evaluating ", name, " ", str(i), " mse =====")
            with open(eval_save_path, 'a') as f:
                f.write("====== evaluating " + name + " " + str(i) + " mse =====\n")
            min_, max_ = data_[i], data_[i + 1]
            index.append(str(round(min_, 2)) + '~' + str(round(max_, 2)))
            this_attr_data = data.ix[(data[name] >= min_) & (data[name] <= max_), :]
            if this_attr_data is None:
                continue
            this_data = this_attr_data.ix[:, :data_num]
            this_xy_attr_data = data_xy.ix[(data_xy[name] >= min_) & (data_xy[name] <= max_), :]
            this_xy_data = this_xy_attr_data.ix[:, :data_xy_num]
            mse, mse_only = eva.eval(self._origin_xy_data,
                                     this_xy_data, this_data, eval_save_path, self._cfg['skip'])
            mse_list.append(mse)
            mse_only_list.append(mse_only)
        mse_list = np.array(mse_list)
        mse_only_list = np.array(mse_only_list)

        for i in range(0, mse_list.shape[1]):
            x_axis = [i for i in range(interval)]
            y_axis = mse_list[:, i]
            plt.plot(x_axis, y_axis, label="mse_" + str(self._frame_list[i]))
            for a, b in zip(x_axis, y_axis):
                plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=20)
            y_axis = mse_only_list[:, i]
            plt.plot(x_axis, y_axis, label="mse_only_" + str(self._frame_list[i]))
            plt.xticks(x_axis, index, fontsize=40)
            for a, b in zip(x_axis, y_axis):
                plt.text(a, b, round(b, 3), ha='center', va='bottom', fontsize=20)
        plt.title(name + '_mse', fontsize=40)
        plt.legend(loc='best', fontsize=40)
        plt.yticks(fontsize=40)
        if save_path is not None:
            plt.gcf().savefig(save_path, bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    class Config():
        config_ped = {
            'direction_monotonic_point_deg': 0.2,
            'direction_monotonic_point_num': 5,
            'direction_dist': 0.5,
            'direction_deg': 5,
            'shake_d': 3.0,  # 3
            'skip': 5,
            'seq': 10,
            'crowd_dist': 5
        }

    @click.command()
    @click.option('--origin_path', default="../../data/origin_tc/test_selected",
                  help="A string recording origin path")
    @click.option('--result_path', default="../../data/result_tc/test_selected/result_skip5",
                  help="A string recording result path")
    @click.option('--eval_path', default="./eval.txt",
                  help="A string recording eval txt path")
    @click.option('--paint_save_path', default="../../data/paint/model_1",
                  help="A string recording image saved path")
    @click.option('--actor', default="ped",
                  help="A string recording actor to evaluate")
    @click.option('--is_eval_all', default=True, type=str,
                  help="all attr to eval")
    @click.option('--is_group_attr', default=False, help="if make group attr")
    def start(origin_path, result_path, eval_path,
     paint_save_path, actor, is_eval_all, is_group_attr):
        attr_name = ['velocity_mean', 'shake']
        origin = DataLoader()
        origin.variable_operate(origin_path, os.path.join(origin_path, 'test.txt'), actor)
        origin_data = origin.variable_get_data()
        origin_xy_data = origin.variable_get_data_with_window(2, each_len=2, stride=12)
        orign_single_dict = origin.variable_convert_data_to_dict(origin_data, stride=12)
        pred = DataLoader()
        pred.operate(result_path, os.path.join(result_path, "data_out.txt"), actor)
        pred_data = pred.get_data()
        pred_xy_data = pred.get_data_with_window(4, 2)
        config = getattr(Config, 'config_ped')
        frame_list = [4, 10]
        eva = Evaluation(frame_list)
        eval_save_path = eval_path
        image_save_path = paint_save_path
        with open(eval_save_path, 'a') as f:
            f.write("skip is %d\n" % config['skip'])
        attr = Attributes(origin_data, origin_xy_data,
                          pred_data, pred_xy_data, config, orign_single_dict, frame_list)
        if is_eval_all:
            attr.operate()
            attr.paint_operate(eva, eval_save_path, save_path=image_save_path)
        else:
            for type_ in attr_name:
                getattr(attr, type_)()
                getattr(attr, "paint_" + type_)(eva, eval_save_path, save_path=image_save_path)

        if is_group_attr:
            # smooth & v <= 3
            result_data = attr.get_result_data()

            index1 = attr.select_continuous_attr(result_data, 'velocity_mean', 0, 3)
            index2 = attr.select_discrete_attr(result_data, 'shakes', 'smooth')
            index = (index1 & index2)
            index = index.to_frame()
            index.columns = ["un_normal"]
            attr.concate_trace_result(index, is_result=True)
            attr_type = [True, False]
            save_path_pie = os.path.join(paint_save_path, 'un_normal_pie.png')
            attr.paint_discrete_pie("un_normal", attr_type, save_path=save_path_pie)
            save_path_pie = os.path.join(paint_save_path, 'un_normal_trace.png')
            attr.paint_discrete_trace(eva, "un_normal",
             attr_type, save_path=save_path_pie, eval_save_path=eval_save_path)

    # pylint: disable=no-value-for-parameter
    start()
