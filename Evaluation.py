import os

import numpy as np
import pandas as pd

from dataloader import DataLoader


class Evaluation(object):
    """evaluation data here

    Attributes:
    """

    def __init__(self, frame_list):
        self._frame_list = frame_list
        self._mse = None
        self._mse_list = None
        self._mse_only = None
        self._mse_only_list = None
        self._pred_data = None

    def eval(self, gt_xy, pred_xy, pred_info, save_path, skip):
        """Eval data here"""
        mse, mse_only = self.mse(gt_xy, pred_xy, pred_info, skip)
        self.save_eval_result(save_path, 'mse')
        self.save_eval_result(save_path, 'mse_only')
        return mse, mse_only

    def mse(self, gt_xy, pred_xy, pred_info, skip):
        """mse metrics

        Args:
            gt_data: A dict contains gt x/y data
            pred_data: A dict contains pred x/y data
            prediction_frame_index_data: A dict contains prediction_frame_index_data
            skip: A int recordings skip

        Returns:
            mse_out: A numpy [m,n], m is number of frame, y is number of data
            mse_only_out: A numpy [m,n], m is number of frame, y is number of data

        """
        index = pred_xy.index
        pred_xy_np = pred_xy.values
        mse_out_list = mse_only_out_list = None
        for i, pred_element in enumerate(pred_xy_np):
            if index[i] not in gt_xy:
                continue
            gt_element = gt_xy[index[i]]
            start_fid = int(pred_info.ix[i, 1])
            pred_fid = int(pred_info.ix[i, 2])
            mse_list = mse_only_list = None
            for frame in self._frame_list:
                befor_len = pred_fid - start_fid
                after_len = len(gt_xy[index[i]]) // 2 - befor_len
                if (skip - 1 + after_len) >= frame * skip:
                    gt_element_np = gt_element.reshape(-1, 2)
                    gt_element_np = gt_element_np[
                                    (pred_fid - start_fid):
                                    (pred_fid - start_fid + (frame - 1) * skip + 1):skip, :]
                    gt_element_np = gt_element_np.reshape(-1)
                    mse = (pred_element[:frame * 2] - gt_element_np) ** 2

                    mse = np.sqrt(mse.reshape(-1, 2).sum(-1)).mean(0)
                    # print('mse is: ',mse)
                    mse_only = gt_element_np[-2:] - pred_element[(frame * 2 - 2):frame * 2]
                    mse_only = np.sqrt((mse_only ** 2).sum(-1))
                else:
                    # gt不存在
                    mse = mse_only = -1
                if mse_list is None:
                    mse_list = mse
                    mse_only_list = mse_only
                else:
                    mse_list = np.hstack((mse_list, mse))
                    mse_only_list = np.hstack((mse_only_list, mse_only))
            if mse_out_list is None:
                mse_out_list = mse_list
                mse_only_out_list = mse_only_list
            else:
                mse_out_list = np.vstack((mse_out_list, mse_list))
                mse_only_out_list = np.vstack((mse_only_out_list, mse_only_list))

        for i in range(0, mse_only_out_list.shape[1]):
            mse_only_tmp = mse_only_out_list[mse_only_out_list[:, i] != -1, i].mean(0)
            mse_tmp = mse_out_list[mse_out_list[:, i] != -1, i].mean(0)
            if i == 0:
                mse_only = mse_only_tmp
                mse = mse_tmp
            else:
                mse_only = np.hstack((mse_only, mse_only_tmp))
                mse = np.hstack((mse, mse_tmp))
        self._mse, self._mse_only = mse, mse_only
        self._mse_list, self._mse_only_list = mse_out_list, mse_only_out_list
        print("mse is: \n\n", mse)
        print("mse_only is: \n\n", mse_only)
        for i in range(0, mse_only_out_list.shape[1]):
            mse_tmp = mse_out_list[mse_out_list[:, i] != -1, i].std(0)
            if i == 0:
                mse_std = mse_tmp
            else:
                mse_std = np.hstack((mse_std, mse_tmp))
        print("mse_std is: \n\n", mse_std)
        for i in range(0, mse_only_out_list.shape[1]):
            mse_tmp = mse_out_list[mse_out_list[:, i] != -1, i].max()
            if i == 0:
                mse_max = mse_tmp
            else:
                mse_max = np.hstack((mse_max, mse_tmp))
        print("mse_max is: \n\n", mse_max)
        for i in range(0, mse_only_out_list.shape[1]):
            mse_tmp = mse_out_list[mse_out_list[:, i] != -1, i].min()
            if i == 0:
                mse_min = mse_tmp
            else:
                mse_min = np.hstack((mse_min, mse_tmp))
        print("mse_min is: \n\n", mse_min)
        return mse, mse_only

    def save_eval_result(self, save_path, metrics):
        """Save evaluation result here"""
        if not os.path.exists(save_path):
            os.system(r"touch {}".format(save_path))
        with open(save_path, 'a') as f:
            data = self.get_metrics(metrics, use_list=False)
            i = 0
            for frame in self._frame_list:
                f.write("%s_%d \n" % (metrics, frame))
                f.write("%s\n" % data[i])
                i += 1
            f.write("\n")

    def get_metrics(self, metrics, use_list=False):
        """Get metrics data here"""
        if use_list:
            if metrics == 'mse':
                return self._mse_list
            if metrics == 'mse_only':
                return self._mse_only_list
        else:
            if metrics == 'mse':
                return self._mse
            if metrics == 'mse_only':
                return self._mse_only

    def get_top_k(self, metrics, k, frame, num):
        """"Get top_k trace here

        Args:
            metrics: A string recording metrics name
            k: A int recoding top k number
            frame: A int list
        """
        metric_data = self.get_metrics(metrics, use_list=True)

        this_frame_data = metric_data[metric_data[:, num] != -1, :]
        this_frame_data = pd.DataFrame(this_frame_data,
                                       index=self._pred_data.index, columns=[str(frame) + 'eval'])
        data = pd.concat([this_frame_data, self._pred_data], axis=1)
        data.dropna()
        data.sort_values(by=str(frame) + 'eval')
        top_k = data.iloc[:k, :]
        down_k = data.iloc[-k:, :]

        return top_k, down_k

    def get_name(self, name):
        """Get metrics pandas name

        Args:
            name: a string recording prefix of metrics name
        """
        namelist = []
        for frame in self._frame_list:
            namelist.append(name + '_' + str(frame))
        return namelist


if __name__ == '__main__':
    from config import args
    origin = DataLoader()
    input_base = args.origin_input_base
    path = os.path.join(input_base, 'test.txt')
    actor = args.actor
    origin.variable_operate(input_base, path, actor)
    origin_x_y = origin.variable_get_data_with_window(2, each_len=2, stride=12)

    pred = DataLoader()
    input_base = args.result_input_base
    path = os.path.join(input_base, 'data_out.txt')
    pred.operate(input_base, path, actor)
    pred_x_y = pred.get_data_with_window(4, 2)
    pred_info = pred.get_data()

    frame_list = [4, 10]
    eva = Evaluation(frame_list)
    save_path = args.eval_path
    skip = 5

    eva.eval(origin_x_y, pred_x_y, pred_info, save_path, skip)
