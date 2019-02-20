import json
import os

import cv2
import numpy as np

from DataLoader import DataLoader
from util import coordinateTrans, parse_json


class VideoPre(object):
    def __init__(self, folders, image_root, video_root):
        self._folders = folders
        self._image_root = image_root
        self._video_root = video_root

    def get_image(self, end_fid=None):
        """Save images from video
        """
        for folder in self._folders:
            print("This folder is: %s" % folder)
            if not os.path.exists(os.path.join(self._image_root, folder)):
                os.makedirs(os.path.join(self._image_root, folder))
            # self.del_file(folder)
            cap = cv2.VideoCapture(
                os.path.join(self._video_root, folder, 'object_recovery_video.avi'))
            fid = 0
            while cap.isOpened():
                status, image = cap.read()
                if status:
                    if fid % 20 == 0:
                        print("save ", fid)
                    if not os.path.exists(os.path.join(image_root, folder, str(fid) + '.jpg')):
                        cv2.imwrite(os.path.join(image_root, folder, str(fid) + '.jpg'), image)
                    else:
                        continue
                    if end_fid is not None:
                        if fid == end_fid:
                            break
                else:
                    print(fid, " break")
                    break
                fid += 1
            cap.release()
            cv2.destroyAllWindows()
        return fid

    def del_file(self, folder):
        file_list = os.listdir(os.path.join(self._image_root, folder))
        for file in file_list:
            os.remove(os.path.join(self._image_root, folder, file))


class Visualization(object):
    """Visualization
    """

    def __init__(self, folders, video_root, image_root):
        self._save_path = None
        self._address = None
        self._image_root = image_root
        self._video_root = video_root
        self._folders = folders
        self._skip = None
        self._len = None
        # self.json_root
        self._PIXEL_TO_M = 0.08
        self._OBS_WIDTH = 20
        self._OBS_BACK = 50
        self._OBS_FRONT = 30
        self._H = int((self._OBS_BACK + self._OBS_FRONT) / self._PIXEL_TO_M)
        self._W = int(self._OBS_WIDTH * 2 / self._PIXEL_TO_M)
        self._CAR_COLOR = (255, 255, 0)
        self._OBS_COLOR = (0, 0, 255)
        self._GT_COLOR = (0, 255, 255)
        self._PRED_COLOR = (0, 255, 0)
        self._FONT_COLOR = (255, 255, 255)
        self._offx = 10
        self._offy = 95
        self._JUNCTION_COLOR = (100, 100, 100)
        self._LINE_COLOR = (225, 225, 225)
        self._LINE_WIDTH = 5
        self._PAIR_COLOR = (100, 100, 100)
        self._CROSSWALK_COLOR = (225, 225, 225)
        """
        if address == 'hz':
            self._H = 800
            self._W = 2158
            self._image_W = 1758
        else:
            self._H = 1208
            self._W = 2508
            self._image_W = 1920
        """

    def vis_config(self, skip, len_):
        self._skip = skip
        self._len = len_

    def vis(self, gt_xy_data, pred_data,
            pred_xy_data, address, folder, save_path, json_root, start_fid,
            end_fid, actor, rd_root):
        """Visulization
        """
        # pred_data = pred_data.sort_values(by=2)
        save_folder = os.path.join(save_path, folder + '.avi')
        print(save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        out = cv2.VideoWriter(save_folder,
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              10, (int(self._H * 20 / 9), self._H))

        car_json = parse_json(json_root)

        for mid_fid in range(start_fid, end_fid + 1):
            image_path = os.path.join(self._image_root, folder, str(mid_fid) + '.jpg')
            image = cv2.imread(image_path)
            if address == 'sh':
                if image.shape[1] != 1920:
                    image = image[:, :1920:, :]
            else:
                if image.shape[1] != 1758:
                    image = image[:, :1758, :]
            image = cv2.resize(image, (int(self._H * 20 / 9) - self._W, self._H))
            img = np.zeros((self._H, self._W, 3), dtype=np.uint8)

            img = self.draw_rd(img, mid_fid, rd_root, folder, car_json)

            car_pose = car_json[str(mid_fid)]['car_pose']
            car_coor = np.array(car_pose[:2])
            car_head = car_pose[-1]

            pred_fid = mid_fid + self._skip
            pred_data_now = pred_data.ix[(pred_data.ix[:, 2] == pred_fid), :]
            pred_xy_data_now = pred_xy_data.ix[(pred_data.ix[:, 2] == pred_fid), :]

            for i in range(max(mid_fid - self._len, 0), mid_fid + 1):
                car_pose = car_json[str(i)]['car_pose']
                car_tmp = np.array(car_pose[:2])
                if i == max(mid_fid - self._len, 0):
                    car_trace = car_tmp
                else:
                    car_trace = np.vstack((car_trace, car_tmp))
            car_trace = car_trace.reshape(-1, 2)
            car_trace = coordinateTrans(car_trace, car_head, car_coor)
            img = self.draw_trace(img, car_trace, self._CAR_COLOR)
            if mid_fid % 20 == 0:
                print(mid_fid)

            for i in range(0, pred_data_now.shape[0]):
                index = pred_data_now.index[i]
                pid = int(pred_data_now.ix[i, 0])

                start_fid = int(pred_data_now.ix[i, 1])
                obs_len = int(pred_data_now.ix[i, 3])
                former_pos = mid_fid - obs_len * self._skip - start_fid
                rear_pos = mid_fid + (obs_len - 1) * self._skip - start_fid
                print("index is : ", index)
                if index not in gt_xy_data:
                    continue
                print(mid_fid, " is true")
                gt_data = gt_xy_data[index].reshape(-1, 2)
                gt_data_history = gt_data[max(0, former_pos):
                                          min(len(gt_data), mid_fid - start_fid + 1):]
                gt_data_future = gt_data[min(len(gt_data), mid_fid - start_fid + 1):
                                         min(len(gt_data), rear_pos):]

                pred_trace_data = pred_xy_data_now.ix[i, :].values
                pred_trace_data = pred_trace_data.reshape(-1, 2)

                gt_single_data_history = coordinateTrans(gt_data_history, car_head, car_coor)
                gt_single_data_future = coordinateTrans(gt_data_future, car_head, car_coor)
                pred_trace_data = coordinateTrans(pred_trace_data, car_head, car_coor)
                img = self.draw_trace(img, gt_single_data_history, self._OBS_COLOR)
                img = self.draw_trace(img, gt_single_data_future, self._GT_COLOR)
                img = self.draw_trace(img, pred_trace_data, self._PRED_COLOR)

                cv2.putText(img, text=str(int(pid)),
                            org=(int((self._OBS_WIDTH -
                                      gt_single_data_history[-1, 1]) / self._PIXEL_TO_M)
                                 + self._offx,
                                 int((self._OBS_FRONT - gt_single_data_history[
                                     -1, 0]) / self._PIXEL_TO_M) + 500 - self._offy),
                            fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                            color=self._FONT_COLOR, thickness=1)
                # tmp = gt_origin[index][(4+(mid_fid-start_fid)*12):(4+(mid_fid-start_fid)*12+2)]
                # img = self.draw_bdfd(gt_single_data_history, img, tmp, car_head, (120, 120, 120))
                # tmp = gt_origin[index][(6+(mid_fid-start_fid)*12):(6+(mid_fid-start_fid)*12+2)]
                # img = self.draw_bdfd(gt_single_data_history, img, tmp, car_head, (180, 180, 180))
            img = self.put_text(img, False, actor)

            out.write(np.concatenate([image, img.astype(np.uint8)], 1))
        out.release()
        cv2.destroyAllWindows()

    def draw_bdfd(self, gt_single_data_history, img, tmp, a, color):
        tmp = np.array(tmp).astype(np.float32).reshape(-1, 2)
        tmp = coordinateTrans(tmp, a)[0]
        pt1 = (int((self._OBS_WIDTH -
                    gt_single_data_history[-1, 1]) / self._PIXEL_TO_M) + self._offx,
               int((self._OBS_FRONT - gt_single_data_history[-1, 0])
                   / self._PIXEL_TO_M) - self._offy)
        pt2 = (int((self._OBS_WIDTH -
                    gt_single_data_history[-1, 1] + 3 * tmp[1]) / self._PIXEL_TO_M) + self._offx,
               int((self._OBS_FRONT - gt_single_data_history[-1, 0]
                    + 3 * tmp[0]) / self._PIXEL_TO_M) - self._offy)
        cv2.line(img, pt1=pt1, pt2=pt2, color=color, thickness=2)
        return img

    def draw_trace(self, img, trace, color):
        """Draw trace here

        Args:
            img: A img to draw
            trace: A numpy recording xy trace, each line recording a couple of xy
            color: A tuple recording color
        """
        for i in range(len(trace) - 1):
            # pt1, pt2
            pt1 = (int((self._OBS_WIDTH - trace[i, 1]) / self._PIXEL_TO_M) + self._offx,
                   int((self._OBS_FRONT - trace[i, 0]) / self._PIXEL_TO_M) + 500 - self._offy)
            pt2 = (int((self._OBS_WIDTH - trace[i + 1, 1]) / self._PIXEL_TO_M) + self._offx,
                   int((self._OBS_FRONT - trace[i + 1, 0]) / self._PIXEL_TO_M) + 500 - self._offy)
            cv2.line(img=img, pt1=pt1, pt2=pt2, color=color, thickness=1)
            x, y = pt1
            if 0 <= y - 1 < self._H and 0 <= y + 1 < self._H\
                    and 0 <= x - 1 < self._W and 0 <= x + 1 < self._W:
                img[y - 1:y + 1, x - 1:x + 1, :] = color
        if len(trace) > 0:
            x, y = (int((self._OBS_WIDTH - trace[-1, 0]) / self._PIXEL_TO_M) + self._offx,
                    int((self._OBS_FRONT - trace[-1, 1]) / self._PIXEL_TO_M) + 500 - self._offy)
            if 0 <= y - 1 < self._H and 0 <= y + 1 < self._H\
                    and 0 <= x - 1 < self._W and 0 <= x + 1 < self._W:
                img[y - 1:y + 1, x - 1:x + 1, :] = color

        return img

    def put_text(self, img, without_gt, actor):
        """Put annotation text

        Args:
            img: A numpy recording palette
            without_gt: A boolean recording whether with or without gt
            actor: A string recording actor
        """

        if not without_gt:
            cv2.putText(img=img, text='myself',
                        org=(20, 970), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=self._CAR_COLOR, thickness=2)
            cv2.putText(img=img, text=actor + ' history',
                        org=(20, 970 - 15), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=self._OBS_COLOR,
                        thickness=2)
            cv2.putText(img=img, text=actor + ' groudtruth',
                        org=(20, 970 - 30), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=self._GT_COLOR,
                        thickness=2)
            cv2.putText(img=img, text=actor + ' prediction',
                        org=(20, 970 - 45), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=self._PRED_COLOR,
                        thickness=2)
        else:
            cv2.putText(img=img, text='myself',
                        org=(20, 970), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=self._CAR_COLOR, thickness=1)
            cv2.putText(img=img, text=actor + ' history',
                        org=(20, 970 - 15), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=self._OBS_COLOR,
                        thickness=1)
            cv2.putText(img=img, text=actor + ' prediction',
                        org=(20, 970 - 30), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1, color=self._PRED_COLOR,
                        thickness=1)
        return img

    def draw_rd(self, img, mid_fid, rd_root, folder, car_json):

        def trans2img(points):
            y = (self._OBS_FRONT - points[:, 0]) / self._PIXEL_TO_M
            x = (self._OBS_WIDTH - points[:, 1]) / self._PIXEL_TO_M
            points = np.array([x, y]).T
            return points

        has_road, rd_path = car_json[str(mid_fid)]['rd']
        rd_path = os.path.join(rd_root, folder, rd_path)
        if has_road == 1:
            print("has road")
            rodas = json.load(open(rd_path, 'r'))['value0']
            # junction
            junction = []
            for item in rodas['junction']['junction_boundary']:
                xy = [item['x'], item['y']]
                junction.append(xy)
            # lane_pairs
            lane_pairs = []
            crosswalks_points = []
            for road_map in rodas['road_map']:
                lanes = {}
                cw_points = []
                for lane_line_map in road_map['value']['lane_line_map']:
                    points = []
                    for point in lane_line_map['value']['data']['laneline_point_set']:
                        xy = [point['x'], point['y']]
                        points.append(xy)
                    key = lane_line_map['key']
                    lanes[key] = points
                for lane_map in road_map['value']['lane_map']:
                    left_key = lane_map['value']['left_laneline_idx']
                    right_key = lane_map['value']['right_laneline_idx']
                    lane_pair = {'left': lanes[left_key], 'right': lanes[right_key]}
                    lane_pairs.append(lane_pair)
                for crosswalks in road_map['value']['crosswalks']:
                    for each in crosswalks['point_set']:
                        xy = [each['x'], each['y']]
                        cw_points.append(xy)
                crosswalks_points.append(cw_points)

            # draw junction
            if len(junction) > 0:
                junction = trans2img(np.array(junction))
                junction[:, 1] = junction[:, 1] + 350
                cv2.fillPoly(img, [junction.astype(np.int64)], color=self._JUNCTION_COLOR)
            # draw lane_mask/lane_line
            for lane_pair in lane_pairs:
                left_lane, right_lane = lane_pair['left'], lane_pair['right']
                pair = left_lane + list(reversed(right_lane))
                if len(left_lane) > 0:
                    left_lane = trans2img(np.array(left_lane))
                    for i in range(len(left_lane) - 1):
                        cv2.line(img, (int(left_lane[i, 0]), int(350 + left_lane[i, 1])),
                                 (int(left_lane[i + 1, 0]), int(350 + left_lane[i + 1, 1])),
                                 color=self._LINE_COLOR, thickness=self._LINE_WIDTH)
                if len(right_lane) > 0:
                    right_lane = trans2img(np.array(right_lane))
                    for i in range(len(right_lane) - 1):
                        cv2.line(img, (int(right_lane[i, 0]), int(350 + right_lane[i, 1])),
                                 (int(right_lane[i + 1, 0]), int(350 + right_lane[i + 1, 1])),
                                 color=self._LINE_COLOR, thickness=self._LINE_WIDTH)
                if len(pair) > 0:
                    pair = trans2img(np.array(pair))
                    pair[:, 1] = pair[:, 1] + 350
                    cv2.fillPoly(img, [pair.astype(np.int64)], color=self._PAIR_COLOR)

            for each_cross in crosswalks_points:
                if len(each_cross) > 0:
                    each_cross = trans2img(np.array(each_cross))
                    each_cross[:, 1] = each_cross[:, 1] + 350
                    for i in range(len(each_cross) - 1):
                        cv2.line(img, (int(each_cross[i, 0]), int(each_cross[i, 1])),
                                 (int(each_cross[i + 1, 0]), int(each_cross[i + 1, 1])),
                                 color=self._CROSSWALK_COLOR, thickness=self._LINE_WIDTH)
                    cv2.fillPoly(img, [each_cross[:4, :].astype(np.int64)],
                                 color=self._CROSSWALK_COLOR)
        return img


if __name__ == "__main__":
    from config import args

    origin = DataLoader()
    origin_input_base = args.origin_input_base
    origin_info_path = os.path.join(origin_input_base, 'test.txt')
    actor = args.actor
    origin.variable_operate(origin_input_base, origin_info_path, actor)
    origin_x_y = origin.variable_get_data_with_window(2, each_len=2, stride=12)
    # origin_data = origin.variable_get_data()

    result_input_base = args.result_input_base
    result_info_path = os.path.join(result_input_base, 'data_out.txt')
    folders = origin.get_folder_name()

    count = 0
    for folder in folders:
        if count < 2:
            count += 1
            continue
        result = DataLoader()
        rd_root = args.rd_root
        result.operate(result_input_base, result_info_path, actor, foldername=[folder])
        result_x_y = result.get_data_with_window(4, 2)
        pred_info = result.get_data()

        video_root = args.video_root
        image_root = args.image_root
        save_path = args.save_path
        json_root = os.path.join(origin_input_base, folder, 'infos.json')

        address = 'hz'
        start_fid = args.start_fid

        if args.is_saveimage:
            video = VideoPre([folder], image_root, video_root)
            end_fid_ = video.get_image(args.save_image_num)
            # end_fid = video.get_image() - 1
        else:
            save_folder_path = os.path.join(image_root, folder)
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            count = 0
            for fn in os.listdir(save_folder_path):
                if fn[0] != ".":
                    count += 1
            end_fid = count

        print("min: ", args.end_fid, " , ", end_fid_ - 2)
        end_fid = min(args.end_fid, end_fid_ - 2)

        attr = Visualization(folders=folders, video_root=video_root, image_root=image_root)
        attr.vis_config(3, 20)

        attr.vis(gt_xy_data=origin_x_y,
                 pred_data=pred_info, pred_xy_data=result_x_y,
                 address=address, folder=folder, json_root=json_root,
                 save_path=save_path, start_fid=start_fid, end_fid=end_fid,
                 actor=actor, rd_root=rd_root)
