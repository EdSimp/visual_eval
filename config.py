import argparse

parse = argparse.ArgumentParser(description="Args of folder path for vis")
parse.add_argument('--origin_input_base',
                   default="/mnt/lustrenew/share/zhangshiquan/AP2/Temp-AllInOne/ap2/",
                   help="origin input base")
parse.add_argument('--result_input_base',
                   default="/mnt/lustrenew/share/zhangshiquan"
                           "/AP2/RD_code/pedestrian_prediction_general/result",
                   help="result input base")
parse.add_argument('--rd_root',
                   default="/mnt/lustre/share/fangliangji/hz_data/output_sdk/road_recovery_result/",
                   help="rd_root")
parse.add_argument('--video_root',
                   default='/mnt/lustre/share/'
                           'fangliangji/hz_data/output_sdk/object_recovery_result',
                   help="video save root")
parse.add_argument('--image_root',
                   default="/mnt/lustrenew/share/chenjinsheng/image",
                   help="image save root")
parse.add_argument('--save_path',
                   default="/mnt/lustrenew/share/chenjinsheng/result",
                   help="video result save path")
parse.add_argument('--start_fid',
                   default=500,
                   help="start from this fid")
parse.add_argument('--end_fid',
                   default=2000,
                   help="end from this fid")
parse.add_argument('--is_saveimage',
                   default=False,
                   help="if save image, (default:True)")
parse.add_argument('--save_image_num', default=1500,
                   help="image number to save")

parse.add_argument('--actor', default='ped', help='actor type:ved/peh')

parse.add_argument('--eval_path',
                   default="./eval.txt",
                   help="A string recording eval txt path")

args = parse.parse_args()
