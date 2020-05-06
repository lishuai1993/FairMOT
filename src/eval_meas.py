import os
import os.path as osp
import numpy as np
from tqdm import tqdm

import _init_paths
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
import motmetrics as mm
from tracking_utils.evaluation import Evaluator





def measure2files(results_root, data_root, seqs):
    """

    @param results_root:        mot轨迹预测结果文件的根目录，文件中格式 <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    @param data_root:           gt文件的路径，不包含后三级，因为在Evaluator初始化函数中已经写了全路径的拼接方式
    @param seqs:                gt路径的倒数第三级路径
    @return:                    存储seqs中，每一个路径下track结果的评价指标，以及全部文件夹汇总后的指标
    """

    data_type = 'mot'
    result_root = "/home/shuai.li/code/FairMOT/MOT15/images/results/temp/"
    exp_name = "test_evalMot15"

    accs = []
    # eval
    for seq in tqdm(seqs):

        result_filename = osp.join(results_root, seq) + '.txt'
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)    # 在初始化中，根据data_root, seq自动加载ground truth数据
        accs.append(evaluator.eval_file(result_filename))   # 在读取存储的检测结果，并进行计算，一帧对应一个acc对象
        # if save_videos:
        #     output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
        #     cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
        #     os.system(cmd_str)

    # get summary
    metrics = mm.metrics.motchallenge_metrics                       # 18个评价指标
    mh = mm.metrics.create()                                        # 创建指标计算工厂，后续传入相关数据即可给出指标
    summary = Evaluator.get_summary(accs, seqs, metrics)            # 计算MOT challenge中的指标，参数：、eval的帧序列名称、指标序列
    strsummary = mm.io.render_summary(                              # 将eval指标进行字符串格式化，用于在console上面的显示
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)                                               # 显示在命令行中
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))




if __name__ == "__main__":
    file_pre = "/home/shuai.li/code/FairMOT/MOT15/images/results/temp/"
    data_root = "/home/shuai.li/code/FairMOT/MOT15/images/train"
    seqs = ['KITTI-13',
            'KITTI-17',
            'ETH-Bahnhof',
            'ETH-Sunnyday',
            'PETS09-S2L1',
            'TUD-Campus',
            'TUD-Stadtmitte',
            'ADL-Rundle-6',
            'ADL-Rundle-8',
            'ETH-Pedcross2',
            'TUD-Stadtmitte',
           ]
    measure2files(file_pre, data_root, seqs)