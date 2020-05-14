import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import cv2
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import _init_paths
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
# import motmetrics as mm
from tracking_utils.evaluation import Evaluator

import sys
sys.path.append(r'D:\work\code\package\code\code_ls\package')
sys.path.append('/home/shuai.li/code/package')
import pathcostom as pm





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





def getbb_perid():
    file_result = '/home/shuai.li/dset/MOT/cue_video/compare_index_yhw/result/results.txt'           # 追踪的结果文件，result.txt
    dir_img = '/home/shuai.li/dset/MOT/cue_video/compare_index_yhw/result/frame'                    # 视频帧的文件夹\
    dir_dst = '/home/shuai.li/dset/MOT/cue_video/compare_index_yhw/result/patch_ids'               # 结果截图的根目录，每个子文件夹以id命名，长度为5，前面补零

    formater = "{:05d}"

    # 读取结果文件
    result_data = np.loadtxt(file_result, delimiter=',', dtype=np.float, usecols=[0, 1, 2, 3, 4, 5])
    result_data = result_data.astype(np.int)

    # 创建全部文件夹
    ids = result_data[:, 1].tolist()
    ids_set = set(ids)
    for temp in tqdm(ids_set):
        pm.mkdir_recu(osp.join(dir_dst, formater.format(int(temp))))

    # 保存id的bb
    for temp in tqdm(result_data):

        imgpath_src = osp.join(dir_img, formater.format(temp[0] - 1) + '.jpg')             # 视频帧转换
        imgpath_dst = osp.join(dir_dst, formater.format(temp[1]), formater.format(temp[0] - 1) + '.jpg')     # id作为文件夹命名 + 视频帧作为文件名
        top, bot, left, right = temp[3], temp[3]+temp[5], temp[2], temp[2]+temp[4]

        img = cv2.imread(imgpath_src)
        cv2.imwrite(imgpath_dst, img[top:bot, left:right])


def shutil_idfolder_byTspan(Fspan = 18, shutil_func=shutil.copy):
    """

    @param Fspan:       该id的trajectory的时间跨度需要大于等于 1s，当前程序 1s 对应的是18帧左右
    @return:
    """

    dir_src = r'D:\work\dset\cue_mot_Data\compare_index_yhw\patch_ids\\'
    dir_dst = r'D:\work\dset\cue_mot_Data\compare_index_yhw\ge{}frame_perid_inpatch\\'.format(Fspan)

    idfolders = os.listdir(dir_src)
    if not osp.exists(dir_dst):
        os.makedirs(dir_dst)

    for folder in tqdm(idfolders):
        framelist = list(map(lambda x: int(x.split('.jpg')[0]), os.listdir(osp.join(dir_src, folder))))
        diff = max(framelist) - min(framelist)
        if diff >= Fspan:
            if shutil_func == shutil.copytree:
                shutil_func(osp.join(dir_src, folder), osp.join(dir_dst, folder))



def hist_fspan(bins=5, imgnumbins = 5, range=None):

    # dir_src_list = [r'D:\work\dset\cue_mot_Data\compare_index_yhw\eq1frame_perid_movefpatch_ids\\',
    #                 r'D:\work\dset\cue_mot_Data\compare_index_yhw\lt18frame_perid_inpatch_movefpatch_ids\\',
    #                 r'D:\work\dset\cue_mot_Data\compare_index_yhw\patch_ids\\',
    #                 ]
    dir_src_list = [r'D:\work\dset\cue_mot_Data\compare_index_yhw\gap8f_ge60frame_perid_inpatch\\']
    rotation = -90

    id_folderlist = pm.get_dirflistdir(dir_src_list)
    print('id_folderlist:\t', len(id_folderlist))

    numlist = []
    img_numlist = []
    for folder in tqdm(id_folderlist):
        num_perid = list(map(lambda x: int(x.split('.jpg')[0]), os.listdir(folder)))    # 注意是帧跨度，而不是图片量
        imgnum_perid = len(os.listdir(folder))

        try:
            fspan = max(num_perid) - min(num_perid)
        except:
            pass
        numlist.append(fspan)
        img_numlist.append(imgnum_perid)

    a, _ = np.histogram(numlist, bins=bins, density=True)

    bins = np.array(bins)
    factor =bins[1:] - bins[:-1]
    f = a*factor

    bar_xlabel = []
    for i, j in zip(bins[:-1], bins[1:]):
        bar_xlabel.append(str(i) + '-' + str(j))

    fig, ax = plt.subplots(2, 2)
    fig.suptitle("gap8f_ge60frame_perid_inpatch")

    xticks = (np.array(bins, dtype=np.int) - 1)//15            # 换算成秒
    ax[0][0].hist(numlist, bins=bins)                # bins是左闭右开区间
    ax[0][0].set_xticks(ticks=bins)
    ax[0][0].set_xticklabels(labels=xticks, rotation=rotation)
    ax[0][0].set_xlabel('value of time span in per trajectory')
    ax[0][0].set_ylim((0, 400))
    ax[0][0].set_ylabel('num in per bin')

    ax[0][1].bar(np.arange(1, len(xticks), 1), f)                                                   # bins是左闭右开区间
    ax[0][1].set_xticks(ticks=np.arange(1, len(xticks), 1))
    ax[0][1].set_xticklabels(labels=xticks[1:], rotation=rotation)
    ax[0][1].set_xlabel('the percentage of time span in per trajectory')
    ax[0][1].set_ylim((0, 1))
    ax[0][1].set_ylabel('num in per bin')
    ax[0][1].set_title('time span(s) of per trajectory')


    b, _ = np.histogram(img_numlist, bins=imgnumbins, density=True)
    imgnumbins = np.array(imgnumbins)
    factor = imgnumbins[1:] - imgnumbins[:-1]
    bf = b * factor

    xticks = imgnumbins
    ax[1][0].hist(img_numlist, bins=imgnumbins)                      # bins是左闭右开区间
    ax[1][0].set_xticks(ticks=xticks)
    ax[1][0].set_xticklabels(labels=xticks, rotation=rotation)
    ax[1][0].set_xlabel('num of img per trajectory')
    ax[1][0].set_ylim((0, 400))
    ax[1][0].set_ylabel('num in per bin')

    bar_xlabel = []
    for i, j in zip(imgnumbins[:-1], imgnumbins[1:]):
        bar_xlabel.append(str(i) + '-' + str(j))

    ax[1][1].bar(np.arange(1, len(xticks), 1), bf)                                                   # bins是左闭右开区间
    ax[1][1].set_xticks(ticks=np.arange(1, len(xticks), 1))
    ax[1][1].set_xticklabels(labels=xticks[1:], rotation=rotation)
    ax[1][1].set_xlabel('The percentage of the number of images per trajectory')
    ax[1][1].set_ylim((0, 1))
    ax[1][1].set_ylabel('num in per bin')
    # plt.title('time span(s) of per trajectory')

    plt.show()

def get_frame_gap_srcvideo(gap = 3, shutil_func=shutil.copy):

    dir_src = r'D:\work\dset\cue_mot_Data\compare_index_yhw\get60frame_perid_inpatch'
    dir_dst = r'D:\work\dset\cue_mot_Data\compare_index_yhw\gap{}f_ge60frame_perid_inpatch'.format(gap)
    frame_orderset = set(range(0, 54000, gap))

    folder_list = os.listdir(dir_src)
    for folder in tqdm(folder_list):
        frame_set_perid = set(map(lambda x: int(x.split('.jpg')[0]), os.listdir(osp.join(dir_src, folder))))
        intersection = frame_set_perid.intersection(frame_orderset)

        dst_path = osp.join(dir_dst, folder)
        src_path = osp.join(dir_src, folder)
        pm.mkdir_recu(dst_path)

        for order in intersection:
            shutil_func(osp.join(src_path, '{:05d}.jpg'.format(order)), dst_path)

def get_errnum():

    dir_src = r'D:\work\dset\cue_mot_Data\compare_index_yhw\gap8f_ge60frame_perid_inpatch\\'

    folder_list = os.listdir(dir_src)

    num = 0
    for folder in tqdm(folder_list):
        if '_' in folder:
            num += 1
    print(num)







if __name__ == "__main__":
    # file_pre = "/home/shuai.li/code/FairMOT/MOT15/images/results/temp/"
    # data_root = "/home/shuai.li/code/FairMOT/MOT15/images/train"
    # seqs = ['KITTI-13',
    #         'KITTI-17',
    #         'ETH-Bahnhof',
    #         'ETH-Sunnyday',
    #         'PETS09-S2L1',
    #         'TUD-Campus',
    #         'TUD-Stadtmitte',
    #         'ADL-Rundle-6',
    #         'ADL-Rundle-8',
    #         'ETH-Pedcross2',
    #         'TUD-Stadtmitte',
    #        ]
    # measure2files(file_pre, data_root, seqs)

    # getbb_perid()
    Fspan = 60
    # shutil_func = shutil.copytree
    # shutil_idfolder_byTspan(Fspan= Fspan, shutil_func=shutil_func)
    # get_frame_gap_srcvideo(gap=8, shutil_func=shutil.copy)

    bins = list(range(1, 15*8+2, 15))                   # [1, 16)
    tail = list(range(121, 60*10+2, 60))
    tail.pop(0)
    bins.extend(tail)

    imgnumbins = list(range(0, 150+1, 10))
    range=None
    hist_fspan(bins=bins, imgnumbins=imgnumbins, range=range)
    # get_errnum()

