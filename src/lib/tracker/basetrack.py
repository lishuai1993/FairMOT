import numpy as np
from collections import OrderedDict


class TrackState(object):           # 轨迹的四种状态
    New = 0                         # 创建新的轨迹
    Tracked = 1                     # 追踪状态
    Lost = 2                        # 丢失
    Removed = 3                     # 轨迹完成了，将其从当前帧中删除

# 存储一条轨迹信息的基本单元
class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()         # 有序的
    features = []                   # 存储该轨迹在不同帧对应位置通过ReID提取到的特征
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0           # 每次轨迹调用predict函数的时候就会+1

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):            # 返回当前帧的id
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed