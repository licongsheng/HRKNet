import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import re


class FMCWDataset(Dataset):
    def __init__(self, src_path, tau=5, nframes=25, nchips=2, nsamples=256, stride=None, transform=None):
        """
        FMCW雷达数据集类（适配[1,60]心率格式）
        参数：
            src_path   : 数据根目录
            tau        : 时间窗长度（秒）

            stride     : 窗口滑动步长（秒），None表示非重叠
            transform  : 数据增强变换
        """
        super().__init__()
        # 参数校验
        assert tau == int(tau), "时间窗tau需为整数值（秒）"
        self.tau = int(tau)
        self.window_size = tau*nframes*nchips
        self.stride = stride or tau
        self.transform = transform
        self.nsamples=nsamples
        # 元数据存储结构
        self.meta_data = []

        # 遍历目录解析mat文件
        for filename in os.listdir(src_path):
            if not filename.endswith('.mat'):
                continue

            # 解析文件名元数据
            match = re.match(r"subject_(\d+)_P(\d+)_O(\d+).mat", filename)
            if not match:
                continue

            subj_id, p_pos, o_ori = map(int, match.groups())
            file_path = os.path.join(src_path, filename)

            # 计算可用时间窗口
            with h5py.File(file_path, 'r') as f:
                total_samples = f['data'].shape[0]  # [nchirps, nADCs, nReceivers]

            # 生成窗口起始点（秒为单位）
            max_start = total_samples - self.window_size
            window_starts = np.arange(0, max_start + 1, self.stride)

            # 记录元数据（按秒存储）
            for start_sec in window_starts:
                self.meta_data.append({
                    "file_path": file_path,
                    "start_sec": int(start_sec),
                    "subj_id": subj_id,
                    "position": p_pos,
                    "orientation": o_ori
                })

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        meta = self.meta_data[idx]

        # 读取雷达数据切片 ------------------------------------------------
        with h5py.File(meta["file_path"], 'r') as f:
            # 计算样本起始位置
            start_sample = int(meta["start_sec"] * self.fs)

            # 读取复数数据 [4, window_size]
            data = f['data'][start_sample:start_sample + self.window_size, :, :]
            data = data.view(np.complex64)
            shape = data.shape()
            data = data.reshape(shape[0]*shape[1], shape[2])

            # 读取心率数据
            hr_matrix = np.array(f['hr']).T.squeeze()  # 转换为(60,)

        # 处理心率标签 ----------------------------------------------------
        hr_start = meta["start_sec"]
        hr_end = hr_start + self.tau

        # 安全截断处理
        hr_window = hr_matrix[hr_start:hr_end]
        if len(hr_window) < self.tau:  # 处理边缘情况
            hr_window = np.pad(hr_window, (0, self.tau - len(hr_window)),
                               mode='edge')

        hr_mean = np.mean(hr_window)

        # 数据转换 -------------------------------------------------------
        # 拆分实虚部并转换为张量 [4, N, 2]
        data_real = torch.FloatTensor(data.real)
        data_imag = torch.FloatTensor(data.imag)
        data_tensor = torch.stack([data_real, data_imag], dim=-1)
        data_tensor = data_tensor.permute(1, 0)
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, torch.FloatTensor([hr_mean])

# 使用验证 --------------------------------------------------------------
