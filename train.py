import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import visdom
import numpy as np
from dataloader import FMCWDataset  # 替换为实际数据集模块路径
from model import HRKNet


# 可视化环境初始化
viz = visdom.Visdom(env='HR_Prediction')
loss_window = viz.line(X=[0], Y=[0], opts=dict(title='Loss'))
metric_window = viz.line(X=[0], Y=[0], opts=dict(title='MAE'))

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:
    mask_spectrum = [0]
    enc_in = 8
    seq_len = 12800
    pred_len = 1024
    seg_len = 512
    num_blocks = 5
    dynamic_dim = 128
    hidden_dim = 128
    hidden_layers = 4
    multistep = True


config = Config()

# 初始化组件
model = HRKNet(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.MSELoss()

# 数据加载
dataset = FMCWDataset(src_path='/path/to/data', tau=5)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          collate_fn=dataset.collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8,
                        collate_fn=dataset.collate_fn, num_workers=4)

# 训练参数
best_mae = float('inf')
epochs = 120

# 训练循环
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs_complex = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(inputs_complex)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

        # 每100批次打印进度
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch + 1}/{epochs} | Batch: {batch_idx}/{len(train_loader)}'
                  f' | Loss: {loss.item():.4f}')

    # 验证阶段
    model.eval()
    val_loss = 0
    mae = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs_complex = torch.view_as_complex(inputs.permute(0, 2, 3, 1).contiguous())
            inputs_complex = inputs_complex.to(device)
            labels = labels.to(device)

            outputs = model(inputs_complex)
            val_loss += criterion(outputs, labels).item()
            mae += torch.mean(torch.abs(outputs - labels)).item()

    # 计算平均指标
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    mae /= len(val_loader)

    # 更新可视化
    viz.line(X=[epoch], Y=[train_loss], win=loss_window, name='train', update='append')
    viz.line(X=[epoch], Y=[val_loss], win=loss_window, name='val', update='append')
    viz.line(X=[epoch], Y=[mae], win=metric_window, update='append')

    # 保存最佳模型
    if mae < best_mae:
        best_mae = mae
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'mae': mae
        }, 'best_model.pth')

    print(f'Epoch {epoch + 1} Summary:')
    print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.2f} bpm')

# 加载最佳模型
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f'Best MAE: {checkpoint["mae"]:.2f} bpm at epoch {checkpoint["epoch"]}')