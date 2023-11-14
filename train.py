import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CTImagesDataset
from model import DCDDPM
from torchvision import transforms
from DCUnet import *
import time

# 超参数设置
learning_rate = 1e-4
batch_size = 16
num_epochs = 100

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 或其他您需要的大小
    transforms.ToTensor(),
])
high_dir = '/Users/zane/Desktop/B301MM/high'
low_dir = '/Users/zane/Desktop/B301MM/low'

train_dataset = CTImagesDataset(high_dir, low_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 模型初始化
unet_model = DualChannelUnet(in_channels=1, out_channels=1)  # Replace with your Unet model
model = DCDDPM(unet_model)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    start_time = time.time()
    for high_img, low_img in train_loader:
        # 假设模型的输入是高低图像的组合
        # inputs = torch.cat((high_img, low_img), dim=1)
        loss = model(low_img)

        # loss = criterion(outputs, high_img)  # 假设目标是重建高图像
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Used: {time.time() - start_time:.4f}s')

# 保存模型
torch.save(model.state_dict(), 'ddpm_model.pth')
