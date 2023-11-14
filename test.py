import torch

# 假设的参数
batch_size = 4
channels = 3
height = 224
width = 224

# 创建一个形状为 (batch_size, 1, 1, 1) 的张量
beta_t = torch.ones(batch_size, 1, 1, 1)*0.5

# 创建一个形状为 (batch_size, channels, height, width) 的张量
data = torch.randn(batch_size, channels, height, width)

# 执行元素级别的乘法
result = beta_t * data

# 打印结果以验证形状
print("beta_t shape:", beta_t.shape)
print("data shape:", data.shape)
print("result shape:", result.shape)
