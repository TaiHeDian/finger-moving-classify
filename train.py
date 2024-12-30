"""
Finger Movement Classification Training and Evaluation.

Description:
1. Data preprocessing and loading
2. Simple 1D CNN model definition
3. Model training and evaluation
4. Result visualization

Classes:
    SimpleCNN: A simple 1D convolutional neural network model

Functions:
    shuffle: Randomly shuffle input data and labels
    prepare_data: Prepare training and testing data
    train_model: Train model for one epoch
    test_model: Test model and output evaluation metrics
    
Author: Yingxin Gao
Last modified: 12/24/2024
Version: 1.0

Dependencies:
    - numpy
    - matplotlib
    - torch
    - sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

class SimpleCNN(nn.Module):
    """
    A simple 1D convolutional neural network model for finger movement classification.
    """
    def __init__(self, input_size, num_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * input_size // 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def shuffle(inputs, labels):
    """
    Shuffle the inputs and labels arrays using the same random permutation.

    Args:
        inputs (numpy.ndarray): Input array to be shuffled along first axis
        labels (numpy.ndarray): Labels array to be shuffled

    Returns:
        tuple: A tuple containing (shuffled_inputs, shuffled_labels) with the same
               dimensions as the input arrays but randomly reordered

    Note:
        Uses fixed random seed of 3 for reproducibility
    """
    np.random.seed(3)
    total_length = len(labels)
    permutation = np.random.permutation(total_length)
    labels = labels[permutation]
    inputs = np.take(inputs, permutation, axis=0)
    return inputs, labels

def prepare_data(num_samples, input_size, num_channels, num_classes, test_rate=0.25):
    """
    Prepare the data for training and testing.
    Args:
        num_samples (int): Number of samples per gesture class
        input_size (int): Size of each input sample
        num_channels (int): Number of channels in the input data
        num_classes (int): Number of gesture classes
        test_rate (float, optional): Fraction of data to use for testing. Defaults to 0.25
    Returns:
        tuple: Contains:
            - inputs (torch.Tensor): Training input data of shape (num_samples*5, num_channels, input_size)
            - labels (torch.Tensor): Training labels
            - inputs_test (torch.Tensor): Test input data
            - labels_test (torch.Tensor): Test labels
    Notes:
        Assumes data.npy exists in the current directory
    """
    inputs = np.zeros((num_samples * 5, num_channels, input_size))
    data = np.load('data.npy')

    labels = []

    for gesture_index in range(5):
        for sample_index in range(num_samples):
            sample_idx = gesture_index * num_samples + sample_index
            inputs[sample_idx, :, :] = data[gesture_index, :, sample_index, :]
            labels.append(gesture_index)

    train_index = int(len(labels) * (1 - test_rate))
    labels = np.array(labels)
    inputs, labels = shuffle(inputs, labels)

    # 转换为PyTorch张量
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    inputs_test = inputs[train_index:]
    labels_test = labels[train_index:]

    return inputs, labels, inputs_test, labels_test

def train_model(model, criterion, optimizer, inputs, labels):
    """Train the model for one epoch."""
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    return loss.item()

def test_model(model, inputs, labels, num_classes):
    """Test the model and print evaluation metrics."""
    model.eval()

    # 前向传播
    outputs = model(inputs)

    # 预测结果
    _, predicted = torch.max(outputs.data, 1)

    # 计算混淆矩阵
    label = labels.numpy()
    predict = predicted.numpy()
    cm = confusion_matrix(label, predict, labels=np.arange(num_classes))

    accuracy = accuracy_score(label, predict)
    print(f"准确率: {accuracy*100:.1f}%")
    precision = precision_score(label, predict, average='macro')
    print(f"精确率: {precision*100:.1f}%")
    recall = recall_score(label, predict, average='macro')
    print(f"召回率: {recall*100:.1f}%")
    f1 = f1_score(label, predict, average='macro')
    print(f"F1值: {f1*100:.1f}%")

    for i in range(5):
        sum_val = np.sum(cm[i, :])
        for j in range(5):
            cm[i, j] = cm[i, j] / sum_val * 100

    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.colorbar()
    labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 添加数值标签
    thresh = cm.max() / 2  # 阈值，用于决定文本颜色
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.annotate(
                f'{cm[i, j]:.1f}%',
                (j, i),
                ha='center',
                va='center',
                color='white' if cm[i, j] > thresh else 'black'
            )
    plt.tight_layout()
    plt.show()

    return cm

# 参数设置
input_size = 36  # 输入数据的长度，窗口宽度的两倍
num_channels = 5  # 一维通道数
num_classes = 5   # 分类的类别数
num_samples = int(3200 / input_size * 2)  # 样本数量
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# 创建模型实例
model = SimpleCNN(input_size, num_channels, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和测试
losses = []
inputs, labels, inputs_test, labels_test = prepare_data(
    num_samples, input_size, num_channels, num_classes
)
for epoch in range(num_epochs):
    num_batches = len(inputs) // batch_size
    epoch_loss = 0

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = (batch + 1) * batch_size

        batch_inputs = inputs[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        loss = train_model(model, criterion, optimizer, batch_inputs, batch_labels)
        losses.append(loss)
        epoch_loss += loss

    # 打印每个epoch的loss值
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / num_batches:.4f}')

# 测试模型
cm = test_model(model, inputs_test, labels_test, num_classes)

# 绘制loss变化曲线
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Variation')
plt.tight_layout()
plt.show()
