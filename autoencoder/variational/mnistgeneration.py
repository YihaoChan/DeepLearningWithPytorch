"""
变分自编码器生成图片
"""
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.fc1 = nn.Linear(784, 256)

        self.fc2_mean = nn.Linear(256, 20)
        self.fc2_std = nn.Linear(256, 20)

        self.fc3 = nn.Linear(20, 256)
        self.fc4 = nn.Linear(256, 784)

        self.relu = nn.ReLU()

    def encoder(self, x):
        x_batch_size = x.size(0)

        x = x.view(x_batch_size, 784)  # [batch_size, 1, 28, 28] -> [batch_size, 784]

        x = self.fc1(x)  # [batch_size, 784] -> [batch_size, 256]

        x = self.relu(x)

        h_mean = self.fc2_mean(x)  # 均值：-> [batch_size, 20]
        h_std = self.fc2_std(x)  # 标准差：-> [batch_size, 20]

        return h_mean, h_std

    def reparametrization(self, mu, sigma):
        # 和采样自高斯分布进行圈乘，每个像素点的元素分别相乘，所以计算loss时要除以像素和batch
        std = sigma * torch.randn_like(sigma)

        z = mu + std  # [batch_size, 20]

        return z

    def decoder(self, z):
        z = self.fc3(z)  # [batch_size, 20] -> [batch_size, 256]

        z = self.relu(z)

        z = self.fc4(z)  # [batch_size, 256] -> [batch_size, 784]

        return z

    def forward(self, x):
        mu, sigma = self.encoder(x)

        z = self.reparametrization(mu, sigma)

        out = self.decoder(z)

        return out, mu, sigma


def get_dataloader(train=True, batch_size=64):
    transform_fn = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root='./dataset/mnist_data',
        transform=transform_fn,
        train=train,
        download=True
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


def cal_kl_divergence(mu, sigma, batch):
    """
    :param batch: batch_size
    :param mu: 均值
    :param sigma: 标准差
    :Description: 当计算当前分布和高斯分布的KL散度时，计算公式为：
                  ½ * (σ^2 + μ^2 - log(σ^2) - 1)
    """
    return 0.5 * torch.sum(
        torch.pow(mu, 2) +
        torch.pow(sigma, 2) -
        torch.log(1e-8 + torch.pow(sigma, 2)) -  # 如果x接近于0，log为趋向负无穷，会导致KL很小(不加小常数会出现nan)
        1
    ) / (batch * 28 * 28)


def train(net, num_epochs, batch_size, optimizer, device):
    net.train()

    criterion = nn.MSELoss()

    train_dataloader = get_dataloader(train=True, batch_size=batch_size)

    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(train_dataloader):
            num_img = data.size(0)
            data = data.view(num_img, -1).to(device)

            output, mu, sigma = net(data)
            output = output.to(device)

            loss_reconstruct = criterion(output, data)
            kl_divergence = cal_kl_divergence(mu, sigma, batch_size)

            loss = loss_reconstruct + kl_divergence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('epoch: {}, batch: {}, loss: {}'.format(
                    epoch, batch_idx, loss.item()
                ))

        gen_img = output.cpu().data.view(-1, 1, 28, 28)  # 铺开成64张小图拼在一起，每一张1通道及28*28大小
        save_image(gen_img, './img/gen_img-{}.png'.format(epoch + 1))


if __name__ == '__main__':
    if not os.path.exists('./img'):
        os.mkdir('./img')

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        torch.cuda.manual_seed(7)
    else:
        dev = torch.device("cpu")
        torch.manual_seed(7)

    vae = VariationalAutoEncoder().to(dev)

    optimizer_vae = optim.Adam(vae.parameters())

    train(net=vae, num_epochs=20, batch_size=64, optimizer=optimizer_vae, device=dev)
