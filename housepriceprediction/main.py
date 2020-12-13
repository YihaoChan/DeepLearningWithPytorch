"""
训练集有data和target(SalePrice)，测试集只有data。完成预测任务，即对训练集进行训练，并对测试集进行预测。
根据前面的特征，预测最后的SalePrice
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")


class HousePriceNet(nn.Module):
    """
    feature_shape -> 全连接 -> Relu -> 1 (预测的房价)
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HousePriceNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self._init_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def _init_parameters(self):
        """
        权重初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
            else:
                nn.init.constant_(p, 0)


def preprocess_data(train_dataset, test_dataset):
    # 数据的第一个特征为id，不起作用，所以去掉这一项特征，进而将有用的训练、测试集特征进行组合
    # 训练集的标签不加入，所以iloc的终点有限制
    all_features = pd.concat((train_dataset.iloc[:, 1:-1], test_dataset.iloc[:, 1:]))

    # 获得数值类型的字段
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # 对于数值类型的数据，进行标准化处理。 **apply结合lambda**
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )

    # 对于缺失的特征值，将其替换成该特征的均值，即0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)

    """
    对于值为字符串类型的数据，转化为哑变量
    假设特征MSZoning里面有两个不同的离散值RL和RM，则这一步转换去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。
    如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。
    """
    all_features = pd.get_dummies(all_features, dummy_na=True)

    return all_features


def get_dataloader(dataset, batch_size=64):
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


def log_rmse(output, label):
    """
    对数均方根损失：把传入MSELoss的output和label都取对数，最后开方即可
    """
    criterion = nn.MSELoss()

    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(output, torch.tensor(1.0))
        pred = torch.log(clipped_preds.float())
        label = torch.log(label.float())

        loss = criterion(pred, label)

        rmse = torch.sqrt(loss)

    return rmse.item()


def train(num_epochs, net, train_features, train_labels):
    """
    :param num_epochs: 迭代次数
    :param net: 实例化后的网络
    :param train_features: 训练集的特征
    :param train_labels: 训练集的标签
    """
    train_dataloader = get_dataloader(
        torch.utils.data.TensorDataset(train_features, train_labels),  # 特征和标签打包送入loader
        batch_size=64
    )

    net.train()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters())

    for epoch in range(num_epochs):
        epoch += 1

        for batch_idx, (train_loader_data, train_loader_label) in enumerate(train_dataloader):
            # 模型训练产生的预测
            train_loader_output = net(train_loader_data.float())

            loss = criterion(train_loader_output.float(), train_loader_label.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch: {}, batch: {}, loss: {}'.format(
                epoch, batch_idx, log_rmse(train_loader_output, train_loader_label)
            ))

            # 保存模型
            if not os.path.exists('./model/'):
                os.mkdir('./model')

            torch.save(net.state_dict(), './model/model.pkl')


def predict(net, test_features):
    """
    测试/预测
    :param net: 实例化后的模型
    :param test_features: 测试集的特征
    """
    test_dataloader = get_dataloader(
        torch.utils.data.TensorDataset(test_features),
        batch_size=64
    )

    net.eval()

    # 读取模型参数
    if os.path.exists('./model/model.pkl'):
        net.load_state_dict(torch.load('./model/model.pkl'))
    else:
        print('No model for evaluation.')
        return

    test_output_list = []

    for batch_idx, test_loader_data in enumerate(test_dataloader):
        # 预测结果
        test_loader_output = net(test_loader_data[0].float())

        test_output_item = torch.squeeze(test_loader_output, 1) # 只要第0维作为预测数据即可
        test_output_item = test_output_item.detach().numpy().tolist()

        for i in range(len(test_output_item)):
            test_output_list.append(test_output_item[i])

        print('batch: {}, \noutput: {}'.format(
            batch_idx, test_loader_output
        ))

    df = pd.DataFrame(test_output_list, columns=['SalePrice'])

    df.to_csv('./submission.csv')


if __name__ == '__main__':
    # 读取数据
    train_raw_dataset = pd.read_csv('./dataset/train.csv')
    test_raw_dataset = pd.read_csv('./dataset/test.csv')

    # 特征处理
    all_features = preprocess_data(train_raw_dataset, test_raw_dataset)

    # 训练集数据个数，用于划分提取特征后的训练、测试数据
    train_dataset_length = train_raw_dataset.shape[0]

    # 特征预处理
    train_processed_feature = torch.tensor(all_features.iloc[:train_dataset_length, :].values)
    test_processed_feature = torch.tensor(all_features.iloc[train_dataset_length:, :].values)

    # 训练集标签
    train_label = torch.tensor(train_raw_dataset.iloc[:, -1].values).view(-1, 1)

    # 特征数量，即输入层维数
    data_shape = train_processed_feature.shape[1]

    # 实例化模型
    model = HousePriceNet(data_shape, 32, 1)

    # 训练
    train(1000, model, train_processed_feature, train_label)

    # 测试
    predict(model, test_processed_feature)

"""
总结：
一、训练：
1. 在训练阶段时，训练数据有train_data和label，而模型将train_data从331维，降低到32维，最后输出为1维。即：该过程中，模型根据
   train_data生成的输出维度为1，需要与label维度相同；
2. 将(train_data, label)打包送入dataloader，模型训练时产生的输出output = model(train_data)，维度为1；
3. 将模型产生的输出output和label作loss，在降低loss值的过程中，output在不断地接近label，这就是一个学习过程；
4. 直到训练完毕的时候，就是模型学习完毕的时候。此时模型已经具备能够根据train_data生成接近label的数据的能力。
二、测试：
1. 在测试阶段时，测试数据只有test_data没有label，而采用已经训练好的模型对test_data进行拟合，生成的数据维度就是1；
2. 意思是让一个已经具备根据数据生成接近真实标签的能力的模型，对新数据进行拟合，尝试"复刻"这种能力，将能力迁移到未知的数据/环境中；
2. 这个过程可以看做模型在测试阶段时，在"模仿"训练过程，从而生成测试阶段的output，而这个output就是预测数据，从而达到了预测新数据的目的。 
"""
