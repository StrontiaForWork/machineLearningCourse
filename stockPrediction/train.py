from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parserChange import args
from dataset import getData
import matplotlib.pyplot as plt

# 声明损失
iter = []  # 迭代次数
lossTotal = []  # 损失

def plot(iter, lossTotal):
    # 画图
    plt.plot(iter, lossTotal)
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.show()

    # plt.plot(result_real, label='real')
    # plt.plot(result_predict, label='pred')
    # plt.legend()
    # plt.show()

def train():

    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001

    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size )
    # 训练轮数
    # args.epochs = 200
    print(args.epochs)

    for i in range(args.epochs):
        total_loss = 0
        for idx, (data, label) in enumerate(train_loader):
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(Variable(data1).cuda())
                # print(pred.shape)
                pred = pred[1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(Variable(data1))
                pred = pred[1, :, :]
                label = label.unsqueeze(1)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 损失
        iter.append(i)
        lossTotal.append(total_loss)

        print(total_loss)
        if i % 10 == 0:
            # torch.save(model, args.save_file)
            torch.save({'state_dict': model.state_dict()}, args.save_file)
            print('第%d epoch，保存模型' % i)
    # torch.save(model, args.save_file)
    torch.save({'state_dict': model.state_dict()}, args.save_file)
    #画图
    plot(iter, lossTotal)


train()