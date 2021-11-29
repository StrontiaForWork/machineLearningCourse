from LSTMModel import lstm
from dataset import getData
from parserChange import args
import torch
import matplotlib.pyplot as plt
import numpy as np


def eval(result_predict, result_real):
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    close_max, close_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred = model(x)
        list = pred.data.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())

    for i in range(len(preds)):
        print('预测值是%.2f,真实值是%.2f' % (
        preds[i][0] * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))

        result_predict.append(preds[i][0] * (close_max - close_min) + close_min)
        result_real.append(labels[i] * (close_max - close_min) + close_min)

# 画图
def plot():
    # 声明预测列表
    result_predict = []
    result_real = []
    eval(result_predict, result_real)

    result_predict = np.array(result_predict)
    result_real = np.array(result_real)

    # 计算模型评估结果
    acc = np.average(np.abs(result_predict - result_real[:len(result_predict)]) / result_real[:len(result_predict)])  # 偏差程度
    print('真实数据和预测数据的偏差值为: ', acc)

    # 画图
    plt.plot(result_real, label='real')
    plt.plot(result_predict, label='pred')
    plt.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('Time/天数')
    plt.ylabel('收盘价')
    plt.show()



#
plot()


