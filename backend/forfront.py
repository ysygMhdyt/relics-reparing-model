import argparse
from solver import Solver
import os

def repair_img(img_path):
    print(img_path)
    parser = argparse.ArgumentParser()
    # 用于初始化
    parser.add_argument('--use_mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--batch_size', type=int, default=4, help='batch大小')
    parser.add_argument('--num_iters', type=int, default=10, help='总训练次数')
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0002)  # 学习率
    parser.add_argument('--beta1', type=float, default=0.5)  # Adam优化器衰减率
    parser.add_argument('--model_save_step', type=int, default=500)  # 每多少次循环保存模型，默认500
    parser.add_argument('--data_dir', type=str, default='data/processed')  # 训练数据路径
    parser.add_argument('--test_dir', type=str, default='data/test')  # 测试数据路径
    parser.add_argument('--result_dir', type=str, default='repair/result')  # 结果存储路径
    parser.add_argument('--model_save_dir', type=str, default='repair/model')  # 模型存储路径

    config = parser.parse_args()
    solver = Solver(config)

    newpath = solver.repair(img_path)

    return newpath
