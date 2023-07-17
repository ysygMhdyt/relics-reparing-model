import argparse
import os
from solver import Solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 训练参数
    parser.add_argument('--use_mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--batch_size', type=int, default=4, help='batch大小')
    parser.add_argument('--num_iters', type=int, default=100, help='总训练次数')
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0002)  # 学习率
    parser.add_argument('--beta1', type=float, default=0.5)  # Adam优化器衰减率
    parser.add_argument('--model_save_step', type=int, default=99)  # 每多少次循环保存模型，默认500
    parser.add_argument('--data_dir', type=str, default='data/processed')  # 训练数据路径
    parser.add_argument('--test_dir', type=str, default='data/test')  # 测试数据路径
    parser.add_argument('--result_dir', type=str, default='repair/result/')  # 结果存储路径
    parser.add_argument('--model_save_dir', type=str, default='repair/model/')  # 模型存储路径

    config = parser.parse_args()
    # print(config)

    # 若文件夹不存在则创建文件夹
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)
    if not os.path.exists(config.test_dir):
        os.makedirs(config.test_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    solver = Solver(config)
    if config.use_mode == 'train':
        solver.train()
    if config.use_mode == 'test':
        solver.test()