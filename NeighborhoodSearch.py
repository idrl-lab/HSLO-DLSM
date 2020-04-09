import numpy as np
import torch
import scipy.io as io

save_path = 'layout_design/Output'


class NeighborSearch(object):
    def __init__(self, dim, lower_bound, upper_bound, model, device):
        self.dim = dim  # 设计变量维度
        self.x_bound_lower = lower_bound
        self.x_bound_upper = upper_bound    # 设计变量上界，注意取不到该数值
        self.net = model
        self.device = device
        # self.x = np.zeros((1, self.dim))
        self.x = np.array([1,3,5,7,9,31,33,35,37,39,61,63,65,67,69,91,93,95,97,99])
        fitness = self.calculate_fitness(self.x)
        self.pg = self.x
        self.pg_fitness = fitness[0, 0]

    def calculate_fitness(self, x):
        device1 = self.device
        if x.ndim == 1:
            x = x[np.newaxis, :]
        fitness = np.zeros([x.shape[0], 2])
        for j in range(x.shape[0]):
            layout_map = np.zeros((200, 200))
            location = x[j][:].astype(int)

            location -= 1
            for i in location:
                layout_map[(i % 10) * 20:(i % 10) * 20 + 20, i // 10 * 20:i // 10 * 20 + 20] = np.ones((20, 20))

            layout_tensor = torch.from_numpy(layout_map).float().to(device1)
            layout_tensor = layout_tensor.unsqueeze(0).unsqueeze(0)
            preds_heat = self.net(layout_tensor)
            pred_heat_numpy = (preds_heat.cpu().detach().numpy()[0, 0, :, :]) * 100 + 290

            # 根据预测得到的温度分布，计算温度场性能指标：最高温度和温度方差
            t_0 = 298   # unit: K       

            # 归一化处理
            phi_0 = 10000       # the intensity of heat source: phi0 = 10000W/m^2
            l_side = 0.1        # L = 0.1m
            k = 1               # the thermal conductivity k = 1W/(m.K)


            t_max = np.max(pred_heat_numpy)

            t_m_norm = (t_max - t_0)/(phi_0*(l_side**2)/k)
            sigma_norm = np.sqrt(np.var(pred_heat_numpy))/(phi_0*(l_side**2)/k)

            temp = np.array([t_m_norm, sigma_norm])


            fitness[j, :] = temp
        return fitness

    def neighborhood(self, x, location):
        neighbor = np.zeros([80, self.dim])
        k = 0

        for j in np.arange(self.x_bound_lower, self.x_bound_upper):
            if np.isin(j, x):
                continue
            neighbor_x = x.copy()
            neighbor_x[location] = j
            neighbor_x = np.sort(neighbor_x)

            neighbor[k][:] = neighbor_x
            k = k + 1

        return neighbor

    def evolve(self):
        iteration_best_fitness = self.pg_fitness
        flag = 1   # 用来指示目标函数是否有改进
        step = 0
        indicator = 0
        while flag == 1:
            flag = 0
            indicator += 1
            print('Indicator = ', str(indicator))
            for i in range(self.dim):
                neighbor = self.neighborhood(self.pg, i)
                fitness_neighbor = self.calculate_fitness(neighbor)
                temp = np.min(fitness_neighbor[:, 0])
                if temp < self.pg_fitness:
                    flag = 1
                    self.pg = neighbor[np.argmin(fitness_neighbor[:, 0])]
                    self.pg_fitness = np.min(fitness_neighbor[:, 0])

                iteration_best_fitness = np.append(iteration_best_fitness, self.pg_fitness)
                step += 1
                print('Position: %d, Iter: %d, Best fitness: %.5f' % (i, step, self.pg_fitness))
        return iteration_best_fitness


def prepare_fpn_model(given_image):
    '''
    准备代理模型
    :param given_image: layout: x2y
    :return: model and device
    '''
    import sys
    sys.path.append("..")

    import os
    from fpn.model import fpn

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = fpn().to(device)
    if given_image == 'layout':
        model_path = modelpath

    print("model path:", model_path)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, device


def evaluation_layout(list, save_name):
    given_image = 'layout'
    model, device = prepare_fpn_model(given_image)

    layout_map = np.zeros((200, 200))
    location = list.astype(int)
    location -= 1
    for i in location:
        layout_map[(i % 10) * 20:(i % 10) * 20 + 20, i // 10 * 20:i // 10 * 20 + 20] = np.ones((20, 20))

    layout_tensor = torch.from_numpy(layout_map).float().to(device)
    layout_tensor = layout_tensor.unsqueeze(0).unsqueeze(0)
    preds_heat = model(layout_tensor)
    pred_heat_numpy = (preds_heat.cpu().detach().numpy()[0, 0, :, :]) * 100 + 290

    # 根据预测得到的温度分布，计算温度场性能指标：最高温度和温度方差
    t_0 = 298  # unit: K
    update_id = np.greater(t_0, pred_heat_numpy)
    pred_heat_numpy[update_id] = t_0  # 把预测出来小于T_0的温度值抹平到T_0

    # 归一化处理
    phi_0 = 10000  # the intensity of heat source: phi0 = 10000W/m^2
    l_side = 0.1  # L = 0.1m
    k = 1  # the thermal conductivity k = 1W/(m.K)

    t_max = np.max(pred_heat_numpy)
    t_m_norm = (t_max - t_0) / (phi_0 * (l_side ** 2) / k)
    sigma_norm = np.sqrt(np.var(pred_heat_numpy)) / (phi_0 * (l_side ** 2) / k)
    temp = np.array([t_m_norm, sigma_norm])
    print(temp)
    print('Code is executed.')

    # 保存结果
    io.savemat(save_path + save_name,
               {'list': list, 'pred_heat': pred_heat_numpy, 'norm_indicator': temp})


if __name__ == "__main__":
    import time

    list1 = ['2k', '4k', '6k', '8k', '1w', '2w', '3w', '4w', '5w', 'ex']

    for i in list1:
        print(i)

        modelpath = 'modelfile/fpn_x2y_' + i + '.pth'
        save_name = '/Evaluation_NS_' + i + '.mat'

        start = time.process_time()
        # 神经网络模型初始化
        given_image = 'layout'
        model, device = prepare_fpn_model(given_image)
        # 神经网络模型准备完毕
        print('---------------------------------------------')
        end1 = time.process_time()
        print('FPN model loading time: %s Seconds' % (end1 - start))

        # 领域搜索算法参数初始化
        dimension = 20
        x_min = 1
        x_max = 101

        # 采用领域搜索算法进行优化
        print('---------------------------------------------')
        print('NeighborhoodSearch is running ... ')
        ns = NeighborSearch(dimension, x_min, x_max, model, device)
        print('Initialization succeed.')
        iter_best_fitness = ns.evolve()
        print('NeighborhoodSearch stops.')
        print('---------------------------------------------')
        # 优化结束
        end2 = time.process_time()
        print('NS running time: %s Seconds' % (end2 - end1))

        # 保存优化结果
        io.savemat(save_path + '/Optimized_NS_' + i + '.mat', {'xbest': ns.pg, 'fitness_best': ns.pg_fitness,
                                                        'iter_best_fitness': iter_best_fitness})

        # 保存最优结果对应的预测温度场
        evaluation_layout(ns.pg, save_name=save_name)


