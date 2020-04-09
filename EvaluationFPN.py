import numpy as np
import torch
import scipy.io as io

save_path = 'layout_design/Output'

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
        model_path = 'modelfile/fpn_x2y_ex.pth'
        # model_path = 'modelfile/fpn_x2y_5w.pth'

    print("model path:", model_path)

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, device


def evaluation_layout(list):
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
    io.savemat(save_path + '/Evaluation_' + str(6) + '.mat',
               {'list': list, 'pred_heat': pred_heat_numpy, 'norm_indicator': temp})


if __name__ == "__main__":

    # 验证 1：类均匀算例 using 5w
    # layoutlist = np.array([1,3,5,7,9,31,33,35,37,39,61,63,65,67,69,91,93,95,97,99])

    # 验证 2：随机算例 using 5w
    # layoutlist = np.array([3,7,12,18,20,22,25,28,33,34,45,46,48,55,61,63,64,65,73,84])

    # 验证3：极端算例1 using 5w
    # layoutlist =  np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    # 验证4：极端算例2 using 5w
    # layoutlist = np.array([81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100])

    # 验证5：极端算例1 using ex
    # layoutlist = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

    # 验证6：极端算例2 using ex
    layoutlist = np.array([81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100])


    evaluation_layout(layoutlist)

