# Name: MINE_simple
# Author: Reacubeth
# Time: 2020/12/15 18:49
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


class MINE(nn.Module):
    def __init__(self, data_dim=256, hidden_size=512):
        super(MINE, self).__init__()
        self.layers = nn.Sequential(nn.Linear(2 * data_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x, y):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x, ], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = y[idx]
        concat_y = torch.cat([y, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size]
        pred_x_y = logits[batch_size:]
        loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        # compute loss, you'd better scale exp to bit
        return loss

def cal_mi(x_sample,y_sample):
    model = MINE().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    plot_loss = []
    all_mi = []
    for epoch in tqdm(range(100)):
        loss = model(x_sample, y_sample)

        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        all_mi.append(-loss.item())
    return max(all_mi)

if __name__ == '__main__':
    x_sample = gen_x(num_instances, data_dim)
    y_sample = gen_y(x_sample, num_instances, data_dim)

    x_sample = torch.from_numpy(x_sample).float().cuda()
    y_sample = torch.from_numpy(y_sample).float().cuda()
    print(cal_mi(x_sample,y_sample))
    
