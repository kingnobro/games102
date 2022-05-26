import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import copy

# store selected points
points = []


class RBFModel(nn.Module):
    # n is the # of gauss kernel
    def __init__(self, n = 5):
        super(RBFModel, self).__init__()
        self.linear0 = nn.Linear(1, n);
        self.linear1 = nn.Linear(n, 1);
        self.c = 1 / np.sqrt(2 * np.pi);
    
    def forward(self, x):
        x = self.linear0(x)
        x = torch.mul(torch.pow(x, 2), -0.5)
        x = torch.mul(torch.exp(x), self.c)
        x = self.linear1(x)
        return x


def trainer(points, model, config):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 

    loss_record = []

    for epoch in range(config['n_epochs']):
        optimizer.zero_grad()
        for p in points:
            x = torch.Tensor([p[0]])
            y = torch.Tensor([p[1]])
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            loss_record.append(loss.detach().item())
        optimizer.step()
        mean_loss = sum(loss_record) / len(loss_record)

        if epoch % 10 == 0:
            print('epoch:{} loss:{}'.format(epoch, mean_loss))
        

# click on the canvas, draw and save the points
def onclick(event):
    x, y = event.xdata, event.ydata
    points.append([x, y])
    print('click ({:.2f}\t{:.2f})'.format(x, y))

    plt.scatter(x, y, color='r')
    plt.draw()


def draw_model(model, label, color):
    x = np.linspace(-5, 5, 100) 
    y = [model(torch.Tensor([xi])).item() for xi in x]
    plt.plot(x, y, label=label, c=color)


def main():
    config = {
        'n_epochs': 500,
        'learning_rate': 1e-2,
    }

    model = RBFModel(7)
    init_model = copy.deepcopy(model)

    # first pass: click on the screen
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()

    trainer(points, model, config)
    
    # second pass: compare two models
    plt.scatter([p[0] for p in points], [p[1] for p in points], color='r')
    draw_model(init_model, label='Init Model', color='g')
    draw_model(model, label='RBF Model', color='b')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
