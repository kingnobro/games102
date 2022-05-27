import matplotlib.pyplot as plt
import numpy as np


points = []


# Ax = b
# https://zhuanlan.zhihu.com/p/131097680
def solve(A, b):
    n = A.shape[0]
    m = A.shape[1]
    if n >= m:
        # 方程组个数大于变量个数
        u, s, vt = np.linalg.svd(A)
        x = np.matmul(vt.T, np.matmul(u.T, b)[:m] / s)
        return x
    elif n < m:
        # 方程组个数小于变量个数, 将前几个变量设置为 0
        start = m - n
        A = A[:, start:]
        u, s, vt = np.linalg.svd(A)
        x = np.matmul(vt.T, np.matmul(u.T, b)[:m] / s)
        x = np.concatenate((np.zeros(start), x), axis=0) # 补 0
        return x


# calculate y
def eval_polynomial(x, weights):
    y = []
    n = len(weights)
    for e in x:
        val = 0
        for i in range(n):
            val += weights[i] * pow(e, i)
        y.append(val)
    return np.array(y)


def regression_LSM(points):
    n = len(points)
    m = min(n - 1, 5)   # 固定幂基函数最高次数 m < n
    A = []
    b = []
    # construct A, b
    for p in points:
        a = []
        x, y = p[0], p[1]
        b.append(y)
        for i in range(m):
            a.append(pow(x, i))
        A.append(a)
    weights = solve(np.array(A), np.array(b))
    return weights


# 参数化曲线拟合
# x = x(t), t ∈ [0, 1]
def curve_fitting(x, parameterization):
    t = []
    n = len(x)
    if parameterization == 'uniform':
        step = 1 / (n - 1)
        for i in range(n):
            t.append(i * step)
    elif parameterization == 'chordal':
        t.append(0)
        for i in range(1, n):
            t.append(t[i - 1] + np.abs(x[i] - x[i-1]))
        t = [t[i] / t[-1] for i in range(n)]
    elif parameterization == 'centripetal':
        t.append(0)
        for i in range(1, n):
            t.append(t[i - 1] + np.sqrt(np.abs(x[i] - x[i-1])))
        t = [t[i] / t[-1] for i in range(n)]
    else:
        assert(False, 'Wrong Parameterization Way')
    
    ps = [[t[i], x[i]] for i in range(n)]
    weights = regression_LSM(ps)
    return weights


# click on the canvas, draw and save the points
def onclick(event):
    x, y = event.xdata, event.ydata
    points.append([x, y])
    print('click ({:.2f}\t{:.2f})'.format(x, y))

    plt.scatter(x, y, color='r')
    plt.draw()


def main():
    # first pass: click on the screen
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.show()

    # second pass: draw lines
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    Xcoord = [p[0] for p in points]
    Ycoord = [p[1] for p in points]
    plt.scatter(Xcoord, Ycoord, color='r')
    t = np.linspace(0, 1, 100)

    # uniform parameterization
    weights_X = curve_fitting(Xcoord, 'uniform')
    weights_Y = curve_fitting(Ycoord, 'uniform')
    x = eval_polynomial(t, weights_X)
    y = eval_polynomial(t, weights_Y)
    plt.plot(x, y, color='b', label='Normal')

    # chordal parameterization
    weights_X = curve_fitting(Xcoord, 'chordal')
    weights_Y = curve_fitting(Ycoord, 'chordal')
    x = eval_polynomial(t, weights_X)
    y = eval_polynomial(t, weights_Y)
    plt.plot(x, y, color='g', label='Chordal')

    # centripetal parameterization
    weights_X = curve_fitting(Xcoord, 'centripetal')
    weights_Y = curve_fitting(Ycoord, 'centripetal')
    x = eval_polynomial(t, weights_X)
    y = eval_polynomial(t, weights_Y)
    plt.plot(x, y, color='y', label='Centripetal')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()