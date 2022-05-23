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


# input: n points
# output: n weights
def interpolate_polynomial(points):
    A = []
    b = []
    n = len(points)
    # construct A, b
    for p in points:
        a = []
        x, y = p[0], p[1]
        b.append(y)
        for i in range(n):
            a.append(pow(x, i))
        A.append(a)
    weights = solve(np.array(A), np.array(b))
    return weights


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


def gauss(x, xi, sigma=1):
    return np.exp(-(x - xi)**2 / (2 * sigma))


# input: n points
# output: n+1 weights
def interpolate_gauss(points):
    n = len(points)
    A = []
    b = []
    for p in points:
        a = [1]
        x, y = p[0], p[1]
        b.append(y)
        for i in range(n):
            xi = points[i][0]
            a.append(gauss(x, xi))
        A.append(a)
    weights = solve(np.array(A), np.array(b))
    return weights


def eval_gauss(x, points, weights):
    y = []
    n = len(weights)
    for e in x:
        val = weights[0]
        for i in range(1, n):
            xi = points[i - 1][0]
            val += weights[i] * gauss(e, xi)
        y.append(val)
    return np.array(y)


def regression_LSM(points):
    n = len(points)
    m = n - 1   # 固定幂基函数最高次数 m < n
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


def regression_ridge(points):
    n = len(points)
    m = n - 1   # 固定幂基函数最高次数 m < n
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
    A = np.concatenate((np.array(A), np.eye(m)), axis=0)
    b = np.concatenate((np.array(b), np.zeros(m)))
    weights = solve(A, b)
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
    plt.scatter([p[0] for p in points], [p[1] for p in points], color='r')
    x = np.linspace(-5, 5, 100)

    # polynomial interpolation
    weights = interpolate_polynomial(points)
    y = eval_polynomial(x, weights)
    plt.plot(x, y, color='r', label='polynomial interpolation')

    # gauss interpolation
    weights = interpolate_gauss(points)
    y = eval_gauss(x, points, weights)
    plt.plot(x, y, color='g', label='gauss interpolation')

    # least squares method regression
    weights = regression_LSM(points)
    y = eval_polynomial(x, weights)
    plt.plot(x, y, color='b', label='LSM regression')

    # ridge regression
    weights = regression_ridge(points)
    y = eval_polynomial(x, weights)
    plt.plot(x, y, color='y', label='ridge regression')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
