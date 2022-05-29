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


def three_moment_equation(x, y):
    # len(x) 个点, 下标范围为 0, 1, ..., len(x)-1 
    n = len(x) - 1
    
    # v1 b1 的需要计算 h0 b0
    h = [x[1] - x[0]]
    b = [(y[1] - y[0]) * 6 / h[0]]
    u, v = [], []
    for i in range(1, n):
        h.append(x[i+1] - x[i])
        u.append(2 * (h[i] + h[i-1]))
        b.append((y[i+1] - y[i]) * 6 / h[i])
        v.append(b[i] - b[i-1])

    # 构造方程组
    A = np.zeros((n - 1, n - 1))
    B = np.array(v)
    for i in range(n - 1):
        A[i][i] = u[i]
        if i + 1 < n - 1:
            A[i][i+1] = h[i]
        if i > 0:
            A[i][i-1] = h[i-1]
    
    M = solve(A, B)
    # 自然边界条件 M0 = Mn = 0
    return [0] + M.tolist() + [0]


def draw_cubic_spline(x, y, M):
    def S(M1, M2, x1, x2, y1, y2, h, x):
        t1 = M1 * (x2 - x)**3 / (6 * h)
        t2 = M2 * (x - x1)**3 / (6 * h)
        t3 = (y2 / h - M2 * h / 6) * (x - x1)
        t4 = (y1 / h - M1 * h / 6) * (x2 - x)
        return t1 + t2 + t3 + t4

    # len(x) 个点, len(x)-1 段曲线
    n = len(x) - 1
    for i in range(n):
        hi = x[i+1] - x[i]
        px = np.linspace(x[i], x[i+1], 100)
        py = np.array([S(M[i], M[i+1], x[i], x[i+1], y[i], y[i+1], hi, e) for e in px])
        plt.plot(px, py, color='b')
        

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

    M = three_moment_equation(Xcoord, Ycoord)
    draw_cubic_spline(Xcoord, Ycoord, M)

    plt.show()


if __name__ == "__main__":
    main()