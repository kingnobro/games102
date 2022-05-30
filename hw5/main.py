import matplotlib.pyplot as plt
import numpy as np


def Chaiukin(points):
    ps = points
    new_ps = []
    n = len(ps)
    k1 = 3 / 4
    k2 = 1 / 4
    for i in range(n):
        Q = k1 * ps[i] + k2 * ps[(i + 1) % n]
        R = k2 * ps[i] + k1 * ps[(i + 1) % n]
        new_ps.append(Q)
        new_ps.append(R)
    return new_ps


def uniform_cubic_spline(points):
    ps = points
    new_ps = []
    n = len(ps)
    for i in range(n):
        Q = 1/8 * ps[(i - 1) % n] + 3/4 * ps[i]  + 1/8 * ps[(i + 1) % n]
        R = 1/2 * ps[i]  + 1/2 * ps[(i + 1) % n]
        new_ps.append(Q)
        new_ps.append(R)
    return new_ps


def interpolate_subdivision(points):
    ps = points
    new_ps = []
    n = len(ps)
    alpha = 1 / 8
    for i in range(n):
        new_ps.append(ps[i])
        mid1 = (ps[i] + ps[(i+1)%n]) / 2
        mid2 = (ps[(i-1)%n] + ps[(i+2)%n]) / 2
        p = mid1 + alpha * (mid1 - mid2)
        new_ps.append(p)
    return new_ps


def main():
    fig = plt.figure()

    # 初始矩形
    points = np.array([
        [ 3.0,  3.0],
        [ 2.0, -3.0],
        [-3.0, -2.0],
        [-4.0,  1.0],
        [ 0.0,  4.0],
    ])

    num_subdivision = 10
    for k in range(num_subdivision):
        # 清空坐标
        plt.clf()
        plt.xlim((-5, 5))
        plt.ylim((-5, 5))

        # 画点
        # x = [p[0] for p in points]
        # y = [p[1] for p in points]
        # plt.scatter(x, y, color='r')

        # 画点之间的线段
        n = len(points)
        for i in range(n):
            x = [points[i][0], points[(i + 1) % n][0]]
            y = [points[i][1], points[(i + 1) % n][1]]
            plt.plot(x, y, color='b')
        plt.pause(0.5)

        # 细分曲线
        # points = Chaiukin(points)
        # points = uniform_cubic_spline(points)
        points = interpolate_subdivision(points)
    
    plt.show()


if __name__ == "__main__":
    main()