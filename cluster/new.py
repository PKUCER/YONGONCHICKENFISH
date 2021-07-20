import matplotlib.pyplot as plt
import math
import sys
from visualize import *


def dfs(data, vis, i, thresh):  # 给定一个目标，递归搜索和它在同一个集群内的所有目标

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    vis[i] = 1
    for j in range(len(data)):
        if vis[j] == 0 and dist(data[i], data[j]) < thresh:
            vis = dfs(data, vis, j, thresh)
    return vis


def detect(year):
    dist_thresh = 500
    size_thresh = 3
    filename = './data/{}.txt'.format(year)
    save_path = './result/'

    task = filename.split('/')[-1].split('.')[0]
    fin = open(filename, 'r')
    data = fin.read().strip().split('\n')
    fin.close()
    for i in range(len(data)):
        tmp = data[i].split()
        data[i] = (eval(tmp[0]), eval(tmp[1]))

    scale = 111000
    dist_thresh = dist_thresh / scale

    group = [0] * len(data)
    result = []
    p = 1
    for i in range(len(data)):
        if group[i] != 0:
            continue
        vis = dfs(data, [0] * len(data), i, dist_thresh)
        choice = []
        min_x = math.inf
        min_y = math.inf
        for j in range(len(data)):
            if vis[j] == 1:
                choice.append((data[j][0], data[j][1], j))
                min_x = min(min_x, data[j][0])  #最小经度
                min_y = min(min_y, data[j][1])  #最小纬度
                group[j] = p
        if len(choice) >= size_thresh:
            print('Found size {} cluster.'.format(len(choice)))
            result.append((min_x+min_y*1000, choice))
        p += 1

    result.sort(key=lambda x:x[0])  #按最小纬度+最小经度的字典序升序排序

    fout = open(save_path + '{}.txt'.format(task), 'w')
    fdetail = open(save_path + '{}_集群信息.txt'.format(task), 'w')
    mem = [0] * len(data)

    for idx, (_, cluster) in enumerate(result):
        xmin, ymin, xmax, ymax = math.inf, math.inf, -math.inf, -math.inf   
        for item in cluster:
            xmin = min(xmin, item[0])
            xmax = max(xmax, item[0])
            ymin = min(ymin, item[1])
            ymax = max(ymax, item[1])          
        fout.write('cluster\n')
        fout.write('{} {} {} {} {}\n'.format(len(cluster), xmin, ymin, xmax, ymax))   
        fdetail.write('集群{}：大小{}，经纬度范围{} {} {} {}，包含目标编号：'.format(idx, len(cluster), xmin, ymin, xmax, ymax))
        for item in cluster:
            fout.write('{} {}\n'.format(item[0], item[1]))   
            fdetail.write('{}号 '.format(item[2])) 
            mem[item[2]] = 1
        fdetail.write('\n')
    fdetail.write('不在集群中的目标编号：')
    for k in range(len(data)):
        if mem[k] == 0:
            fdetail.write('{}号 '.format(k))
    fout.close()
    fdetail.close()
    

def main():
    years = (2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018)
    # years = (2010,)
    for year in years:
        detect(year)  # 在result目录下生成集群的检测结果
        mapcut_single(year)  
    comparison(years)


if __name__ == '__main__':
    main()

