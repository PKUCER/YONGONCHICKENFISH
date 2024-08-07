import math
import sys
from visualize import *


def dfs(data, vis, i, thresh):  # recursively search targets that are in the same cluster as the given target

    def dist(a, b):
        x_a = (a[0] + a[2]) / 2
        y_a = (a[1] + a[3]) / 2
        x_b = (b[0] + b[2]) / 2
        y_b = (b[1] + b[3]) / 2
        return ((x_a - x_b) ** 2 + (y_a - y_b) ** 2) ** 0.5

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
        data[i] = (eval(tmp[0]), eval(tmp[1]), eval(tmp[2]), eval(tmp[3]))

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
                choice.append((data[j][0], data[j][1], data[j][2], data[j][3], j))
                min_x = min(min_x, data[j][0])  # minimum longitude
                min_y = min(min_y, data[j][1])  # minimum latitude
                group[j] = p
        if len(choice) >= size_thresh:
            print('Found size {} cluster.'.format(len(choice)))
            result.append((min_x+min_y*1000, choice))
        p += 1

    result.sort(key=lambda x:x[0])

    fout = open(save_path + '{}.txt'.format(task), 'w')
    fdetail = open(save_path + '{}_cluster_info.txt'.format(task), 'w')
    mem = [0] * len(data)

    for idx, (_, cluster) in enumerate(result):
        xmin, ymin, xmax, ymax = math.inf, math.inf, -math.inf, -math.inf   
        for item in cluster:
            xmin = min(xmin, item[0])
            xmax = max(xmax, item[2])
            ymin = min(ymin, item[1])
            ymax = max(ymax, item[3])          
        fout.write('cluster\n')
        fout.write('{} {} {} {} {}\n'.format(len(cluster), xmin, ymin, xmax, ymax))   
        fdetail.write('cluster {}: size {}, range {} {} {} {}, containing target id: '.format(idx, len(cluster), xmin, ymin, xmax, ymax))
        for item in cluster:
            fout.write('{} {} {} {}\n'.format(item[0], item[1], item[2], item[3]))   
            fdetail.write('{} '.format(item[4])) 
            mem[item[4]] = 1
        fdetail.write('\n')
    fdetail.write('Target ids not in any clusters: ')
    for k in range(len(data)):
        if mem[k] == 0:
            fdetail.write('{} '.format(k))
    fout.close()
    fdetail.close()
    

def main():
    # years = range(2010, 2022)
    years = range(20230827, 20230828)
    for year in years:
        detect(year)
        mapcut_single(year, size=200, annotate=False)
        # mapcut_single(year, size=500, annotate=False)
        # mapcut_single(year, size=1000, annotate=False)  
        # mapcut_cluster(year)
    # comparison(years)


if __name__ == '__main__':
    main()

