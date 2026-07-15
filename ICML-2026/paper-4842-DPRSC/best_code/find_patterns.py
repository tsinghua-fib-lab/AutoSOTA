import array
import numpy as np
from multiprocessing import Pool, cpu_count, Manager

def cmp(x, y, deg):
    return deg[x] < deg[y] or (deg[x] == deg[y] and x < y)

def enumerate_triangles(edges, deg, n):
    adj_list = [array.array('i') for _ in range(n)]
    for edge in edges:
        x, y = edge
        if not cmp(x, y, deg):
            x, y = y, x
        adj_list[x].append(y)
        
    cnt = 0
    vis = array.array('i', [-1] * n)
    for x in range(n):
        for y in adj_list[x]:
            vis[y] = x
        for y in adj_list[x]:
            for z in adj_list[y]:
                if (vis[z] == x):
                    cnt += 1
    return cnt

def enumerate_2stars(deg, n):
    cnt = 0
    for x in range(n):
        cnt += deg[x] * (deg[x] - 1) // 2
    return cnt

def enumerate_triangles_project(edges, deg, n, h, d):
    adj_list = [array.array('i') for _ in range(n)]
    for edge in edges:
        x, y = edge
        if not cmp(x, y, deg):
            x, y = y, x
        adj_list[x].append(y)

    index_count = {}
    vis = array.array('i', [-1] * n)
    for x in range(n):
        for y in adj_list[x]:
            vis[y] = x
        for y in adj_list[x]:
            for z in adj_list[y]:
                if (vis[z] == x):
                    interval = []
                    for j in range(d):
                        sorted_list = sorted([h[x, j], h[y, j], h[z, j]])
                        interval.extend([sorted_list[0], sorted_list[2]])
                    interval = tuple(interval)   
                    index_count[interval] = index_count.get(interval, 0) + 1
    return index_count

def enumerate_2stars_project(edges, n, h, d):
    adj_list = [array.array('i') for _ in range(n)]
    for edge in edges:
        x, y = edge
        adj_list[x].append(y)
        adj_list[y].append(x)

    index_count = {}
    for x in range(n):
        len_adj = len(adj_list[x])
        for i in range(len_adj):
            y = adj_list[x][i]
            for j in range(i + 1, len_adj):
                z = adj_list[x][j]
                interval = []
                for k in range(d):
                    sorted_list = sorted([h[x, k], h[y, k], h[z, k]])
                    interval.extend([sorted_list[0], sorted_list[2]])
                interval = tuple(interval)
                index_count[interval] = index_count.get(interval, 0) + 1
    return index_count

def enumerate_edges_project(edges, h, d):
    index_count = {}
    for edge in edges:
        x, y = edge
        interval = []
        for k in range(d):
            sorted_list = sorted([h[x, k], h[y, k]])
            interval.extend([sorted_list[0], sorted_list[1]])
        interval = tuple(interval)
        index_count[interval] = index_count.get(interval, 0) + 1
    return index_count