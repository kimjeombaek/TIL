# -*- coding: utf-8 -*-
# 소-난다

N, M = map(int, input().split())
H = sorted(list(map(int, input().split())))
cnt = int(sum(range(1, N+1)) / (sum(range(1, N+1)) - sum(range(1, M+1))) * sum(range(1, M+1)))