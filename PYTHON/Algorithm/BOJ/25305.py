# -*- coding: utf-8 -*-
# 커트라인

N, k = list(map(int, input().split()))
print(sorted(list(map(int, input().split())), reverse=True)[k-1])