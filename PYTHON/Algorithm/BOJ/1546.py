# -*- coding: utf-8 -*-
# 평균

N = int(input())
score = list(map(int, list(input().split())))
M = [s/max(score)*100 for s in score]
print(sum(M)/len(M))
