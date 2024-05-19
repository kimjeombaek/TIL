# -*- coding: utf-8 -*-
# 평균은 넘겠지

C = int(input())

for _ in range(C):
    N, *score = list(map(int, input().split()))
    result = round(len([n for n in score if n > sum(score)/N])/N * 100, 3)
    print(f"{result}%")
