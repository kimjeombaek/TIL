# -*- coding: utf-8 -*-
# 치킨댄스를 추는 곰곰이를 본 임스 2

N = int(input())
if 1 <= N <= 1000:
    result = [int(input().split('-')[-1]) for _ in range(N)]
print(len([n for n in result if n <= 90]))
