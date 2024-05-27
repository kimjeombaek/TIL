# -*- coding: utf-8 -*-
# 에라토스테네스의 체

N, K = map(int, input().split())

prime_number = [True for _ in range(N+1)]
num = 1

for i in range(2, N+1):
    if prime_number[i] == True:
        for j in range(i, N+1, i):
            if prime_number[j] == False:
                continue
            if num == K:
                print(j)
            prime_number[j]= False
            num += 1
