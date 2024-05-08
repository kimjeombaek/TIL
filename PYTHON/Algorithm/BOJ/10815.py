# 숫자카드
from sys import stdin

N = int(input())
s_score = list(map(int, stdin.readline().split()))
M = int(input())
b_score = list(map(int, stdin.readline().split()))
result = {}
for idx, num in enumerate(b_score):
    if num in s_score:
        result[idx] = 1
print(*result)