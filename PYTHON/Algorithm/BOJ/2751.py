# 수 정렬하기 2
from sys import stdin

N = int(input())
num_list = []
for _ in range(N):
    num_list.append(int(stdin.readline()))
[print(n) for n in sorted(num_list)]