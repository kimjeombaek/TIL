# 수 찾기

N = int(input())
A = set(list(map(int, input().split())))
M = int(input())
number = list(map(int, input().split()))

for num in number:
    if num in A:
        print(1)
    else: print(0)
