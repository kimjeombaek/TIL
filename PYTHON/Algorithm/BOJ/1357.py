# -*- coding: utf-8 -*-
# 뒤집힌 덧셈

def Rev(X: str) -> int:
    return int(X[::-1])

X, Y = input().split()
print(Rev(str(Rev(X) + Rev(Y))))
