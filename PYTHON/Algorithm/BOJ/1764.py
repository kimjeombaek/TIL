# 듣보잡

N, M = list(map(int, input().split()))

_n = {input() for n in range(N)}
_m = {input() for n in range(M)}

result = sorted(_n & _m)
print(len(result))
[print(name) for name in result]
