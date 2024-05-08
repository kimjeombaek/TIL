# 8진수, 10진수, 16진수

X = input()
if '0x' in X:
    print(int(X, 16))
elif X[0] == '0':
    print(int(X, 8))
else:
    print(int(X))