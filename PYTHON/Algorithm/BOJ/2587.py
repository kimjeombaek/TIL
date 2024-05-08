# 대표값2

result = []
for i in range(5):
    result.append(int(input()))

print(int(sum(result)/5))
print(sorted(result)[len(result) // 2])