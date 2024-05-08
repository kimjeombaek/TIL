# 행복한지 슬픈지

H = ":-)"
S = ":-("
sentence = input()
hc = sentence.count(H)
sc = sentence.count(S)
if hc > sc:
    print('happy')
elif hc < sc:
    print('sad')
elif hc + sc == 0:
    print('none')
elif hc == sc:
    print('unsure')