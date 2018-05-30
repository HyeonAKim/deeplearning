

# 가중치와 편항도입

import numpy as np
x = np.array([0,1])
w = np.array([0.5,0.5])
b = -0.7
# 같은 배열 끼리 곱셈
print(w*x)
# 배열의 곱셈의 덧셈
print(np.sum(w*x))
# 덧셈의 편향값 더하기
print(np.sum(w*x)+b)


# AND 게이트
def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# print(AND(0,0))
# print(AND(0,1))
# print(AND(1,0))
# print(AND(1,1))

# NAND  게이트
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# OR 게이트
def  OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0 :
        return 0
    else:
        return 1

# 기존게이트 조합하기 : OR게이트
def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))