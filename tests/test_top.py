import os
from tqdm import tqdm
script_dir = os.path.dirname(os.path.abspath(__file__))
from project.int8.main import *


def martix_mult_test():
    '''
    矩阵乘法 (a,b)*(b,c)

    martix_mult(a, b, c)
    !!! a*c < 4096 否则会溢出 !!!
    args:
        a (int): 矩阵 a 的行数
        b (int): 矩阵 a 的列数和矩阵 b 的行数
        c (int): 矩阵 b 的列数
    return: 
        0: 成功
        1: 失败        
    '''
    # a > 4, b < 64, c = a
    assert martix_mult(5, 8, 5) == 0

    # a > 4, b > 64, c = a
    assert martix_mult(7, 70, 7) == 0

    # a > 4, b < 64, c > 4
    assert martix_mult(11, 25, 6) == 0

    # a > 4, b > 64, c > 4
    assert martix_mult(16, 90, 23) == 0

    for i in tqdm(range(1000), desc="martix_mult_test "):
        a, b, c = generate_random_numbers()
        assert martix_mult(a, b, c) == 0

def test_main():
    martix_mult_test()

