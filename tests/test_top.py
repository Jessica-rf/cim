import os
import sys
from tqdm import tqdm
script_dir = os.path.dirname(os.path.abspath(__file__))
from project.int8 import main as int8
from project.fp32 import main as fp32


def martix_mult_test_int8():
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
    assert int8.martix_mult(5, 8, 5) == 0

    # a > 4, b > 64, c = a
    assert int8.martix_mult(7, 70, 7) == 0

    # a > 4, b < 64, c > 4
    assert int8.martix_mult(11, 25, 6) == 0

    # a > 4, b > 64, c > 4
    assert int8.martix_mult(16, 90, 23) == 0

    for i in tqdm(range(1000), desc="martix_mult_test_int8 "):
        a, b, c = int8.generate_random_numbers()
        assert int8.martix_mult(a, b, c) == 0
    
    print("--------martix_mult_test_int8 passed--------")
    return 

def martix_mult_test_fp32():
    '''
    矩阵乘法 (a,b)*(b,c)

    martix_mult(a, b, c)
    !!! a*c < 1024 否则会溢出 !!!
    args:
        a (int): 矩阵 a 的行数
        b (int): 矩阵 a 的列数和矩阵 b 的行数
        c (int): 矩阵 b 的列数
    return: 
        0: 成功
        1: 失败        
    '''
    # a > 4, b < 64, c = a
    assert fp32.martix_mult(5, 8, 5) == 0

    # a > 4, b > 64, c = a
    assert fp32.martix_mult(7, 70, 7) == 0

    # a > 4, b < 64, c > 4
    assert fp32.martix_mult(11, 25, 6) == 0

    # a > 4, b > 64, c > 4
    assert fp32.martix_mult(16, 90, 23) == 0

    for i in tqdm(range(1000), desc="martix_mult_test_fp32 "):
        a, b, c = fp32.generate_random_numbers()
        assert fp32.martix_mult(a, b, c) == 0
    
    print("--------martix_mult_test_fp32 passed--------")
    return
def test_main():
    martix_mult_test_fp32()
    martix_mult_test_int8()
    

