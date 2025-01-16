from project.main import *
def martix_mult_test1():
    # (a,b)*(b,c)
    ### a*c < 4096
    a,b,c = 2,3,3
    # a,b,c = 10,90,9
    # a,b,c = 155,180,10
    # a,b,c = 9,80,10
    if a * c > 4096:
        print("error! a*c should not bigger than 4096")
        exit()
    martix_mult_with_output(a, b, c)
    exit()
    
    a, b, c = generate_random_numbers()
    print(f"({a}, {b})*({b}, {c})")
    martix_mult_with_output(a, b, c)
    
def conv_test1():
    pass
# 用于手动debug及生成数据
if __name__ == '__main__':
    martix_mult_test1()
    # conv_test1()
    # Test for cim_conv2d
