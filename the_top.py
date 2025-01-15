from project.main import *
# 用于手动debug及生成数据
if __name__ == '__main__':
   # (a,b)*(b,c)
    ### a*c < 4096
    
    a,b,c = 2,5,3
    # a,b,c = 10,90,9
    # a,b,c = 155,180,10
    # a,b,c = 9,80,10
    if a * c > 4096:
        print("error! a*c should not bigger than 4096")
        exit()
    # martix_mult_with_output(a, b, c)
    
    a, b, c = generate_random_numbers()
    print(f"({a}, {b})*({b}, {c})")
    martix_mult_with_output(a, b, c)
    # Test for cim_conv2d
