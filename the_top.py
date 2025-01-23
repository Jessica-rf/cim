from project.main import *
def martix_mult_test1():
    # (a,b)*(b,c)
    ### a*c < 4096
    # a,b,c = 10,90,9
    # a,b,c = 155,180,10
    # a,b,c = 9,80,10
    
    #TEST
    a,b,c = 2,3,3 #PASS
    a,b,c = 2,31,3 #
    # a,b,c = 2,35,3 #
    # a,b,c = 28,35,17 #FAIL
    if a * c > 4096:
        print("error! a*c should not bigger than 4096")
        exit()
    martix_mult_with_output(a, b, c)
    exit()
    
    a, b, c = generate_random_numbers()
    print(f"({a}, {b})*({b}, {c})")
    martix_mult_with_output(a, b, c)
    
def conv_test1():
    a,b,c,d = 5,2,2,2
    d = np.arange(a*b*c*d, dtype=np.int32).reshape((a,b,c,d))
    M0 = cim_load_weight(d,mem=[])
    with open('output1.txt', 'w') as file:
        for item in M0:
            file.write(f"{item}\n")
    # M1 = cim_load_weight(d,M0)

    # run_conv2d_sim_1(W1,W2)
    
# 用于手动debug及生成数据
if __name__ == '__main__':
    martix_mult_test1()
    # conv_test1()
    # Test for cim_conv2d
