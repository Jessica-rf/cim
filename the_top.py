from project.top import *

if __name__ == '__main__':
   # (a,b)*(b,c)
    ### a*c < 4096
    a = 6
    b = 9
    c = 6
    if a * c > 4096:
        print("error! a*c should not bigger than 4096")
        exit()

    martix_mult_with_output(a, b, c)
    # Test for cim_conv2d
