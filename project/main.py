# import sys
import os
import random
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from project.cim_hw_sim import *
from project.inst_gen import *

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def write_data_to_file(file, M, mode="a"):
    """
    将数组 M 的内容转换为格式化的十六进制字符串并写入文件。
    - 每次运行主函数时覆盖文件内容（通过指定 mode="w" 调用一次）。
    - 单次运行内多次调用时追加数据（mode 默认为 "a"）。
    """
    cnt_line = 0
    with open(file, mode) as f:
        for arr in M:
            rows = math.ceil(len(arr) / 16)
            for r in range(rows):
                line = ""
                for s in range(16):
                    index = r * 16 + s
                    if index >= len(arr):
                        line += "00"  # 补齐不足16字节的部分
                    else:
                        line += f"{arr[index] & 0xFF:02x}"
                f.write(line + "\n")  # 写入每一行十六进制字符串
                cnt_line = cnt_line + 1
        if mode == "w":
            f.write(f"@{hex(len(M) * 4)[2:]}\n")
    return cnt_line


def write_inst_to_file(file, inst, mode="w"):
    with open(file, mode) as f:
        for line in inst:
            f.write(line + "\n")  # 写入每一行十六进制字符串


def martix_mult(aa, bb, cc):
    print(f"---> Cim_matrix_mult_test ({aa},{bb})*({bb},{cc})", end=',  ')
    # d1 = np.arange(aa * bb, dtype=np.int32).reshape((aa, bb))
    # d2 = np.arange(bb * cc, dtype=np.int32).reshape((bb, cc))
    d1 = np.arange(aa * bb, dtype=np.int32).reshape((aa, bb)) % 11
    d2 = np.arange(bb * cc, dtype=np.int32).reshape((bb, cc)) % 13 - 6

    M0, M1 = cim_matrix_load(d1, d2)
    out, data = cim_matrix_mult_sim(M0, M1, d1.shape, d2.shape)
    ref = d1.dot(d2)

    if np.array_equal(ref, out):
        print(f"passed\n")
        return 0
    else:
        print(f"failed\n")
        return 1


def martix_mult_with_output(aa, bb, cc):
    output_folder = os.path.join(script_dir, "output")
    # output_folder = os.path.join("D:\work\_VMshare\CIM", "output")
    data_file_name = os.path.join(output_folder, "data.txt")
    inst_file_name = os.path.join(output_folder, "inst.txt")
    ref1_file_name = os.path.join(output_folder, "ref1.txt")
    out1_file_name = os.path.join(output_folder, "out1.txt")
    cnt_line = 0
    os.makedirs(output_folder, exist_ok=True)

    # d1 = np.arange(aa * bb, dtype=np.int32).reshape((aa, bb))
    # d2 = np.arange(bb * cc, dtype=np.int32).reshape((bb, cc))
    d1 = np.arange(aa * bb, dtype=np.int32).reshape((aa, bb)) % 11
    d2 = np.arange(bb * cc, dtype=np.int32).reshape((bb, cc)) % 13 - 6

    M0, M1 = cim_matrix_load(d1, d2)
    # M0,M1 = cim_matrix_load_b(d1,d2)

    cnt_line += write_data_to_file(data_file_name, M0, "w")
    cnt_line += write_data_to_file(data_file_name, M1, "a")

    # print_mem_hex(M0)
    # print_mem_hex(M1)
    out, data = cim_matrix_mult_sim(M0, M1, d1.shape, d2.shape)
    # print(out)
    ref = d1.dot(d2)
    with open(ref1_file_name, 'w') as file:
        # 将 ref 转换为字符串并写入文件
        for i in ref:
            file.write(str(i) + "\n")

    with open(out1_file_name, 'w') as file:
        # 将 out 转换为字符串并写入文件
        for i in out:
            file.write(str(i) + "\n")

    print(data)
    if np.array_equal(ref, out):
        print("---> Cim_matrix_mult_test passed")
        inst = []
        GEN1 = InstGenerator()
        inst.extend(GEN1.gen_gpr_ldr(bb, data["Last"]))
        inst.extend(GEN1.gen_cnt_ldr("R", data["R"], data["RS"], -3))
        inst.extend(GEN1.gen_cnt_ldr("V", data["V"], data["VS"], -2))
        inst.extend(GEN1.gen_cnt_ldr("X", data["X"], data["XS"], 0))
        inst.extend(GEN1.gen_cnt_ldr("T", data["R"], cc, 0))
        inst.extend(GEN1.gen_ldr_ptr("R", 16))
        inst.extend(GEN1.gen_ldr_ptr("T", 0))
        inst.extend(GEN1.gen_ldr_ptr("G", 16 + len(M0) * 8))
        inst.extend(GEN1.gen_temp())
        write_inst_to_file(inst_file_name, inst, "w")
        print(f'int write_cycle = {cnt_line//4}-1;')
        print(f'int read_cycle = {math.ceil(aa*cc/4)};')
        print(f'reg [64*8:0] memfile="../sw/inst.txt";')
        print(f'reg [64*8:0] datfile="../sw/data.txt";')
        return 0
    else:
        # print("---> Cim_matrix_mult_test failed\n", out, "\n", ref)
        print("---> Cim_matrix_mult_test failed\n")
        return 1


def generate_random_numbers():
    while True:
        a = random.randint(1, 4096)  # a can be an integer
        b = random.randint(5, 199)  # b must be in the range (4, 200)
        c = random.randint(1, 4096)  # c can also be an integer

        if a * c < 4096:
            return a, b, c


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
