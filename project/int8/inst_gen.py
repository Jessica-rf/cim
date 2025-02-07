class InstGenerator:

    def __init__(self):
        self.format_4 = 1       # 1:输出指令时, 按照xxxx_xxxx_xxxx_xxxx的格式输出; 0:不进行格式化, xxxxxxxxxxxxxxxx格式输出
        self.data_bytes = 1     # 一个数据需要的byte数, 在fp32中为4字节, int8中为1字节
        self.vec_bytes = 64     # 一个vector中包含的byte数

    def formatter_4(self, input):
        return "_".join([input[j:j + 4] for j in range(0, len(input), 4)])

    def return_idx(self, name):
        if name == 'X': idx = 0
        elif name == 'Y': idx = 1
        elif name == 'Z': idx = 2
        elif name == 'V': idx = 3
        elif name == 'H': idx = 4
        elif name == 'R': idx = 5
        elif name == 'G': idx = 6
        elif name == 'T': idx = 7
        else: raise ValueError("Invalid name")
        return idx

    def gen_gpr_ldr(self, size, last):
        '''
        Args:
            size (int): 输入1的列数:  (a,b)*(b,c)中的b
            last (int): number of address for last segment
        Returns:
            六行，格式为'XXXX_XXXX_XXXX_XXXX'
        '''
        result = []
        result.append(f"1000000000000001")
        result.append(f"0011000000000100")  # T link to R, X active mode, int8模式
        result.append(f"{last:04b}000101010101")
        result.append(f"1000000000001001")
        if (size > self.vec_bytes/self.data_bytes):
            result.append(f"1111111111111111")
        else:
            result.append(f"0000000000000000")
        result.append(f"{'1' * (2 * last):<016}".replace(' ', '0'))
        if self.format_4:
            for i in range(len(result)):
                formatted_string = self.formatter_4(result[i])
                result[i] = formatted_string
        return result

    def gen_cnt_ldr(self, name, cnt, step, offset):
        """
        Args:
            name (int): 计数器的编号
            cnt (int): cnt
            step (int): step
            offset (int): offset
        Returns:
            四行，格式为'XXXX_XXXX_XXXX_XXXX'
        """
        offset &= 0b1111111111 #补码表示负数
        idx = self.return_idx(name)
        result = []
        result.append(f"1000000000{idx:03b}011")
        result.append(f"{(cnt-1):016b}")
        result.append(f"{(step):016b}")
        result.append(f"000000{(offset):010b}")
        if self.format_4:
            for i in range(len(result)):
                formatted_string = self.formatter_4(result[i])
                result[i] = formatted_string
        return result

    def gen_ldr_ptr(self, name, offset):
        '''
        Args:
            idx (int): 计数器的编号
            offset (int): offset
        Returns:
            2行，格式为'XXXX_XXXX_XXXX_XXXX'
        '''
        idx = self.return_idx(name)
        result = []
        result.append(f"1000000000{idx:03b}010")
        result.append(f"{offset:016b}")
        if self.format_4:
            for i in range(len(result)):
                formatted_string = self.formatter_4(result[i])
                result[i] = formatted_string
        return result

    def gen_temp(self):
        result = []        
        result.append(f"1011000000000000")
        result.append(f"1011000000000000")
        result.append(f"1011000000000000")
        result.append(f"1001000100111000")
        result.append(f"1001000010101000")
        result.append(f"1010000001011110")
        result.append(f"1111111101000001")
        result.append(f"1111001110011101")
        result.append(f"1111001110101110")
        result.append(f"0000000000000000")
        if self.format_4:
            for i in range(len(result)):
                formatted_string = self.formatter_4(result[i])
                result[i] = formatted_string
        return result

# # 字典中嵌套字典
# counter = {
#     'X': {
#         'CNT': 1,
#         'STEP': 2,
#         'OFFSET': 3
#     },
#     'V': {
#         'CNT': 1,
#         'STEP': 2,
#         'OFFSET': 3
#     },
#     'R': {
#         'CNT': 1,
#         'STEP': 2,
#         'OFFSET': 3
#     },
#     'T': {
#         'CNT': 1,
#         'STEP': 2,
#         'OFFSET': 3
#     },
# }

# # Example usage
if __name__ == "__main__":

#     cnt = 3
#     step = 2
#     offset = 5
    binary_strings = []
    formatter = InstGenerator()
    formatter.gen_cnt_ldr("R", 1, 1, 3)
#     binary_strings.append(formatter.gen_cnt_ldr(0, 11, 9, 0))
#     binary_strings.append(formatter.gen_cnt_ldr(0, 11, 9, 0))
#     binary_strings.append(formatter.gen_cnt_ldr(0, 11, 9, 0))
#     binary_strings.append(formatter.gen_cnt_ldr(0, 11, 9, 0))

#     for binary_string in binary_strings:
#         print(binary_string)
