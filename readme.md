# CIM python部分

## 项目概述
本项目实现了一个基于硬件模型的量化MNIST图像分类神经网络，并提供了矩阵乘法和卷积操作的硬件仿真功能。项目使用Python编写，主要依赖于`numpy`库进行数值计算。


## 项目依赖
- Python 3.x
- numpy

安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

### 运行项目
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行主程序：
   ```bash
   python the_top.py
   ```

3. 运行测试：
   ```bash
   pytest 
   pytest --capture=no > test_output.txt # 输出到文件
   ```

### 参数说明
- `a`, `b`, `c`：矩阵乘法中的维度参数，需满足`a * c <= 4096`。


## 目录结构
```
   cim
    │  .gitignore           # Git忽略文件
    │  pytest.ini           # Pytest配置文件
    │  readme.md            # 项目说明文档
    │  requirements.txt     # Python依赖库
    │  log.txt              # 更新日志
    │  the_top.py           # 主程序入口文件
    │
    ├─.github               # GitHub Actions配置目录
    │  └─workflows
    │          ci.yml       # GitHub Actions配置文件
    │
    ├─project
    │      cim_hw_sim.py    # 硬件仿真核心代码
    │      inst_gen.py      # 指令生成器类
    │      top.py           # 主程序入口文件
    │      __init__.py      # 模块初始化文件
    │
    └─tests
            test_top.py     # 顶层测试文件

```


## 功能模块介绍

### `cim_hw_sim.py`
该文件实现了硬件仿真的核心功能，包括但不限于：
- **卷积层**：通过`conv2d_layer_hw_sim`函数实现卷积操作。
- **矩阵乘法**：通过`cim_matrix_mult_sim`函数实现矩阵乘法的硬件仿真。
- **权重加载**：通过`cim_load_weight`函数将权重数据加载到内存中。
- **特征图加载**：通过`cim_load_fmap`函数将特征图数据加载到内存中。
- **推理过程**：通过`model_infer_hw_sim`函数实现整个神经网络的推理过程。

### `inst_gen.py`
定义了`InstGenerator`类，用于生成硬件指令。主要方法包括：
- `gen_gpr_ldr`：生成通用寄存器加载指令。
- `gen_cnt_ldr`：生成计数器加载指令。
- `gen_ldr_ptr`：生成指针加载指令。
- `gen_temp`：生成临时指令。

### `top.py`
主要用于调用其他模块的功能并执行具体的测试任务。包含以下主要函数：
- `write_data_to_file`：将数据写入文件。
- `write_inst_to_file`：将指令写入文件。
- `martix_mult`：实现矩阵乘法并验证结果。
- `martix_mult_with_output`：实现矩阵乘法并将结果输出到文件。

### `the_top.py`
顶层文件，运行项目的主程序。

### `test/test_top.py`
用于添加在CI中的测试用例。

## 注意事项
- 项目中涉及的硬件仿真部分假设了特定的硬件架构和内存布局，请根据实际情况调整相关参数。
- 项目中的一些测试用例可能会生成输出文件，建议在运行前确认输出路径。

