# 2022 年国赛 B 题求解代码

本目录包含《无人机遂行编队飞行中的纯方位无源定位》的全部求解代码。

## 文件说明

- `solve_problemB.py`：主求解程序，输出问题 1(1)、1(2)、1(3) 和问题 2 的数值结果，并生成 `problemB_results/results.json`。
- `make_figures.py`：调用 `solve_problemB.py` 中的数据和函数，生成论文插图到 `figures/` 目录。
- `results.json`：运行 `solve_problemB.py` 后得到的标准结果，便于核对。

## 环境要求

- Python 3.8 及以上
- numpy
- scipy
- matplotlib

安装依赖：

```bash
pip install numpy scipy matplotlib
```

## 运行方法

```bash
python solve_problemB.py
python make_figures.py
```

运行后将在当前目录生成 `problemB_results/results.json` 和 `figures/` 中的四张插图。

## 主要结果

- 问题 1(1)：三架已知位置发射机可精确定位，标称仿真定位误差低于 1e-8 m。
- 问题 1(2)：除 FY00、FY01 外，还需 2 架编号未知的无人机发射信号。
- 问题 1(3)：两轮调整加一轮校验即可完成，单步最大位移 12.011 m，最终最大误差低于 1e-6 m。
- 问题 2：锥形编队示例两轮调整完成，单步最大位移 3.601 m，最终最大误差低于 1e-6 m。
