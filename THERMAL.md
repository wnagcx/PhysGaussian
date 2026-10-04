# Gaussian 温度与热传导

伪代码公式、变量与函数的逐项对照见 [热传导伪代码与实现对照](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/THERMAL_PSEUDOCODE_MAPPING.md)。

按修改后的伪代码实现：每个 Gaussian / MPM 粒子携带 `temperature` 和三维 `temperature_gradient`，执行仿射温度 P2G、网格热边界更新、D3Q7 LBM 热扩散和温度 G2P。温度是被动的物理属性，不是训练参数；材料模型由原来的 `material` 参数决定。

## 场景配置

原配置添加 `thermal` 对象即可启用热传导。未添加时温度属性仍存在，默认 20，热求解器不运行。没有温度的 Gaussian PLY 可以直接读取，默认温度 20、梯度 0。使用一致的摄氏度或开尔文，示例采用摄氏度。

```json
"thermal": {
  "enabled": true,
  "initial_temperature": 20.0,
  "diffusivity": 0.02,
  "initial_regions": [
    {"point": [0.7, 1.0, 1.0], "size": [0.2, 0.8, 0.8], "temperature": 80.0}
  ],
  "boundary_conditions": [
    {"type": "fixed_temperature", "point": [0.7, 1.0, 1.0], "size": [0.2, 0.8, 0.8],
     "temperature": 80.0, "start_time": 0.0, "end_time": 2.0},
    {"type": "heat_source", "point": [1.3, 1.0, 1.0], "size": [0.1, 0.2, 0.2],
     "rate": 10.0, "start_time": 0.0, "end_time": 1.0}
  ]
}
```

`point` 是区域中心，`size` 是三个轴的半尺寸，均为预处理后的 MPM 空间坐标。`diffusivity` 的单位是该空间长度单位的平方/秒；若实际长度映射至 MPM 时放大 s 倍，物理热扩散率也需要乘 s²。它不是热导率 k；α = k/(ρ c)。示例中的 0.02 用于观察传播，不代表某种实际材料。

`initial_temperature` 覆盖输入 PLY 温度；省略它可继承 PLY 中的温度和梯度。`initial_regions` 设置起始热点；`fixed_temperature` 维持区域内有质量的网格节点温度；`heat_source.rate` 是每秒升温量，可为负表示冷却，多个源相加。固定温度优先于升温源，重叠固定温度按配置列表顺序覆盖。热边界采用 `[start_time, end_time)`，自动在启停时刻切分时间步。

在正常 PhysGaussian 环境中，从项目根目录运行：

```powershell
python gs_simulation.py --model_path ./model/ficus_whitebg-trained --config ./config/ficus_thermal_config.json --output_path ./output/ficus_thermal --render_img --render_temperature --output_temperature --output_h5 --temperature_range 20 80 --white_bg
```

`thermal_images` 保存蓝到红的温度伪彩图，采用 Gaussian 原有的透明度与协方差。颜色限幅仅用于显示，原始温度不限幅。这是伪彩色温度显示，不是热辐射亮度。

`temperature/0000.npz` 是初始状态，后续文件为每帧结束后的状态，包含：

- `gaussian_temperature`：原模型所有 Gaussian 温度 `(N, 1)`，保持原始编号。
- `gaussian_temperature_gradient`：原始场景坐标下的梯度 `(N, 3)`。
- `simulated_gaussian_indices`：模拟的原始 Gaussian 编号，对应 MPM 数组的前 K 项。
- `particle_temperature`、`particle_temperature_gradient`、`particle_position`：包括内部填充粒子的完整状态，梯度与位置处于 MPM 空间。
- `time`：仿真时间。

低透明度筛除或 `sim_area` 外的 Gaussian 保留温度属性，只有进入仿真的粒子参与传热。内部填充粒子也有温度并参与传热，启用填充可视化时也有温度颜色。Gaussian 克隆、分裂和裁剪同步更新温度及梯度。

`--output_ply` 的粒子 PLY 和 `--output_h5` 增加温度、梯度字段。粒子 PLY 仍是粒子状态文件；`GaussianModel.save_ply()` 保存包含 SH 等属性的完整 Gaussian PLY，也包含温度。`capture()/restore()` 支持温度和旧版训练检查点。

## 数值方法

`mpm_solver_warp/thermal_utils.py` 中：

1. 用与机械 APIC 相同的二次 B 样条，将 `m_p [T_p + (x_i - x_p)·∇T_p]` 累积至网格，再除以节点质量。
2. D3Q7 BGK 采用 τ=1，执行碰撞与传播。六个轴方向权重为 `r = α Δt/Δx²`，静止方向为 `1-6r`。权重随实际时间步缩放，避免固定权重在很小的 MPM 时间步上引入额外热扩散；r 超过 0.15 时自动细分。
3. 分布函数表示 `m_i T_i`，采用共同的比热。节点质量不同时，通过 `min(m_i,m_j)` 对称链路和部分反弹传递，保持热量守恒。均匀节点质量时退化为标准各向同性 D3Q7 扩散；质量变化处使用该质量相关的链路导通率。
4. 空网格及计算域外反弹，构成绝热边界。有质量的相邻节点可以传热；小于网格分辨率的间隙可能被视为接触，需要提高 `n_grid`。
5. G2P 更新 `T_p = Σ_i w_ip T_i` 和 `∇T_p = Σ_i T_i ∇w_ip`。它使用与 P2G 相同的位置，然后机械 G2P 再移动粒子。粒子运动已携带温度，因此纯热传导网格步骤不重复施加速度对流。

论文主文没有给出完整 LBM 离散细节，本实现选择上述 D3Q7 方案。参考 [D3Q7 标量输运说明](https://docs.aerosim.io/nassu/theory/scalar_transport/velocity_set.html) 和 [NVIDIA Warp 文档](https://github.com/NVIDIA/warp/blob/main/docs/user_guide/runtime.rst)。

P2G/G2P 插值也可能产生数值平滑，所以即使 `diffusivity=0`，不连续温度也不保证完全不变。暂停全部温度更新用 `enabled=false`。梯度项精确重现线性温度场，但不是限幅器，尖锐热点附近可能出现插值超调。当前使用单一热扩散参数、共同比热和绝热外部，不包含空气温度、辐射或热膨胀。

## 无训练模型的示例和测试

```powershell
python examples/thermal_conduction.py --device cuda:0 --steps 100 --output ./output/thermal_demo
python -m unittest discover -s tests -v
```

示例通过真实 `MPM_Simulator_WARP.p2g2p()` 模拟机械固定、左半 80 度右半 20 度的块体，保存初末粒子 PLY/H5、温度历史和守恒报告；不依赖训练模型、Taichi 或 Gaussian CUDA 渲染扩展。

本机测试依赖在 `.thermal-test-deps`，未改全局 Python 包。可直接运行：

```powershell
$env:PYTHONPATH = (Resolve-Path .thermal-test-deps).Path
E:/anaconda/python.exe examples/thermal_conduction.py --device cuda:0
E:/anaconda/python.exe -m unittest discover -s tests -v
```

该依赖目录适配本机 Python 3.13。正式场景运行需要 README 要求的 Python 3.9、CUDA PyTorch、Taichi 和两个 Gaussian CUDA 扩展环境；本次使用的 Python 3.13 PyTorch 为 CPU 版，热核函数直接在 Warp CUDA 上运行。数值测试实际执行 Warp 0.10.1 CUDA 核函数和机械步骤；Gaussian 序列化与裁剪测试仅替代未调用的 KNN 导入，不执行图像渲染。

本机示例实测：576 个粒子、100 步、0.2 秒后，热半区均温 72.9377，冷半区均温 27.0624，热量相对误差 1.32×10⁻⁶，位置变化 0。尖锐冷热分界的仿射插值出现约 1.14 度的轻微超调。数据位于 `output/thermal_demo`。
