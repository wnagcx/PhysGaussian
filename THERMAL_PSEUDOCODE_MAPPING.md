# 修改后的热传导伪代码与 PhysGaussian 实现对照

本文以你提供的修改版伪代码为顺序，说明每个公式中的变量在源码中如何存储、公式在哪个函数中计算，以及这些函数如何接入原有 MPM 时间步。

新增热状态主要位于 `mpm_solver_warp/thermal_utils.py` 和 `MPMStateStruct`。整体时间步仍由 `MPM_Simulator_WARP.p2g2p()` 调度。温度作为被动属性参与传递和扩散，材料类型、应力计算与形变继续使用场景配置中的机械模型。

下方链接指向当前工作区的实际文件和核对过的行号；以后修改源码时，函数名比行号更适合作为长期定位依据。

## 1. 伪代码与函数总览

| 你的伪代码 | 对应函数或位置 | 作用与实现来源 |
|---|---|---|
| `while time_step < total_steps` | [gs_simulation.py 的帧与子步循环](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/gs_simulation.py:358)、[p2g2p()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_solver_warp.py:451) | 原有循环中调用新增热求解步骤 |
| `Transfer Mass & Momentum to Grid via APIC` | [p2g_apic_with_stress()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_utils.py:320) | 原有机械质量、动量传递 |
| $m_iT_i\leftarrow\sum_p w_{ip}m_p[T_p+(\mathbf x_i-\mathbf x_p)\cdot\nabla T_p]$ | [thermal_p2g()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:39)、[normalize_thermal_grid()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:64) | 新增：先累积质量与温度加权量，再归一化 |
| `Update Grid Velocities` | [grid_normalization_and_gravity()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_utils.py:380)，以及原有阻尼和机械边界函数 | 原有网格速度更新 |
| `Update Thermal Boundary` | [ThermalSolver._boundaries()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:255)、[thermal_boundary()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:82) | 新增：恒温区域与升温源 |
| $T_i\leftarrow\mathrm{LBM\_Update}(T_i)$ | [ThermalSolver.advance()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:267)、[lbm_collide()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:113)、[lbm_stream()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:124) | 新增：D3Q7 热扩散的碰撞、传播及自动子步 |
| `Interpolate Velocity from Grid` | [g2p()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_utils.py:394) | 原有网格到粒子的速度插值 |
| `Update Particle Mechanical Properties` | 原有 `g2p()` 与 [compute_stress_from_F_trial()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_utils.py:445) | 更新速度、位置、仿射速度矩阵、试算形变梯度与应力 |
| $T_p\leftarrow\sum_i w_{ip}T_i$ | [thermal_g2p()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:148) | 新增：网格温度插值到粒子 |
| $\nabla T_p\leftarrow\sum_iT_i(\nabla w_{ip})^\top$ | 同一个 `thermal_g2p()` | 新增：网格温度重建粒子温度梯度 |

热状态的核心数据流是：

```mermaid
flowchart LR
    A[粒子温度和梯度] --> B[thermal_p2g]
    B --> C[节点质量和温度加权量]
    C --> D[normalize_thermal_grid]
    D --> E[更新热边界]
    E --> F[lbm_collide 和 lbm_stream]
    F --> G[thermal_g2p]
    G --> H[更新粒子温度和梯度]
    H --> A
```

## 2. 符号与代码变量

| 数学符号 | 代码变量 | 说明 |
|---|---|---|
| $p$ | `p = wp.tid()` | 当前 GPU 线程处理的粒子编号 |
| $i$ | `x, y, z` | 网格节点的三维整数下标；不是单个整数变量 |
| $\mathbf x_p$ | `state.particle_x[p]` | MPM 空间中的粒子位置 |
| $\mathbf x_i$ | `wp.vec3(x, y, z) * model.dx` | 网格节点的空间位置 |
| $m_p$ | `state.particle_mass[p]` | 原有粒子质量，来自密度乘体积 |
| $m_i$ | `grid.mass[x, y, z]` | 热传递使用的节点质量 |
| $T_p$ | `state.particle_temperature[p]` | 每个 MPM 粒子的标量温度 |
| $\nabla T_p$ | `state.particle_temperature_gradient[p]` | 三维温度梯度，单位为温度/长度 |
| $T_i$ | `grid.temperature[x, y, z]` | 网格节点温度 |
| $m_iT_i$ | P2G 阶段的 `grid.energy[x, y, z]` | 温度的质量加权累积量 |
| $w_{ip}$ | `weight` | 粒子与节点之间的 B 样条插值权重 |
| $\nabla w_{ip}$ | `dweight` | 插值权重对粒子位置的梯度 |
| $\Delta x$ | `model.dx` | 网格间距，`grid_lim / n_grid` |
| $1/\Delta x$ | `model.inv_dx` | 网格间距倒数 |
| $\Delta t$ | `p2g2p()` 的 `dt` | 一个机械时间步 |
| $h$ | `ThermalSolver.advance()` 的 `h` | 自动细分后的热时间步 |
| $\alpha$ | `self.config["diffusivity"]` | 热扩散率 |
| $\mathbf F_p^E$、$\boldsymbol\tau_p$ | `particle_F`、`particle_stress` | 原有机械形变梯度与 Kirchhoff 应力 |

**热扩散率不是 `MPMModelStruct.alpha`。** 原有 `model.alpha` 用于砂材料的摩擦模型；本次热扩散率从 `thermal.diffusivity` 读取，两者用途不同。

`ThermalGrid.energy` 在名字上叫 energy，但其数值是 $m_iT_i$，不是直接以焦耳计量的热量。在比热 $c$ 相同的假设下，可用 $c\,m_iT_i$ 表示相对于统一温度参考的热能。该数组存放的是 P2G 累积量；LBM 更新后的温度以 `grid.temperature` 为准，`grid.energy` 不会在每个传播步骤后同步刷新。

## 3. 为每个 Gaussian 和粒子保存温度

MPM 状态中的两个新增字段定义在 [warp_utils.py](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/warp_utils.py:62)：

```python
particle_temperature: wp.array(dtype=float)
particle_temperature_gradient: wp.array(dtype=wp.vec3)
```

它们分别表示 $T_p$ 和 $\nabla T_p$，长度均为 `n_particles`。[initialize()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_solver_warp.py:115) 将默认温度设为 20，默认梯度设为零。配置的统一初温和初始热点由 [configure_thermal()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_solver_warp.py:409) 设置。

Gaussian 模型也增加了 `_temperature` 和 `_temperature_gradient`，形状分别为 `(N, 1)` 和 `(N, 3)`。访问接口是 [get_temperature、get_temperature_gradient 和 set_temperature()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/gaussian-splatting/scene/gaussian_model.py:114)。它们不是可训练的 `nn.Parameter`，不会进入 Gaussian 的优化器。

[gs_simulation.py 的初始化段](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/gs_simulation.py:269) 将 Gaussian 温度、梯度传入 MPM。经过透明度筛选和 `sim_area` 筛选后，`gaussian_indices` 记录参与仿真的原始 Gaussian 编号；MPM 数组的前 K 个粒子对应这些 Gaussian，后面的粒子是内部填充粒子。填充粒子同样具有温度与梯度。

## 4. 温度 P2G 公式如何实现

你的公式为：

$$
m_iT_i\leftarrow\sum_p w_{ip}m_p
\left[T_p+(\mathbf x_i-\mathbf x_p)\cdot\nabla T_p\right].
$$

实现时分成三个量：

$$
\delta\mathbf x_{ip}=\mathbf x_i-\mathbf x_p,
\qquad
\widetilde T_{ip}=T_p+\delta\mathbf x_{ip}\cdot\nabla T_p,
\qquad
a_{ip}=w_{ip}m_p.
$$

[thermal_p2g() 的累积代码](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:54) 对应如下：

```python
weight = w[0, a] * w[1, b] * w[2, c]
dpos = (wp.vec3(wp.float(a), wp.float(b), wp.float(c)) - fx) * model.dx
mass = weight * state.particle_mass[p]
temperature = state.particle_temperature[p] + wp.dot(
    dpos, state.particle_temperature_gradient[p])
wp.atomic_add(grid.mass, x, y, z, mass)
wp.atomic_add(grid.energy, x, y, z, mass * temperature)
```

这里 `dpos` 就是 $\delta\mathbf x_{ip}$，`temperature` 是外推至节点位置的 $\widetilde T_{ip}$，局部变量 `mass` 是 $a_{ip}$，不是完整节点质量。两次原子累积分别得到：

$$
m_i=\sum_p w_{ip}m_p,
\qquad
E_i=\sum_p w_{ip}m_p\widetilde T_{ip}.
$$

随后 [normalize_thermal_grid()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:64) 计算：

$$
T_i=\frac{E_i}{m_i},\qquad m_i>10^{-15}.
$$

```python
grid.temperature[x, y, z] = grid.energy[x, y, z] / grid.mass[x, y, z]
```

伪代码以网格节点为外层循环、对粒子求和。Warp 实现以粒子为并行线程，每个粒子访问周围 27 个节点，通过 `wp.atomic_add()` 完成同一个求和。由于多个粒子会写入同一节点，原子累加保证这些贡献不会被覆盖。

例如，两个质量均为 1、对同一节点权重均为 0.5 的粒子，温度分别为 80 与 20，梯度为零，则该节点的质量为 1、温度加权量为 50，最终温度为 50。

热求解器使用独立的 `grid.mass`，而机械 APIC 使用 `state.grid_m`。原有机械传递跳过 `particle_selection != 0` 的粒子，热传递仍包含它们，使机械固定的粒子也能交换热量。

## 5. 权重和温度梯度项从哪里来

[thermal_weights()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:24) 使用二次 B 样条。对于每个轴的局部坐标 $f$，三个权重为：

$$
B_0(f)=\tfrac12(1.5-f)^2,
\quad B_1(f)=0.75-(f-1)^2,
\quad B_2(f)=\tfrac12(f-0.5)^2.
$$

三维节点权重是三个轴上权重的乘积：

$$
w_{ip}=B_a(f_x)B_b(f_y)B_c(f_z),
\qquad a,b,c\in\{0,1,2\}.
$$

因此三个长度为 3 的循环访问的是 $3\times3\times3=27$ 个节点。

P2G 公式的点积是对粒子温度场做一阶外推。比如某粒子的温度为 50，x 方向梯度为 10 温度单位/长度单位，节点在粒子右侧 0.01 个长度单位，则该粒子贡献的外推温度为 $50+10\times0.01=50.1$。该梯度从上一步 G2P 得到，并供下一步 P2G 使用。

## 6. `Update Thermal Boundary` 对应哪些代码

初始热点由 [set_particle_temperature_region()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:72) 设置，只用于初始化。时间步内的持续加热由 `_boundaries()` 调用 `thermal_boundary()` 完成，两种边界对应：

$$
\text{恒温区域：}\quad T_i\leftarrow T_{\mathrm{source}},
$$

$$
\text{升温源：}\quad T_i\leftarrow T_i+S_i h.
$$

[thermal_boundary() 的赋值](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:89) 为：

```python
if fixed == 1:
    grid.temperature[x, y, z] = value
else:
    grid.temperature[x, y, z] = grid.temperature[x, y, z] + value
```

`fixed_temperature.temperature` 直接作为 `value`；`heat_source.rate` 乘当前热子步 `h` 后作为 `value`。`rate` 的单位是温度/秒，不是瓦特。

热边界只作用于有质量且位于指定长方体内的节点。`point` 是中心，`size` 是半尺寸，均位于 MPM 空间。区域固定在空间中，运动的物体经过它时会被加热。边界在 `[start_time, end_time)` 内生效。

每个热子步先叠加升温源，再施加恒温边界，然后进行扩散；扩散后重新施加恒温边界，维持指定节点温度。恒温源与升温源重叠时，恒温条件优先。

## 7. `LBM_Update` 在本次实现中的展开

你的伪代码用一行 $T_i\leftarrow\mathrm{LBM\_Update}(T_i)$ 概括网格演化。本次将其具体实现为 D3Q7 BGK 热扩散，采用松弛时间 $\tau_{\mathrm{LBM}}=1$。这是该行伪代码的一种离散实现选择；不代表已经复现 MeGAS 的全部流动求解细节。

### 7.1 七个方向与分布函数

[lattice_direction()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:95) 定义六个轴向邻居，加上静止方向：

$$
\mathbf e_q\in\{(0,0,0),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)\}.
$$

`grid.populations[x, y, z, q]` 保存节点在第 q 个方向上的分布函数 $f_{i,q}$，维度为 `(n_grid, n_grid, n_grid, 7)`。这里的 $\mathbf e_q$ 是整数节点偏移。

### 7.2 碰撞：`lbm_collide()`

定义无量纲扩散数：

$$
r=\frac{\alpha h}{\Delta x^2}.
$$

BGK 碰撞写作：

$$
f^*_{i,q}=f_{i,q}-\frac{f_{i,q}-f^{\mathrm{eq}}_{i,q}}{\tau_{\mathrm{LBM}}}.
$$

当 $\tau_{\mathrm{LBM}}=1$ 时，碰撞后的分布直接等于平衡分布。本实现使用：

$$
f^*_{i,0}=(1-6r)m_iT_i,
\qquad f^*_{i,q}=r\,m_iT_i\quad(q=1,\ldots,6).
$$

[对应代码](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:113)：

```python
energy = grid.mass[x, y, z] * grid.temperature[x, y, z]
grid.populations[x, y, z, 0] = (1.0 - 6.0 * r) * energy
for q in range(1, 7):
    grid.populations[x, y, z, q] = r * energy
```

每个热子步重新根据当前温度构建这些分布，不跨机械时间步保存非平衡分布。权重随实际 `h` 缩放，使热扩散量与时间步和 `diffusivity` 对应。

### 7.3 传播：`lbm_stream()`

对于节点 i 和有效邻居 j，代码定义对称链路质量：

$$
M_{ij}=\min(m_i,m_j).
$$

[对应的传播与反弹代码](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:140)：

```python
link_mass = wp.min(mass, neighbor_mass)
incoming = (grid.populations[sx, sy, sz, q] * link_mass / neighbor_mass
            + incoming * (1.0 - link_mass / mass))
```

第一项是从邻居传入的部分，第二项是本节点未传出的反弹部分。综合静止分布和六个方向，可将本实现的更新等价写成：

$$
T_i^{n+1}=T_i^n+r\sum_{j\in\mathcal N(i)}
\frac{M_{ij}}{m_i}(T_j^n-T_i^n),
$$

其中 $\mathcal N(i)$ 只包含域内、有质量的轴向邻居。相邻两节点交换的温度加权量分别是：

$$
\Delta E_i=rM_{ij}(T_j-T_i),
\qquad \Delta E_j=-\Delta E_i.
$$

这使链路交换在没有外部热源时保持 $\sum_i m_iT_i$ 守恒。缺少邻居、邻居没有质量或越出计算域时，分布全部反弹，构成绝热边界。

当节点质量相同时，上式变成七点扩散更新：

$$
T_i^{n+1}=T_i^n+\frac{\alpha h}{\Delta x^2}
\left(\sum_{j\in\mathcal N(i)}T_j^n-6T_i^n\right)
$$

这里假设六个邻居都有效。它对应网格上的纯热扩散项。质量不同时，本实现使用上述质量相关的链路导通率。

例如，中心节点 80、六个邻居均为 20、质量相同且 $r=0.01$，一步后中心变为 76.4，每个邻居从这条链路得到 0.6 的升温。

### 7.4 自动细分热时间步

`advance()` 保证 $r\le0.15<1/6$，使碰撞权重非负。对于两个热边界启停时刻之间的一段时间 $\Delta t_s$，其代码相当于：

$$
N_{\mathrm{sub}}=\max\left(1,\left\lceil
\frac{\alpha\Delta t_s}{0.15\Delta x^2}\right\rceil\right),
\qquad h=\frac{\Delta t_s}{N_{\mathrm{sub}}}.
$$

代码还在热源的开始、结束时刻切分机械时间步，使热源按实际启停时长加热。这些处理具体化了伪代码中的 `LBM_Update`，不增加温度触发的材料切换。

## 8. G2P 温度公式如何实现

你的温度公式为：

$$
T_p\leftarrow\sum_iw_{ip}T_i.
$$

[thermal_g2p() 的插值累积](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:167) 为：

```python
weight = w[0, a] * w[1, b] * w[2, c]
value = grid.temperature[x, y, z]
temperature = temperature + weight * value
weight_sum = weight_sum + weight
```

`value` 是 $T_i$，`temperature` 累积 $\sum_iw_{ip}T_i$。在完整的 27 节点模板内，权重和 $W_p=\sum_iw_{ip}=1$，因此正好得到伪代码的温度插值。

对于被域边界截断或过滤掉无质量节点的模板，代码使用归一化形式：

$$
T_p=\frac{\sum_iw_{ip}T_i}{W_p}.
$$

这样可避免因为少了一些有效权重，使均匀温度被人为降低。此处是对完整模板公式的边界修正；后面的梯度也必须使用同一归一化。

## 9. G2P 温度梯度公式如何实现

你的梯度公式为：

$$
\nabla T_p\leftarrow\sum_iT_i(\nabla w_{ip})^\top.
$$

代码将梯度保存为 `wp.vec3`，用三个分量表达该梯度，不另外存储行矩阵或列矩阵。

一维权重导数为：

$$
B'_0(f)=f-1.5,
\quad B'_1(f)=-2(f-1),
\quad B'_2(f)=f-0.5.
$$

乘以另外两个轴的权重，并除以网格间距，即得到对空间坐标的三维梯度：

$$
\nabla_{\mathbf x_p}w_{ip}=\frac1{\Delta x}
\begin{bmatrix}
B'_a(f_x)B_b(f_y)B_c(f_z)\\
B_a(f_x)B'_b(f_y)B_c(f_z)\\
B_a(f_x)B_b(f_y)B'_c(f_z)
\end{bmatrix}.
$$

[代码中的 `dweight`](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:168)：

```python
dweight = wp.vec3(dw[0, a] * w[1, b] * w[2, c],
                  w[0, a] * dw[1, b] * w[2, c],
                  w[0, a] * w[1, b] * dw[2, c]) * model.inv_dx
gradient = gradient + value * dweight
gradient_sum = gradient_sum + dweight
```

`gradient` 累积 $\sum_iT_i\nabla w_{ip}$，`gradient_sum` 累积 $\nabla W_p$。对上一节的归一化温度使用商法则，得到：

$$
\nabla T_p=
\frac{\sum_iT_i\nabla w_{ip}-T_p\sum_i\nabla w_{ip}}{W_p}.
$$

[最终写回](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/thermal_utils.py:178)：

```python
temperature = temperature / weight_sum
state.particle_temperature[p] = temperature
state.particle_temperature_gradient[p] = (gradient - temperature * gradient_sum) / weight_sum
```

在完整模板中，$W_p=1$、$\sum_i\nabla w_{ip}=0$，这个式子就退化为你的伪代码公式。新的梯度供下一时间步的仿射温度 P2G 使用。

## 10. 与原有机械步骤的连接和执行顺序

原有 [p2g_apic_with_stress()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_utils.py:320) 传递质量与动量，可对应为：

$$
m_i=\sum_p w_{ip}m_p,
$$

$$
\mathbf q_i=\sum_p\left[
w_{ip}m_p\bigl(\mathbf v_p+\mathbf C_p(\mathbf x_i-\mathbf x_p)\bigr)
-\Delta t\,V_p\boldsymbol\tau_p\nabla w_{ip}
\right].
$$

`grid_normalization_and_gravity()` 将网格动量除以质量并添加重力，随后执行原有网格阻尼和机械边界条件。原有 `g2p()` 插值速度并更新粒子位置、仿射速度矩阵和试算形变梯度：

$$
\mathbf v_p^{n+1}=\sum_iw_{ip}\mathbf v_i,
\qquad \mathbf x_p^{n+1}=\mathbf x_p^n+\Delta t\,\mathbf v_p^{n+1},
$$

$$
\mathbf F_p^{\mathrm{trial}}=
\bigl(\mathbf I+\Delta t\,\nabla\mathbf v_p\bigr)\mathbf F_p^E.
$$

试算形变梯度在下一步的 `compute_stress_from_F_trial()` 中进入既有材料模型，更新弹性形变与应力。这些函数没有使用温度来判断材料类型。

[p2g2p() 中的实际顺序](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_solver_warp.py:451) 为：

```text
清空机械网格，执行原有粒子速度/外力条件
    ↓
compute_stress_from_F_trial       原有应力计算
    ↓
p2g_apic_with_stress              原有机械 P2G
    ↓
thermal_solver.p2g               新增热 P2G 和温度归一化
    ↓
grid_normalization_and_gravity    原有速度、重力、阻尼和机械边界更新
    ↓
thermal_solver.advance           热边界与 LBM
    ↓
thermal_solver.g2p               新增温度及梯度插值
    ↓
g2p                              原有速度、位置、C、F_trial 更新
    ↓
self.time += dt
```

调用热 P2G 的位置是 [mpm_solver_warp.py:514](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_solver_warp.py:514)，调用热演化与热 G2P 的位置是 [mpm_solver_warp.py:555](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/mpm_solver_warp.py:555)。

**温度 G2P 在机械 G2P 前执行。** 这是因为原有机械 `g2p()` 会直接改写 `particle_x`。温度 P2G 与 G2P 必须使用同一时刻的位置来计算权重，否则会把旧网格温度插值到已经移动的粒子位置上。伪代码将两类更新放在一个粒子循环内，源码则通过分开的核函数保持这个位置关系。

本次 LBM 网格步骤计算纯热传导；粒子的移动负责携带温度。没有另外新增 Navier–Stokes、空气对流、热浮力、热膨胀或温度触发的相变模型。

## 11. MPM 温度如何同步到 Gaussian 和输出文件

每帧子步结束后，[synchronize_gaussian_temperature()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/utils/thermal_render.py:17) 将参与仿真的粒子温度按 `gaussian_indices` 写回原始 Gaussian 编号：

```python
gaussians._temperature[gaussian_indices, 0] = temperature
gaussians._temperature_gradient[gaussian_indices] = gradient
```

温度是标量，旋转和平移不会改变它；梯度依赖空间坐标。若预处理变换写作：

$$
\mathbf x_{\mathrm{MPM}}=s\mathbf R\mathbf x_{\mathrm{world}}+\mathbf b,
$$

则有：

$$
\nabla_{\mathrm{MPM}}T=\frac1s\mathbf R\nabla_{\mathrm{world}}T,
\qquad
\nabla_{\mathrm{world}}T=s\mathbf R^\top\nabla_{\mathrm{MPM}}T.
$$

源码的 `scale_origin` 对应 s，旋转矩阵列表对应组成总旋转的各次旋转；同步函数按逆序处理它们并乘缩放系数。

| 接口 | 与温度属性的关系 |
|---|---|
| [GaussianModel.save_ply() / load_ply()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/gaussian-splatting/scene/gaussian_model.py:228) | 保存、读取 `temperature` 和三个梯度字段；旧 PLY 默认温度 20、梯度 0 |
| [GaussianModel.capture() / restore()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/gaussian-splatting/scene/gaussian_model.py:63) | 温度进入训练检查点，并兼容旧格式 |
| [save_temperature_frame()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/utils/thermal_render.py:28) | 保存全部 Gaussian 温度以及完整 MPM 温度、梯度、位置和映射编号 |
| [save_data_at_frame()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/mpm_solver_warp/engine_utils.py:9) | 粒子 PLY/H5 增加温度和梯度 |
| [temperature_to_rgb()](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/utils/thermal_render.py:7) | 温度转换为显示颜色；限幅只用于显示，不修改物理温度 |

这些读写与显示接口是围绕伪代码新增的配套功能，不参与 `LBM_Update` 的数值计算。

## 12. 验证证据与实现边界

下列测试已经在前次实现验证中通过，可用于核查本文中的对应关系：

| 验证项目 | 测试位置 | 主要核查内容 |
|---|---|---|
| 均匀温度保持 | [test_constant_temperature_stays_constant](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_thermal.py:56) | 温度传递与绝热扩散不会破坏常量场 |
| 线性温度场重建 | [test_affine_transfer_preserves_linear_field](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_thermal.py:64) | P2G 一阶梯度项和 G2P 梯度公式 |
| D3Q7 单点扩散 | [test_d3q7_impulse_and_diffusivity_scaling](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_thermal.py:76) | 实际扩散量随 α 和时间步缩放 |
| 与解析热方程对照 | [test_neumann_cosine_matches_heat_equation](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_thermal.py:120) | 网格扩散与绝热余弦模式衰减吻合 |
| 质量不均匀时的守恒 | [test_hot_to_cold_and_grid_energy_conservation](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_thermal.py:105) | 对称链路交换保持温度质量加权量守恒 |
| 热源启停与恒温 | [test_heat_source_schedule_and_fixed_boundary](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_thermal.py:131) | 热边界更新和时间区间处理 |
| 全套热传递的守恒 | [test_particle_heat_conservation_over_transfers](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_thermal.py:156) | P2G、LBM、G2P 的连续循环 |
| 高温下机械模型一致 | [test_temperature_does_not_change_mechanics_or_material](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_thermal.py:144) | 温度不会触发材料切换或改变该测试的机械状态 |
| Gaussian 温度持久化 | [test_gaussian_temperature.py](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/tests/test_gaussian_temperature.py:58) | PLY、裁剪、克隆和坐标映射 |

前次共 15 项测试通过，热数值核函数在 Warp CUDA 上执行；Gaussian 属性测试不执行实际图像渲染。

[冷热块体示例报告](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/output/thermal_demo/report.json) 记录了 576 个机械固定粒子模拟 100 步、总计 0.2 秒的结果：热半区均温由 80 降至 72.9377，冷半区由 20 升至 27.0624，温度质量加权量的相对误差约为 1.32×10⁻⁶。

仿射 P2G/G2P 仍有插值误差。在尖锐冷热分界附近，示例出现约 1.14 度的超调；这两个温度传递公式没有增加斜率限幅器。即使 `diffusivity=0`，插值也可能平滑不连续温度；暂停全部温度更新需要 `thermal.enabled=false`。上述守恒和公式直接对应完整、域内的粒子插值模板；粒子模板被域边界截断时，归一化修正不保证相同的全局守恒性质。

热源配置与运行参数另见 [THERMAL.md](C:/Users/Lenovo/Desktop/physGS/PhysGaussian/THERMAL.md)。
