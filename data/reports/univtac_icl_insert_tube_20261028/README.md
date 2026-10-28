# UniVTAC insert_tube：冻结 Qwen ICL + 触觉 jam guard（2026-10-28）

> 本文件是验收报告。结果表由 `summarize.py` 生成（逐 seed 明细在 `summary.json`）；提示准确率与是否照做由 `hint_accuracy.py` 计算。
> 结论：带方向提示的 visuotactile jam guard（G1）成功 5/10。直接控制 visuotactile（v5）是 0/10，去掉提示的 guard（G2）也是 0/10。两组对比 Fisher 双侧 p≈0.03。
> 本次全程用冻结的本地 Qwen（qwen3.8-flash-next-iq3_s），没有微调。

## 一、设置

- 任务：官方 insert_tube。pre_move 抓住管子，送到 30° 斜孔口，并附加 1–4 mm 隐藏偏移（官方 `random_noise`，孔坐标系）。成功条件：插深 >3 cm、横向误差 <5 mm、轴线夹角约 15° 以内、手内滑移 <3 cm。手内滑移 >3 cm 时官方提前结束。
- 示范（TRAIN）：官方 HDF5 ep37，`data/datasets/univtac_icl/insert_tube_ep37_h8_v2`。示范过程是：推约 2 cm 后卡住，侧移约 5 mm、上抬约 3 mm 重新对准，再推约 4 cm。
- 配置 `prometheus/configs/univtac_icl_insert_tube.yaml`。评测 seed 1000001–1000010，各组相同。
- 旧专家记录（force=False，评测时序）6/6 成功，说明该任务在评测动力学下可解（lift_can 不满足这一点）。

## 二、v5 直接控制（基线）

| 组 | 成功 | 官方提前结束 | 成功：y>0 / y<0 |
|---|---|---|---|
| zero_shot | 0/10 | 6 | 0/4 · 0/6 |
| visual | 3/10 | 7 | 3/4 · 0/6 |
| visuotactile | 0/10 | **10** | 0/4 · 0/6 |

诊断：

- visual 的 3 次成功全部落在 y>0 的 seed 上，y<0 的 6 个全败。原因是模型照抄了示范 ep37 的侧移方向（ep37 的偏移 y=+2.5 mm），碰巧对上，并不是看懂了偏移。
- visuotactile 10/10 都是推得太深才检查，管子在手里滑出 3 cm，触发官方提前结束。卡住时 press_depth 几乎不变，提供不了"卡住了"或"往哪边偏"的信息。

## 三、探针：触觉标记点剪切里有卡住和方向信息

`jam_probe.py` 是诊断用的 Policy，可以读特权量，但特权量只作标定标签。运行于开发 seed 2000101–2000106，记录为 `insert_tube_jamprobe3_20261028`。

- 直推时，孔口倒角会把管子导入约 15 mm，然后卡住。
- 特征取 marker 平均位移，相对于第一次策略观测时的场。当任一侧轴向剪切 ≥6 px 时：
  - lateral 均值的符号 = 隐藏 x 偏移的符号（6/6）；
  - |右轴向|−|左轴向| 的符号 = y 偏移符号的相反数（6/6，其中 y>0 只有 1 例）。
- 专家纠正方法：手往偏移的反方向挪 1–2.5 mm。

## 四、jam guard 设计（`policy.control=guard`）

代码在 `prometheus/policy/univtac_icl/jam_guard.py`，测试在 `tests/test_univtac_icl.py`（16 项通过）。

- System 2 照旧输出开环行 `[dx,dy,dz,rx,ry,rz,finger]`。每行先用 curobo 规划，再逐控制步执行。
- 某行执行中，若任一侧轴向剪切 ≥6 px，且比本行开始时升高 ≥1 px，就停在这一行，本次调用以 `s1_jam` 结束。停下时把手臂保持在实测关节位置、速度置零。第一版没有这一步，停下后手臂会沿推的方向继续漂。
- 有触觉的条件，回执附 `marker_shear_px`。`policy.jam_hint=true` 时，卡住的回执再附 `offset_hint`，内容是按第三节符号映射算出的 `shift_hand_x/y` 和"通常 1–3 mm"。
- 这个映射是针对 insert_tube 在开发 seed 上标定的，换任务需要重新标定。

## 五、正式对照结果

seed 1000001–1000010。G2 的 1000006 因仿真慢启动，在调用模型之前就被结束，用 `_infra1` 补跑，补跑结果计入表中。

| 组 | 成功 | 官方提前结束 | 成功：y>0 / y<0 | 平均调用 | jam 停止 | token/调用 |
|---|---|---|---|---|---|---|
| v5 / visuotactile | 0/10 | 10 | 0/4 · 0/6 | 3.7 | — | 254 |
| v5 / visual | 3/10 | 7 | 3/4 · 0/6 | 2.9 | — | 263 |
| **G1 guard+提示 / visuotactile** | **5/10** | 1 | **3/4 · 2/6** | 8.2 | 44 | 213 |
| G2 guard 无提示 / visuotactile | 0/10 | 5 | 0/4 · 0/6 | 9.9 | 44 | 205 |
| G3 guard+提示 / visual | 2/10 | 3 | 2/4 · 0/6 | 10.4 | 61 | 250 |

G1 成功的 seed：1000002/05/06/07/08。G3 成功的 seed：1000007/10。

注意：G1、G2、v5 zero_shot 的 1000001 都是官方 pre_move 规划失败（`pre_move_planning_failure_before_policy`），策略一次都没被调用。官方把它记为 failed，属于策略结果，所以没有补跑。如果排除这个 seed，G1 是 5/9。

### 读法

1. **只停不给方向没有用**（G2 0/10）。guard 确实挡住了大部分深推：官方提前结束从 v5 的 10 次降到 5 次。但模型只知道"卡住了"，不知道该往哪边挪，在 13 次调用的预算内反复试错。
2. **方向提示是增益来源**（G1 5/10 对 G2 0/10）。G1 也是第一次在 y<0 的 seed 上成功（2/6），说明不再只是照抄示范的侧移方向。
3. **提示准确率（更正版，`hint_residual.py`）**：最初的统计拿每次卡住时的提示去对比最初的隐藏偏移，后续卡住几乎是随机（x 63/121、y 62/121）。但模型挪手之后剩余偏移已经变了，这样比不对。改为按卡住当时的剩余偏移比较（隐藏偏移 + 手的位移，投影到孔坐标系，未计管子在手内的倾斜）：

   | | x 正确 | y 正确 |
   |---|---|---|
   | 第一次卡住（28 次） | 28/28 | 25/28 |
   | 之后的卡住（121 次） | 89/121 | 87/121 |

   后续提示比第一次弱，但不是随机。真正的问题是步长：G1 每次 y 方向挪约 4 mm（中位数），剩余偏移只有约 2.8 mm，模型在孔两边来回过冲。

4. **模型照做了提示**：G1 下一次调用的 dx/dy 方向与提示一致的比例是 x 34/43、y 40/43；G3 是 x 54/58、y 57/58。G3 的第一次提示 x 10/10、y 9/10 都对，模型也照做了，结果却只有 2/10。视觉组同样在孔两边来回过冲，6/10 耗尽调用预算。G1 失败的 5 个里，3 个是耗尽预算，1 个官方提前结束，1 个是 pre_move 失败。
5. 显著性：G1 对 v5 visuotactile、G1 对 G2 都是 5/10 对 0/10，Fisher 双侧 p≈0.03。G1 对 v5 visual（3/10）不显著。每组只有 10 个 seed，需要更多 seed 才能下结论。

## 六、是否需要微调

目前没有证据表明必须微调。增益来自给冻结模型一个可读的方向信号，模型照做的比例很高。现在的瓶颈在信号和步长：提示只有方向、没有幅度，模型挪得太多。

## 后续（同日）

多次卡住标定（`probe_multi.py`、`analyze_multi.py`、`multiprobe2_analysis.txt`）显示 asymmetry_px 能定量估出手的 y 偏差：20/20 次方向全对，残差约 1 mm。由此做了 guard v2（`jam_hint=estimate`，加上 v3 示范，示范里的卡住块写成 guard 回执），开发 seed 冒烟 4/4 成功。30 seed 正式对照见 `README_v2_scale30.md`：v2est 24/30、g1rep 15/30、纯视觉 3/30。下面这一节是写 v2 之前的原始建议。

## 下一步建议（原始）

建议先改信号，不动模型。每一项都是新代码，需要新 run_id，并先在开发 seed 上验证：

1. **每次卡住重新取基线**：以本行开始时的场为基线，只看这一行内的剪切变化；或者只回报第一次卡住的提示。需要先在开发 seed 上用探针重新标定"后续卡住"的符号是否可靠。
2. 给提示加幅度：剪切越大，建议挪得越多。现在只给方向和"1–3 mm"。
3. 扩大评测 seed（例如 20–30 个），确认 G1 的提升。

## 七、run 记录

| run | 说明 |
|---|---|
| `insert_tube_smoke_vt_20261028` | 冒烟（dev 2000001） |
| `insert_tube_v5_{zero_shot,visual,visuotactile}_20261028`（+ `zero_shot_infra1`） | 基线 |
| `insert_tube_jamprobe_20261028`、`jamprobe2` | 启动器错误，没有启动仿真（有 operator_stop.json） |
| `insert_tube_jamprobe3_20261028` | 标定探针（dev 2000101–106） |
| `insert_tube_guard_smoke_20261028` | guard v1（没有 hold_arm），作废并保留 |
| `insert_tube_guard2_smoke_20261028` | guard v2 冒烟（dev 2000203/204，成功 1/2） |
| `insert_tube_guardG1_visuotactile_20261028` | **正式 G1** |
| `insert_tube_guardG2_visuotactile_20261028`（+ `_infra1` 1000006） | **正式 G2** |
| `insert_tube_guardG3_visual_20261028` | **正式 G3** |
