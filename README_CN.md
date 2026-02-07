<p align="right"><a href="./README.md">English</a> | 简体中文</p>

<div align="center">

# Drive-KD：面向自动驾驶 VLM 的多教师蒸馏框架

**连伟童<sup>1,\*</sup>, 唐泽聪<sup>1,\*</sup>, 李灏然<sup>1,\*</sup>, 高天健<sup>1</sup>, 王翌飞<sup>1</sup>, 王紫旭<sup>1</sup>, 孟令祎<sup>1</sup>, 茹腾驹<sup>1</sup>, 崔哲珺<sup>1</sup>, 朱奕辰<sup>1</sup>, 曹航硕<sup>1</sup>, 康祺<sup>1</sup>, 陈天行<sup>2</sup>, 秦玉森<sup>3</sup>, 王开炫<sup>2</sup>, 张宇<sup>1,†</sup>**

<sup>1</sup>浙江大学（杭州） &nbsp;&nbsp; <sup>2</sup>香港大学（香港） &nbsp;&nbsp; <sup>3</sup>地瓜机器人（深圳）  
<sup>*</sup>共同一作 &nbsp;&nbsp; <sup>†</sup>通讯作者

[![Paper](https://img.shields.io/badge/arXiv-2601.21288-b31b1b.svg)](https://arxiv.org/abs/2601.21288)

<img src="docs/figs/pipeline.jpg" width="100%"/>

</div>

Drive-KD 是一个面向**自动驾驶视觉语言模型（VLM）**的**多教师知识蒸馏**框架。  
我们将驾驶能力分解为顺序三元组 **感知 → 推理 → 规划**，通过**分层注意力蒸馏**进行能力迁移，并提出 **非对称梯度投影（AGP, Asymmetric Gradient Projection）** 缓解跨能力梯度冲突。

---

## ✨ 亮点概览

- **能力分解：** 按人类驾驶思维，将任务拆为顺序的 **perception–reasoning–planning**。
- **蒸馏信号选择：** 通过四个预实验——(a) 层级蒸馏对齐分析（Layer-wise distillation alignment），(b) 能力维度的组内一致性（Capability-wise intra-group similarity），(c) 隐状态与注意力图的层级离散度分析（Layer-wise dispersion of hidden states & attention maps），以及 (d) 位置归一化的广义间隔分析（Position-normalized generalized margin）——我们探究了信号选择策略
- **单教师配方（按能力定制）：**
  - **感知：** 第 1 层 **text-to-vision** 注意力蒸馏。
  - **推理：** **中间层**注意力蒸馏 + **layer-group matching**。
  - **规划：** **倒数第二层** **text-to-vision** 注意力蒸馏。
- **多教师蒸馏：** 将三位能力专精教师统一到同一训练目标中，并使用按能力设定的教师混合矩阵。
- **AGP：** 两阶段梯度投影，降低跨能力目标互相“拉扯”的负面影响。
- **效率–性能：** 蒸馏后的 **InternVL3-1B** 达到 **~42× 更低显存**、**~11.4× 更高吞吐**，在 DriveBench 上整体优于同系列预训练 **InternVL3-78B**，并在规划维度超过 **GPT-5.1**。

---

## 🧩 方法概览

### 预实验（层选择 & 信号选择）

<p align="center">
  <img src="docs/figs/pre_study.jpg" alt="Pre-study" width="100%"/><br/>
  <em>InternVL3-8B 预实验总结：(a) 层级蒸馏对齐（相邻层与同层视觉-文本余弦相似度），(b) 能力维度的组内一致性，(c) 隐状态与注意力的层间离散度（1−cos），(d) 答案区段的位置归一化 generalized margin（对比 driving 与 general 数据，τ≈1.0）。</em>
</p>

### AGP（Asymmetric Gradient Projection）

<p align="center">
  <img src="docs/figs/AGP.jpg" alt="AGP" width="40%"/><br/>
  <em><b>AGP。</b> 第一阶段在每个能力内执行“主-从”的非对称投影并合并；第二阶段在能力之间进行随机顺序的两两投影，得到最终梯度方向。</em>
</p>

---

## 📊 结果（DriveBench）

### 表 1：能力分数 & 部署效率指标

<div align="center">

<table>
<thead>
<tr>
  <th rowspan="2">模型</th>
  <th colspan="4">能力分数 (%)</th>
  <th colspan="3">部署指标</th>
</tr>
<tr>
  <th>感知</th><th>推理</th><th>规划</th><th>平均</th>
  <th>显存 (GB)</th><th>速度 (tok/s)</th><th>首 token (s)</th>
</tr>
</thead>
<tbody>
<tr><td><b>GPT-5.1</b></td><td>45.56</td><td>41.02</td><td>51.94</td><td><b><u>46.17</u></b></td><td>-</td><td>-</td><td>-</td></tr>

<tr><td colspan="8"><b>InternVL3（预训练）</b></td></tr>
<tr><td>InternVL3-1B</td><td>33.26</td><td>20.96</td><td>22.36</td><td>25.53</td><td><b><u>4.1</u></b></td><td><b><u>45.7</u></b></td><td><b><u>0.45</u></b></td></tr>
<tr><td>InternVL3-2B</td><td>37.71</td><td>35.99</td><td>26.19</td><td>33.30</td><td>6.3</td><td>39.9</td><td>0.67</td></tr>
<tr><td>InternVL3-8B</td><td>40.05</td><td>41.15</td><td>32.77</td><td>37.99</td><td>18.3</td><td>32.6</td><td>1.58</td></tr>
<tr><td>InternVL3-14B</td><td>39.83</td><td>40.84</td><td>36.35</td><td>39.01</td><td>33.4</td><td>17.0</td><td>3.01</td></tr>
<tr><td>InternVL3-38B</td><td>34.27</td><td>38.48</td><td>40.65</td><td>37.80</td><td>87.0</td><td>7.3</td><td>9.84</td></tr>
<tr><td>InternVL3-78B</td><td>42.01</td><td><b><u>47.16</u></b></td><td>36.31</td><td>41.83</td><td>171.6</td><td>4.0</td><td>16.46</td></tr>

<tr><td colspan="8"><b>Qwen2.5-VL（Instruct）</b></td></tr>
<tr><td>Qwen2.5-VL-3B-Instruct</td><td>35.46</td><td>30.81</td><td>25.29</td><td>30.52</td><td>8.5</td><td>28.0</td><td>0.68</td></tr>
<tr><td>Qwen2.5-VL-7B-Instruct</td><td>36.26</td><td>37.54</td><td>32.84</td><td>35.55</td><td>17.1</td><td>32.0</td><td>0.87</td></tr>
<tr><td>Qwen2.5-VL-32B-Instruct</td><td>38.41</td><td>41.30</td><td>34.29</td><td>38.00</td><td>69.5</td><td>10.8</td><td>2.36</td></tr>
<tr><td>Qwen2.5-VL-72B-Instruct</td><td>23.78</td><td>27.67</td><td>50.76</td><td>34.07</td><td>146.5</td><td>5.8</td><td>4.26</td></tr>

<tr><td colspan="8"><b>Llama-3.2-Vision（Instruct）</b></td></tr>
<tr><td>Llama-3.2-11B-Vision-Instruct</td><td>31.59</td><td>32.91</td><td>29.34</td><td>31.28</td><td>26.2</td><td>16.8</td><td>1.55</td></tr>
<tr><td>Llama-3.2-90B-Vision-Instruct</td><td>27.26</td><td>26.33</td><td>27.72</td><td>27.10</td><td>183.6</td><td>2.7</td><td>8.05</td></tr>

<tr><td colspan="8"><b>Drive-KD（蒸馏后）</b></td></tr>
<tr><td>InternVL3-1B (Single)</td><td>43.13</td><td>34.32</td><td>52.97</td><td>43.47</td><td><b><u>4.1</u></b></td><td><b><u>45.7</u></b></td><td><b><u>0.45</u></b></td></tr>
<tr><td>Qwen2.5-VL-3B-Instruct (Single)</td><td>45.59</td><td>34.47</td><td>51.97</td><td>44.01</td><td>8.5</td><td>28.0</td><td>0.68</td></tr>
<tr><td>InternVL3-1B (Multi)</td><td>43.50</td><td>33.15</td><td><b><u>55.51</u></b></td><td>44.05</td><td><b><u>4.1</u></b></td><td><b><u>45.7</u></b></td><td><b><u>0.45</u></b></td></tr>
<tr><td>Qwen2.5-VL-3B-Instruct (Multi)</td><td><b><u>45.63</u></b></td><td>36.41</td><td>54.07</td><td>45.37</td><td>8.5</td><td>28.0</td><td>0.68</td></tr>
</tbody>
</table>

</div>

### 表 2：不同模型规模下的蒸馏（InternVL3）

<div align="center">

| 教师 | 学生 | 感知 | 推理 | 规划 | 平均 |
|---:|---:|---:|---:|---:|---:|
| 8B  | 1B | **<u>43.50</u>** | 33.15 | 55.51 | 44.05 |
| 14B | 1B | 43.41 | 30.34 | 56.19 | 43.31 |
| 38B | 1B | 43.24 | 29.15 | 56.77 | 43.05 |
| 8B  | 2B | 43.14 | 36.97 | 56.01 | 45.37 |
| 14B | 2B | 41.74 | 35.40 | 56.84 | 44.66 |
| 38B | 2B | 42.87 | **<u>38.25</u>** | **<u>57.63</u>** | **<u>46.25</u>** |

</div>

### 表 3：蒸馏信号与注意力变体（InternVL3-1B）

> “--” 表示不适用（该实验只在单能力数据上训练/评测对应能力）。

<div align="center">

| 设置 / 变体 | 感知 | 推理 | 规划 |
|---|---:|---:|---:|
| **目标函数（单能力训练协议）** |||| 
| CE（SFT） | 40.86 | 29.05 | 45.63 |
| CE + KL | 39.60 | 28.16 | 43.36 |
| CE + Hidden（第 1 层） | 41.27 | -- | -- |
| CE + Hidden（中间层） | -- | 31.65 | -- |
| CE + Hidden（倒数第二层） | -- | -- | 45.04 |
| **更多注意力蒸馏变体** |||| 
| CE + Full Attn（第 1 层） | 42.46 | -- | -- |
| CE + A<sub>t-v</sub>（中间层） | -- | 30.42 | -- |
| CE + Full Attn（倒数第二层） | -- | -- | 51.47 |
| CE + A<sub>t-v</sub>（中间层 2→倒数第二层−1） | -- | 31.43 | -- |
| CE + A<sub>t-v</sub>（第 1 层，cosine） | 41.85 | -- | -- |
| CE + Full Attn（中间层，cosine） | -- | 32.87 | -- |
| CE + A<sub>t-v</sub>（倒数第二层，cosine） | -- | -- | 51.76 |
| **多教师 + 冲突处理** |||| 
| Multi-teacher（无投影） | 42.34 | 25.68 | 51.03 |
| Multi-teacher（G1） | 42.96 | 25.49 | 46.99 |
| Multi-teacher（G2） | 42.64 | 29.18 | 52.19 |
| **我们的方案（单教师）** | **<u>43.13</u>** | **<u>34.32</u>** | **<u>52.97</u>** |
| **我们的方案（多教师 + AGP）** | **<u>43.50</u>** | **<u>33.15</u>** | **<u>55.51</u>** |

</div>

---

## 🚀 快速开始

### 1) 环境准备

建议：**Python 3.10+**，Linux，CUDA GPU（≥ 1 张 GPU）。

```bash
pip install torch torchvision transformers accelerate tqdm pillow
```

### 2) 数据说明

本仓库包含 `data/demo.json`，仅用于展示**标注格式样例**。  
**我们不提供原始图片**：请从官方渠道获取并自行整理路径：

- **nuScenes**（多视角 + 单视角）
- **BDD100K**（单视角）

然后在 JSON 中把 `images` 改成你本地的真实图片路径（或在 loader 中做映射）。

### 3) 模型准备

`train.py` 默认使用以下路径：

- 教师：`models/InternVL3-8B`
- 学生：`models/InternVL3-1B`
- 数据：`data/demo.json`
- 输出：`checkpoints/`

也可用命令行参数覆盖（见下）。

### 4) 启动训练

最小可运行示例：

```bash
python train.py \
  --data-json data/demo.json \
  --teacher-model-path models/InternVL3-8B \
  --student-model-path models/InternVL3-1B \
  --epochs 1
```

三教师训练（可选）：

```bash
python train.py \
  --data-json data/demo.json \
  --teacher-perception-path models/teacher_perception \
  --teacher-reasoning-path models/teacher_reasoning \
  --teacher-planning-path models/teacher_planning \
  --student-model-path models/InternVL3-1B
```

### 5) 常用关键参数

- 损失权重：`--w-ce`, `--w-perception`, `--w-reasoning`, `--w-planning`
- 教师混合权重（每个 3 个浮点数）：  
  - `--mix-perception`（如 `0.8,0.1,0.1`）  
  - `--mix-reasoning`（如 `0.1,0.8,0.1`）  
  - `--mix-planning`（如 `0.1,0.1,0.8`）  
- 动态损失权重（Online Loss Reweighting）：`--use-dynamic-loss-weights`（及相关参数）
- AGP：`--use-agp`

> **收敛提示：** 不同模型/数据下，损失权重、混合权重、学习率、batch/累积步数等都需要按实际情况调整，以保证训练稳定并收敛。

---

## 📁 目录结构

```text
Drive-KD
├── README.md
├── README_CN.md
├── train.py
├── intern/
│   ├── trainer.py
│   ├── model.py
│   ├── qa_loader.py
│   ├── image_loader.py
│   └── markers.py
├── data/
│   └── demo.json
└── docs/
    └── figs/
        ├── pipeline.jpg
        ├── pre_study.jpg
        ├── AGP.jpg
        ├── perception_eval.jpg
        ├── reasoning_eval.jpg
        └── planning_eval.jpg
```

---

## 📌 引用

```bibtex
@article{lian2026drivekd,
  title={Drive-KD: Multi-Teacher Distillation for VLMs in Autonomous Driving},
  author={Lian, Weitong and Tang, Zecong and Li, Haoran and Gao, Tianjian and Wang, Yifei and Wang, Zixu and Meng, Lingyi and Ru, Tengju and Cui, Zhejun and Zhu, Yichen and Cao, Hangshuo and Kang, Qi and Chen, Tianxing and Qin, Yusen and Wang, Kaixuan and Zhang, Yu},
  journal={arXiv preprint arXiv:2601.21288},
  year={2026}
}
```

---

## ⚖️ 开源协议

本项目采用 **Apache License 2.0**。

---

## 🙏 致谢

感谢开源社区与数据集提供方（nuScenes、BDD100K）对本研究工作的支持。
