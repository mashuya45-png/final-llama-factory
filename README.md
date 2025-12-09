# 大模型基础与应用期末作业：复现该模型微调步骤指导

本仓库记录了在 Ubuntu 环境下配置 LLaMA-Factory 训练与推理环境的完整过程，包括依赖安装、版本冲突解决、环境验证与示例运行。

## 环境要求

- **操作系统**：Ubuntu 18.04+
- **Python**：3.10
- **CUDA**：≥ 11.7（需与 PyTorch 版本匹配）
- **GPU**：NVIDIA GPU（已测试 RTX 2080 Ti）
- **包管理工具**：conda、pip

---

## 一、环境准备

### 1. 创建 Conda 环境
```bash

conda create -n llama_factory python=3.10 -y
conda activate llama_factory
```

### 2. 安装 PyTorch 及相关库

```bash

pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 二、安装 LLaMA-Factory

### 1. 克隆仓库
```bash

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### 2. 安装依赖（跳过 `av` 包）
```bash
pip install -e . --no-deps
```

---

## 三、解决依赖冲突

### 1. 升级 CMake（解决 `pyarrow` 编译问题）
```bash
conda install -c conda-forge cmake -y
cmake --version  # 确保 ≥ 3.5.0
```

### 2. 安装核心依赖（指定兼容版本）
```bash
pip install numpy==1.24.0
pip install transformers==4.49.0
pip install accelerate==0.23.0
pip install datasets==2.16.0 --prefer-binary -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install safetensors==0.5.3
pip install gradio>=4.38.0,<=5.45.0
pip install peft==0.14.0
pip install trl==0.8.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 安装其他必要依赖
```bash
pip install sentencepiece scipy tiktoken einops fire librosa matplotlib modelscope omegaconf protobuf sse-starlette tyro hf-transfer propcache
```

---

## 五、运行示例

### 1. 数据准备
- 编辑 LLaMA-Factory/data/dataset_info.json文件，添加本地数据集配置。
- 将alpaca_zh_local的file_name改为我们要使用的数据集alpaca_gpt4_data_zh.json
- alpaca_gpt4_data_zh.json是我们使用的数据集，下载下来存放在LLaMA-Factory/data目录中

### 2. 训练模型（模型配置文件alpaca_zh_sft.yaml）
```bash

llamafactory-cli train examples/alpaca_zh_sft.yaml
```
我们训练好的模型权重存放在output/alpaca_zh-model目录中

### 3. 运行评估脚本
定量分析
```bash
python quantitative_analysis.py

```
定性分析
```bash
python qualitative_analysis.py

```

训练过程可视化
```bash
python training_analysis.py
```
输出的定性分析、定量分析结果以及训练曲线图均存放在根目录中
---

## 复现命令总结

```bash

conda create -n llama_factory python=3.10 -y
conda activate llama_factory

pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1

git clone https://github.com/hiyouga/LLaMA-Factory.git

cd LLaMA-Factory
pip install -e . --no-deps
conda install -c conda-forge cmake -y
pip install transformers==4.49.0 accelerate==0.23.0 datasets==2.16.0 gradio>=4.38.0,<=5.45.0 safetensors==0.5.3 peft==0.14.0 trl==0.8.6
pip install sentencepiece scipy tiktoken einops fire librosa matplotlib modelscope omegaconf protobuf sse-starlette tyro hf-transfer propcache

# 训练模型
llamafactory-cli train examples/alpaca_zh_sft.yaml

# 定量分析
python quantitative_analysis.py
# 定性分析
python qualitative_analysis.py
# 训练过程可视化
python training_analysis.py
```
---

## 许可证与致谢

本复现过程基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 开源项目，遵循其原许可证。感谢所有相关开源工具与社区的贡献。

---
