#!/usr/bin/env python3
"""
终极解决方案：使用与LLaMA-Factory相同的微调方法
但完全绕过其复杂的依赖链
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import json
from datetime import datetime

print("🤖 开始大模型微调 - 终极解决方案")
print(f"开始时间: {datetime.now()}")
print("=" * 50)

# 1. 环境验证
print("1. 验证环境...")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 2. 加载模型
print("2. 加载模型...")
model_name = "Qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto"
)
print(f"  模型: {model_name}")
print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

# 3. 配置LoRA（使用LLaMA-Factory的默认参数）
print("3. 配置LoRA...")
lora_config = LoraConfig(
    r=8,           # LLaMA-Factory默认
    lora_alpha=32, # LLaMA-Factory默认
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
print("  可训练参数:")
model.print_trainable_parameters()

# 4. 准备训练数据
print("4. 准备训练数据...")
train_data = [
    {"instruction": "解释机器学习", "output": "机器学习是人工智能的一个分支，它使计算机能够通过经验自动改进性能。"},
    {"instruction": "什么是神经网络", "output": "神经网络是受人脑启发的计算模型，由相互连接的神经元层组成。"},
    {"instruction": "解释过拟合", "output": "过拟合是模型在训练数据上表现很好，但在新数据上表现差的现象。"},
    {"instruction": "监督学习和无监督学习的区别", "output": "监督学习使用标签数据，无监督学习使用无标签数据。"},
    {"instruction": "什么是深度学习", "output": "深度学习是机器学习的子领域，使用多层神经网络学习数据表征。"}
]
print(f"  训练样本: {len(train_data)} 条")

# 5. 数据格式化
def format_instruction(example):
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

formatted_texts = [format_instruction(ex) for ex in train_data]

# 6. Tokenization - 修复版本
def tokenize_function(texts):
    encodings = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="pt"  # 关键修改：返回PyTorch张量
        )
        encodings.append({
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0)
        })
    return encodings

tokenized_data = tokenize_function(formatted_texts)

# 7. 训练配置（使用LLaMA-Factory的默认参数）
print("5. 配置训练参数...")
training_args = TrainingArguments(
    output_dir="./outputs/ultimate_solution",
    per_device_train_batch_size=1,      # LLaMA-Factory默认
    gradient_accumulation_steps=8,       # LLaMA-Factory默认
    learning_rate=1e-4,                  # LLaMA-Factory默认
    num_train_epochs=3,                  # 训练轮数
    logging_steps=5,
    save_steps=50,
    fp16=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

# 8. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    ),
)

# 9. 开始训练
print("6. 开始训练...")
print("=" * 50)
trainer.train()

# 10. 保存结果
trainer.save_model()
tokenizer.save_pretrained("./outputs/ultimate_solution")

print("=" * 50)
print("🎉 训练完成！")
print(f"📁 模型保存至: ./outputs/ultimate_solution")
print(f"⏰ 完成时间: {datetime.now()}")
print("=" * 50)
print("下一步: 测试微调模型并准备作业报告")