import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import numpy as np
from tqdm import tqdm


def load_finetuned_model():
    """加载微调后的模型"""
    base_model = "Qwen/Qwen2.5-1.5B"
    adapter_path = "./output/alpaca_zh_balanced"

    print("加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("加载LoRA适配器...")
    model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def qualitative_evaluation(model, tokenizer):
    """定性评估：生成示例"""
    test_prompts = [
        "保持健康的三个提示。",
        "如何学习编程？",
        "解释人工智能的概念。",
        "写一首关于春天的诗。",
        "如何做好时间管理？"
    ]

    results = []

    for prompt in test_prompts:
        # 构建Qwen格式的对话
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # 格式化输入
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 生成回复
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "response": response
        })

        print(f"\n提示: {prompt}")
        print(f"回复: {response}")
        print("-" * 50)

    # 保存结果
    with open("qualitative_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def calculate_perplexity(model, tokenizer, texts):
    """计算困惑度"""
    perplexities = []

    for text in tqdm(texts, desc="计算困惑度"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())

    return np.mean(perplexities)


def main():
    # 加载模型
    model, tokenizer = load_finetuned_model()

    print("开始定性评估...")
    qualitative_results = qualitative_evaluation(model, tokenizer)

    print("\n开始定量评估...")
    # 这里可以添加你的测试数据
    test_texts = [
        "保持健康需要均衡饮食和适量运动。",
        "学习编程需要实践和耐心。",
        # 添加更多测试文本...
    ]

    if test_texts:
        avg_perplexity = calculate_perplexity(model, tokenizer, test_texts)
        print(f"平均困惑度: {avg_perplexity:.2f}")

    # 生成评估报告
    generate_report(qualitative_results)


def generate_report(qualitative_results):
    """生成评估报告"""
    report = {
        "model_info": {
            "base_model": "Qwen/Qwen2.5-1.5B",
            "adapter_path": "./output/alpaca_zh_balanced",
            "finetuning_method": "LoRA"
        },
        "training_summary": {
            "final_loss": 1.5112,
            "training_steps": 800,
            "training_time": "32分21秒"
        },
        "qualitative_evaluation": qualitative_results,
        "quantitative_metrics": {
            "perplexity": "需要测试数据计算"
        }
    }

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("评估报告已保存至 evaluation_report.json")


if __name__ == "__main__":
    main()