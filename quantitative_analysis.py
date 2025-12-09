import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def load_models():
    """加载原始模型和微调后的模型"""
    print("正在加载模型...")
    
    # 原始模型
    original_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    original_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    
    # 微调后的模型
    tuned_model = AutoModelForCausalLM.from_pretrained(
        "./output/alpaca_zh_model",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tuned_tokenizer = AutoTokenizer.from_pretrained("./output/alpaca_zh_model")
    
    return original_model, original_tokenizer, tuned_model, tuned_tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """生成模型回答"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.cuda(),
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取生成的回答部分（去掉输入提示）
    if prompt in response:
        response = response[len(prompt):].strip()
    
    return response

def evaluate_response_quality(response, question, category):
    """综合评估回答质量"""
    scores = {}
    
    # 1. 相关性评分
    relevance_score = min(len(response) / 100, 1.0)  # 基于长度初步判断
    
    # 2. 关键词匹配（按类别）
    category_keywords = {
        "基础概念": ["神经网络", "神经元", "层", "权重", "激活函数", "输入", "输出", "学习"],
        "技术问题": ["过拟合", "欠拟合", "正则化", "验证集", "测试集", "泛化", "交叉验证", "早停"],
        "编程能力": ["def", "function", "return", "循环", "递归", "代码", "python", "实现"],
        "专业概念": ["注意力", "query", "key", "value", "权重", "Transformer", "自注意力"],
        "实践应用": ["训练", "数据预处理", "特征工程", "模型选择", "评估指标", "部署", "调参"]
    }
    
    keyword_score = 0
    if category in category_keywords:
        keywords = category_keywords[category]
        matched_keywords = sum(1 for keyword in keywords if keyword in response)
        keyword_score = min(matched_keywords / 3, 1.0)  # 最多匹配3个关键词
    
    # 3. 结构完整性
    structure_score = 0
    if len(response.split('。')) >= 2:  # 至少有两个句子
        structure_score += 0.5
    if any(mark in response for mark in ['，', '。', '；']):  # 有标点符号
        structure_score += 0.3
    if len(response) > 50:  # 回答有一定长度
        structure_score += 0.2
    
    # 4. 信息密度（避免重复）
    words = response.replace(' ', '')
    if len(words) > 0:
        unique_chars = len(set(words))
        info_density = unique_chars / len(words)
        info_score = min(info_density * 2, 1.0)  # 归一化到0-1
    else:
        info_score = 0
    
    # 综合评分
    total_score = (
        relevance_score * 0.3 +
        keyword_score * 0.3 +
        structure_score * 0.2 +
        info_score * 0.2
    )
    
    return {
        "total_score": round(total_score * 5, 2),  # 转换为5分制
        "relevance_score": round(relevance_score * 5, 2),
        "keyword_score": round(keyword_score * 5, 2),
        "structure_score": round(structure_score * 5, 2),
        "info_score": round(info_score * 5, 2)
    }

def quantitative_analysis():
    """定量分析微调效果"""
    print("开始定量分析...")
    print("=" * 60)
    
    # 加载模型
    original_model, original_tokenizer, tuned_model, tuned_tokenizer = load_models()
    
    # 测试问题
    test_questions = [
        {"question": "解释神经网络的基本原理", "category": "基础概念"},
        {"question": "什么是过拟合？如何避免？", "category": "技术问题"},
        {"question": "写一个Python函数计算斐波那契数列", "category": "编程能力"},
        {"question": "用中文解释注意力机制", "category": "专业概念"},
        {"question": "如何训练一个文本分类模型？", "category": "实践应用"},
        {"question": "机器学习中的损失函数是什么？", "category": "基础概念"},
        {"question": "梯度下降算法的工作原理", "category": "技术问题"}
    ]
    
    results = []
    
    for i, item in enumerate(tqdm(test_questions), 1):
        question = item["question"]
        category = item["category"]
        
        prompt = f"问题：{question}\n回答："
        
        # 原始模型生成
        print(f"\n正在测试问题 {i}: {question}")
        print("原始模型生成中...")
        original_response = generate_response(original_model, original_tokenizer, prompt)
        original_scores = evaluate_response_quality(original_response, question, category)
        
        # 微调模型生成
        print("微调模型生成中...")
        tuned_response = generate_response(tuned_model, tuned_tokenizer, prompt)
        tuned_scores = evaluate_response_quality(tuned_response, question, category)
        
        improvement = ((tuned_scores["total_score"] - original_scores["total_score"]) / 
                      original_scores["total_score"] * 100) if original_scores["total_score"] > 0 else 0
        
        results.append({
            "id": i,
            "question": question,
            "category": category,
            "original_response": original_response,
            "tuned_response": tuned_response,
            "original_scores": original_scores,
            "tuned_scores": tuned_scores,
            "improvement": round(improvement, 1)
        })
        
        print(f"问题 {i}: {question}")
        print(f"类别: {category}")
        print(f"原始模型得分: {original_scores['total_score']}/5")
        print(f"微调模型得分: {tuned_scores['total_score']}/5") 
        print(f"提升: {improvement:.1f}%")
        print("-" * 40)
    
    # 计算总体统计
    original_avg = np.mean([r["original_scores"]["total_score"] for r in results])
    tuned_avg = np.mean([r["tuned_scores"]["total_score"] for r in results])
    avg_improvement = np.mean([r["improvement"] for r in results])
    
    print("\n总体统计:")
    print(f"原始模型平均得分: {original_avg:.2f}/5")
    print(f"微调模型平均得分: {tuned_avg:.2f}/5")
    print(f"平均提升: {avg_improvement:.1f}%")
    
    # 保存结果
    with open("quantitative_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "detailed_results": results,
            "summary": {
                "original_avg_score": original_avg,
                "tuned_avg_score": tuned_avg, 
                "avg_improvement": avg_improvement
            }
        }, f, indent=2, ensure_ascii=False)
    
    print("\n详细结果已保存到 quantitative_results.json")
    
    # 清理GPU内存
    del original_model, tuned_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    quantitative_analysis()
