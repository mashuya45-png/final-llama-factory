import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

def load_models():
    """加载原始模型和微调模型"""
    print("正在加载模型...")
    
    # 原始模型
    original_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    original_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    
    # 微调模型
    tuned_model = AutoModelForCausalLM.from_pretrained(
        "./output/alpaca_zh_model",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tuned_tokenizer = AutoTokenizer.from_pretrained("./output/alpaca_zh_model", trust_remote_code=True)
    
    return original_model, original_tokenizer, tuned_model, tuned_tokenizer

def generate_comparison_response(model, tokenizer, prompt, max_length=512):
    """生成模型回答（优化用于对比分析）"""
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.cuda(),
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取生成的回答部分
    if prompt in response:
        response = response[len(prompt):].strip()
    
    return response

def analyze_response_quality(original_response, tuned_response, question, focus):
    """自动分析回答质量差异"""
    analysis_points = []
    
    # 长度分析
    orig_len = len(original_response)
    tuned_len = len(tuned_response)
    
    if tuned_len > orig_len * 1.5:
        analysis_points.append("✓ 微调模型回答更详细丰富")
    elif tuned_len < orig_len * 0.7:
        analysis_points.append("⚠ 微调模型回答相对简略")
    
    # 结构分析
    orig_sentences = original_response.count('。') + original_response.count('!') + original_response.count('?')
    tuned_sentences = tuned_response.count('。') + tuned_response.count('!') + tuned_response.count('?')
    
    if tuned_sentences > orig_sentences:
        analysis_points.append("✓ 微调模型回答结构更清晰")
    
    # 技术深度分析
    technical_indicators = ['首先', '其次', '最后', '步骤', '方法', '原理', '机制', '架构']
    orig_tech = sum(1 for indicator in technical_indicators if indicator in original_response)
    tuned_tech = sum(1 for indicator in technical_indicators if indicator in tuned_response)
    
    if tuned_tech > orig_tech:
        analysis_points.append("✓ 微调模型技术描述更系统化")
    
    # 具体性分析
    specific_indicators = ['例如', '比如', '具体来说', '如下', '包括']
    orig_spec = sum(1 for indicator in specific_indicators if indicator in original_response)
    tuned_spec = sum(1 for indicator in specific_indicators if indicator in tuned_response)
    
    if tuned_spec > orig_spec:
        analysis_points.append("✓ 微调模型包含更多具体示例")
    
    # 根据分析重点特别检查
    if "技术深度" in focus:
        depth_terms = ['原理', '机制', '架构', '组件', '工作流程']
        if any(term in tuned_response and term not in original_response for term in depth_terms):
            analysis_points.append("✓ 在技术深度方面有明显提升")
    
    if "实践指导" in focus:
        practice_terms = ['步骤', '流程', '方法', '实现', '代码']
        if any(term in tuned_response and term not in original_response for term in practice_terms):
            analysis_points.append("✓ 在实践指导性方面更好")
    
    return analysis_points

def qualitative_comparison():
    """定性对比分析"""
    
    print("开始定性分析对比")
    print("=" * 80)
    
    # 加载模型
    original_model, original_tokenizer, tuned_model, tuned_tokenizer = load_models()
    
    # 测试案例 - 覆盖不同方面
    test_cases = [
        {
            "question": "详细解释Transformer架构的工作原理，包括自注意力机制",
            "analysis_focus": "技术深度和准确性",
            "category": "理论基础"
        },
        {
            "question": "用中文写一个完整的机器学习数据预处理流程，包含具体步骤",
            "analysis_focus": "实践指导性和可操作性",
            "category": "实践应用"
        },
        {
            "question": "比较CNN和RNN在自然语言处理中的优缺点和适用场景",
            "analysis_focus": "对比分析的全面性和深度", 
            "category": "技术对比"
        },
        {
            "question": "如何评估一个语言模型的质量？列出主要评估指标和方法",
            "analysis_focus": "系统性和完整性",
            "category": "评估方法"
        },
        {
            "question": "解释梯度下降算法的工作原理和常见变体",
            "analysis_focus": "概念解释的清晰度",
            "category": "算法原理"
        }
    ]
    
    all_results = []
    
    for i, case in enumerate(tqdm(test_cases), 1):
        print(f"\n案例 {i}: {case['category']}")
        print(f"问题: {case['question']}")
        print(f"分析重点: {case['analysis_focus']}")
        print("-" * 60)
        
        prompt = f"问题：{case['question']}\n回答："
        
        # 生成回答
        print("原始模型生成中...")
        original_response = generate_comparison_response(original_model, original_tokenizer, prompt)
        
        print("微调模型生成中...")
        tuned_response = generate_comparison_response(tuned_model, tuned_tokenizer, prompt)
        
        # 分析质量差异
        analysis_points = analyze_response_quality(
            original_response, tuned_response, case['question'], case['analysis_focus']
        )
        
        # 输出结果
        print("\n原始模型回答:")
        print(f"  {original_response}")
        
        print("\n微调模型回答:")
        print(f"  {tuned_response}")
        
        print("\n自动分析:")
        for point in analysis_points:
            print(f"  {point}")
        
        if not analysis_points:
            print("  ⚠ 无明显质量差异")
        
        # 记录结果
        case_result = {
            "case_id": i,
            "category": case['category'],
            "question": case['question'],
            "analysis_focus": case['analysis_focus'],
            "original_response": original_response,
            "tuned_response": tuned_response,
            "analysis_points": analysis_points,
            "original_length": len(original_response),
            "tuned_length": len(tuned_response)
        }
        all_results.append(case_result)
        
        print("\n" + "="*80)
    
    # 保存详细结果
    with open("qualitative_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 总结统计
    total_improvements = sum(len(result["analysis_points"]) for result in all_results)
    print(f"\n定性分析总结:")
    print(f"测试案例数量: {len(test_cases)}")
    print(f"总改进点数量: {total_improvements}")
    print(f"平均每个案例改进点: {total_improvements/len(test_cases):.1f}")
    print(f"\n详细结果已保存到: qualitative_results.json")
    
    # 清理内存
    del original_model, tuned_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    qualitative_comparison()
