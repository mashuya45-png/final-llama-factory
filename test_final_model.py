import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def test_both_models():
    """测试原始模型和微调后的模型，进行对比"""
    
    print("=" * 80)
    print("大模型微调效果对比测试")
    print("=" * 80)
    
    # 测试问题列表
    test_questions = [
        "你好，请介绍一下你自己",
        "什么是大语言模型？",
        "解释一下监督微调(SFT)",
        "DPO和SFT有什么区别？",
        "什么是RAG？"
    ]
    
    # 测试原始模型
    print("\n🔵 测试原始模型 (Qwen2.5-1.5B)...")
    try:
        original_model_path = "Qwen/Qwen2.5-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. 用户: {question}")
            messages = [{"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"   原始模型: {response}")
            
    except Exception as e:
        print(f"❌ 原始模型测试失败: {e}")
    
    print("\n" + "=" * 60)
    
    # 测试微调后的模型
    print("\n🟢 测试微调后的模型...")
    try:
        tuned_model_path = "./output/my_course_sft"
        
        # 检查模型文件是否存在
        if not os.path.exists(tuned_model_path):
            print(f"❌ 微调模型路径不存在: {tuned_model_path}")
            return
        
        print(f"✅ 找到微调模型路径: {tuned_model_path}")
        
        # 重新加载tokenizer和模型
        tokenizer = AutoTokenizer.from_pretrained(tuned_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            tuned_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. 用户: {question}")
            messages = [{"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"   微调模型: {response}")
            
    except Exception as e:
        print(f"❌ 微调模型测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("测试完成！可以在上面的输出中对比原始模型和微调模型的效果差异。")

if __name__ == "__main__":
    test_both_models()
