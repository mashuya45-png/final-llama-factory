import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os


class ModelDemo:
    def __init__(self):
        self.tuned_model = None
        self.tuned_tokenizer = None
        self.original_model = None
        self.original_tokenizer = None
        self.models_loaded = False

    def load_models(self):
        """加载模型"""
        if self.models_loaded:
            return "✅ 模型已加载"

        try:
            print("正在加载模型...")

            # 加载微调模型
            tuned_path = "./output/alpaca_zh_model"
            self.tuned_tokenizer = AutoTokenizer.from_pretrained(
                tuned_path, trust_remote_code=True
            )
            self.tuned_model = AutoModelForCausalLM.from_pretrained(
                tuned_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # 加载原始模型用于对比
            original_path = "Qwen/Qwen2.5-1.5B"
            self.original_tokenizer = AutoTokenizer.from_pretrained(
                original_path, trust_remote_code=True
            )
            self.original_model = AutoModelForCausalLM.from_pretrained(
                original_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            self.models_loaded = True
            print("模型加载完成!")
            return "✅ 模型加载成功！可以开始提问了"

        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}"

    def generate_response(self, model, tokenizer, prompt, max_length=512):
        """生成回答"""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
            return_attention_mask=True
        )

        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response

    def compare_models(self, question):
        """对比两个模型的回答"""
        if not self.models_loaded:
            return "请先加载模型", "请先加载模型", "0.0", "0.0"

        prompt = f"问题：{question}\n回答："

        # 生成原始模型回答
        start_time = time.time()
        original_response = self.generate_response(
            self.original_model, self.original_tokenizer, prompt
        )
        original_time = time.time() - start_time

        # 生成微调模型回答
        start_time = time.time()
        tuned_response = self.generate_response(
            self.tuned_model, self.tuned_tokenizer, prompt
        )
        tuned_time = time.time() - start_time

        return original_response, tuned_response, f"{original_time:.2f}s", f"{tuned_time:.2f}s"


# 创建模型演示实例
demo = ModelDemo()


def create_demo_interface():
    """创建Gradio演示界面"""

    # 预设问题
    preset_questions = [
        "详细解释Transformer的自注意力机制",
        "用Python写一个快速排序算法",
        "什么是过拟合？如何避免？",
        "比较CNN和RNN在自然语言处理中的优缺点",
        "如何评估一个语言模型的质量？",
        "解释梯度下降算法的工作原理"
    ]

    with gr.Blocks(
            title="Qwen微调模型演示",
            theme=gr.themes.Soft(),
            css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .response-box {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            background: white;
        }
        .original-response {
            border-left: 4px solid #ff6b6b;
        }
        .tuned-response {
            border-left: 4px solid #4ecdc4;
        }
        """
    ) as demo_interface:

        gr.Markdown("""
        # 🚀 Qwen2.5-1.5B 微调模型演示
        **对比展示微调前后的模型表现**
        """)

        # 模型加载状态
        with gr.Row():
            load_status = gr.Textbox(
                label="模型状态",
                value="点击下方按钮加载模型",
                interactive=False
            )
            load_btn = gr.Button("🔄 加载模型", variant="primary")

        gr.Markdown("---")

        # 问题输入区域
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="💬 输入您的问题",
                    placeholder="在这里输入您想问的问题...",
                    lines=3,
                    max_lines=6
                )

                submit_btn = gr.Button("🚀 开始对比", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 💡 快速测试")
                preset_btns = []
                for i, question in enumerate(preset_questions):
                    btn = gr.Button(
                        question[:30] + "..." if len(question) > 30 else question,
                        size="sm"
                    )
                    preset_btns.append(btn)

        gr.Markdown("---")

        # 结果显示区域
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔵 原始模型回答")
                original_time = gr.Textbox(label="生成时间", interactive=False)
                original_output = gr.Textbox(
                    label="",
                    lines=8,
                    max_lines=12,
                    show_copy_button=True
                )

            with gr.Column():
                gr.Markdown("### 🟢 微调模型回答")
                tuned_time = gr.Textbox(label="生成时间", interactive=False)
                tuned_output = gr.Textbox(
                    label="",
                    lines=8,
                    max_lines=12,
                    show_copy_button=True
                )

        gr.Markdown("---")

        # 评估反馈区域
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 改进评估")
                improvement_feedback = gr.Textbox(
                    label="改进点分析",
                    lines=3,
                    interactive=False,
                    value="回答生成后将显示改进分析..."
                )

            with gr.Column():
                gr.Markdown("### 🎯 展示建议")
                demo_tips = gr.Textbox(
                    label="现场展示要点",
                    lines=3,
                    interactive=False,
                    value="1. 注意技术术语的准确性\n2. 观察回答结构的完整性\n3. 比较实用建议的丰富程度"
                )

        # 事件处理
        def load_models_wrapper():
            return demo.load_models()

        def process_question(question):
            original, tuned, o_time, t_time = demo.compare_models(question)

            # 简单的改进分析
            improvement = analyze_improvement(original, tuned)

            return original, tuned, o_time, t_time, improvement

        def preset_question_wrapper(question):
            return question, "", "", "", "", "点击'开始对比'查看结果"

        # 绑定事件
        load_btn.click(
            load_models_wrapper,
            outputs=load_status
        )

        submit_btn.click(
            process_question,
            inputs=question_input,
            outputs=[
                original_output, tuned_output,
                original_time, tuned_time,
                improvement_feedback
            ]
        )

        for btn in preset_btns:
            btn.click(
                lambda x=btn.value: preset_question_wrapper(x),
                outputs=[
                    question_input, original_output, tuned_output,
                    original_time, tuned_time, improvement_feedback
                ]
            )

    return demo_interface


def analyze_improvement(original, tuned):
    """分析改进点"""
    improvements = []

    # 长度对比
    if len(tuned) > len(original) * 1.5:
        improvements.append("📈 回答更详细丰富")
    elif len(tuned) < len(original) * 0.7:
        improvements.append("📝 回答更简洁精准")

    # 技术深度
    tech_terms = ['原理', '机制', '架构', '算法', '实现', '步骤']
    tuned_tech = sum(1 for term in tech_terms if term in tuned)
    original_tech = sum(1 for term in tech_terms if term in original)

    if tuned_tech > original_tech:
        improvements.append("🔬 技术描述更深入")

    # 结构完整性
    if tuned.count('。') > original.count('。') + 2:
        improvements.append("📋 结构更清晰完整")

    # 实用性
    practice_terms = ['例如', '比如', '具体来说', '步骤', '方法']
    if any(term in tuned and term not in original for term in practice_terms):
        improvements.append("💡 包含更多实用示例")

    if not improvements:
        improvements.append("⏳ 正在分析改进点...")

    return " | ".join(improvements)


# 创建界面
if __name__ == "__main__":
    interface = create_demo_interface()

    # 尝试多个端口
    ports_to_try = [7861, 7862, 7863, 7864, 7865]

    for port in ports_to_try:
        try:
            print(f"尝试在端口 {port} 启动...")
            interface.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=True,
                inbrowser=True
            )
            break
        except OSError as e:
            if "Cannot find empty port" in str(e):
                print(f"端口 {port} 也被占用，尝试下一个...")
                continue
            else:
                raise e
    else:
        print("所有尝试的端口都被占用，请手动指定端口")
        interface.launch(
            server_name="0.0.0.0",
            server_port=0,  # 自动选择端口
            share=True,
            inbrowser=True
        )