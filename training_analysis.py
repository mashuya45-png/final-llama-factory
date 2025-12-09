import json
import numpy as np
from pathlib import Path
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import requests
import tempfile
import shutil

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将跳过绘图功能")


def parse_trainer_state(checkpoint_dir="./output/alpaca_zh_model/checkpoint-30510"):
    """从trainer_state.json解析训练日志"""

    print(f"正在解析训练日志: {checkpoint_dir}")

    training_data = {
        'steps': [],
        'losses': [],
        'learning_rates': [],
        'grad_norms': [],
        'epochs': []
    }

    trainer_state_file = os.path.join(checkpoint_dir, "trainer_state.json")

    if not os.path.exists(trainer_state_file):
        print(f"错误: 找不到 {trainer_state_file}")
        print("可用的文件:")
        for file in os.listdir(checkpoint_dir):
            print(f"  - {file}")
        return generate_simulated_data()

    try:
        with open(trainer_state_file, 'r', encoding='utf-8') as f:
            trainer_state = json.load(f)

        print("成功加载trainer_state.json")

        # 提取训练历史
        log_history = trainer_state.get("log_history", [])

        for entry in log_history:
            if "loss" in entry and "step" in entry:
                training_data['steps'].append(entry["step"])
                training_data['losses'].append(entry["loss"])

                # 提取其他指标
                if "learning_rate" in entry:
                    training_data['learning_rates'].append(entry["learning_rate"])
                if "grad_norm" in entry:
                    training_data['grad_norms'].append(entry["grad_norm"])
                if "epoch" in entry:
                    training_data['epochs'].append(entry["epoch"])

        print(f"解析到 {len(training_data['steps'])} 个训练数据点")

    except Exception as e:
        print(f"解析trainer_state.json时出错: {e}")
        return generate_simulated_data()

    return training_data


def parse_training_args(checkpoint_dir):
    """从training_args.bin解析训练配置"""
    try:
        # 注意：training_args.bin是二进制文件，我们需要用transformers库来读取
        from transformers import TrainingArguments
        import torch

        args_file = os.path.join(checkpoint_dir, "training_args.bin")
        if os.path.exists(args_file):
            # 这是一个简化版本，实际可能需要更复杂的加载方式
            training_args = torch.load(args_file)
            return {
                'learning_rate': getattr(training_args, 'learning_rate', 2.0e-4),
                'num_train_epochs': getattr(training_args, 'num_train_epochs', 5.0),
                'max_steps': getattr(training_args, 'max_steps', 30510),
                'per_device_train_batch_size': getattr(training_args, 'per_device_train_batch_size', 2),
                'gradient_accumulation_steps': getattr(training_args, 'gradient_accumulation_steps', 4)
            }
    except Exception as e:
        print(f"解析训练参数时出错: {e}")

    return None


def generate_simulated_data():
    """生成模拟训练数据"""
    print("使用模拟数据进行演示")
    steps = list(range(0, 30510, 3051))
    losses = [3.2 * (0.85 ** i) for i in range(len(steps))]

    return {
        'steps': steps,
        'losses': losses,
        'learning_rates': [2.0e-4 * (0.9 ** i) for i in range(len(steps))],
        'grad_norms': [1.0 + 0.1 * np.random.random() for _ in range(len(steps))],
        'epochs': [i * 0.5 for i in range(len(steps))]
    }


def analyze_training_progress(training_data):
    """分析训练进度和效果"""
    if len(training_data['steps']) < 2:
        print("数据点不足，无法进行详细分析")
        return

    steps = training_data['steps']
    losses = training_data['losses']

    print("\n训练进度分析")
    print("=" * 50)

    # 基础统计
    initial_loss = losses[0]
    final_loss = losses[-1]
    total_improvement = initial_loss - final_loss
    improvement_percentage = (total_improvement / initial_loss) * 100

    print(f"训练起点: 步骤 {steps[0]}, 损失 {initial_loss:.4f}")
    print(f"训练终点: 步骤 {steps[-1]}, 损失 {final_loss:.4f}")
    print(f"总损失改进: {total_improvement:.4f} ({improvement_percentage:.1f}%)")

    # 收敛分析
    if len(losses) > 10:
        # 早期阶段（前25%）
        early_cutoff = len(losses) // 4
        early_improvement = losses[0] - losses[early_cutoff]
        early_rate = early_improvement / steps[early_cutoff]

        # 后期阶段（最后25%）
        late_start = -len(losses) // 4
        late_improvement = losses[late_start] - losses[-1]
        late_rate = late_improvement / (steps[-1] - steps[late_start])

        print(f"\n收敛分析:")
        print(f"早期阶段改进率: {early_rate:.6f} 损失/步")
        print(f"后期阶段改进率: {late_rate:.6f} 损失/步")

        convergence_ratio = late_rate / early_rate if early_rate > 0 else 0
        if convergence_ratio < 0.1:
            print("✅ 模型已充分收敛")
        elif convergence_ratio < 0.3:
            print("⚠️ 模型基本收敛，可能还有改进空间")
        else:
            print("❌ 模型可能尚未完全收敛")

    # 训练稳定性
    if len(losses) > 20:
        recent_losses = losses[-20:]
        loss_std = np.std(recent_losses)
        loss_cv = (loss_std / np.mean(recent_losses)) * 100  # 变异系数

        print(f"\n训练稳定性:")
        print(f"近期损失标准差: {loss_std:.4f}")
        print(f"损失变异系数: {loss_cv:.1f}%")

        if loss_cv < 10:
            print("✅ 训练过程非常稳定")
        elif loss_cv < 25:
            print("⚠️ 训练过程基本稳定")
        else:
            print("❌ 训练过程波动较大")


def setup_chinese_font():
    """设置中文字体支持 - 修复版本（兼容新版Matplotlib）"""
    try:
        # 方法1: 下载思源黑体
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"

        # 使用用户目录保存字体，避免权限问题
        font_dir = os.path.expanduser("~/.local/share/fonts")
        os.makedirs(font_dir, exist_ok=True)
        font_path = os.path.join(font_dir, "NotoSansCJKsc-Regular.otf")

        if not os.path.exists(font_path):
            print("正在下载中文字体...")
            try:
                response = requests.get(font_url, timeout=30)
                response.raise_for_status()
                with open(font_path, 'wb') as f:
                    f.write(response.content)
                print("✅ 中文字体下载完成")
            except Exception as e:
                print(f"❌ 字体下载失败: {e}")
                return setup_chinese_font_fallback()

        # 注册字体 - 新方法（兼容新版Matplotlib）
        if os.path.exists(font_path):
            # 方法1: 直接设置字体路径
            try:
                # 清除字体缓存（新方法）
                if hasattr(fm, '_get_fontconfig_fonts'):
                    fm._get_fontconfig_fonts.cache_clear()

                # 重新构建字体列表
                fm._rebuild()

                # 添加字体到管理器
                fm.fontManager.addfont(font_path)

                # 设置字体
                plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False

                # 验证字体是否设置成功
                from matplotlib import font_manager
                available_fonts = [f.name for f in font_manager.fontManager.ttflist]
                if 'Noto Sans CJK SC' in available_fonts:
                    print("✅ 中文字体设置成功")
                    return True
                else:
                    print("⚠️ 字体已添加但未在可用字体列表中")
                    return setup_chinese_font_fallback()

            except Exception as e:
                print(f"❌ 字体注册失败: {e}")
                return setup_chinese_font_fallback()
        else:
            print("❌ 字体文件不存在")
            return setup_chinese_font_fallback()

    except Exception as e:
        print(f"❌ 字体设置过程出错: {e}")
        return setup_chinese_font_fallback()


def plot_detailed_analysis(training_data):
    """绘制详细的分析图表 - 英文版本"""
    if not HAS_MATPLOTLIB:
        return

    # 使用英文标签避免字体问题
    # setup_chinese_font()  # 仍然尝试设置，但不依赖它


    steps = training_data['steps']
    losses = training_data['losses']

    if len(steps) < 3:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 损失曲线 - 英文标签
    axes[0, 0].plot(steps, losses, 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].scatter(steps, losses, c=losses, cmap='viridis', s=30)
    axes[0, 0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 对数坐标损失 - 英文标签
    axes[0, 1].semilogy(steps, losses, 'r-', linewidth=2)
    axes[0, 1].set_title('Training Loss (Log Scale)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Loss Value (Log)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 损失分布 - 英文标签
    axes[1, 0].hist(losses, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_title('Loss Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Loss Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 移动平均损失 - 英文标签
    if len(losses) > 10:
        window = max(3, len(losses) // 20)
        moving_avg = np.convolve(losses, np.ones(window) / window, mode='valid')
        axes[1, 1].plot(steps[window - 1:], moving_avg, 'purple', linewidth=2,
                        label=f'{window}-point Moving Average')
        axes[1, 1].scatter(steps, losses, alpha=0.3, s=10, color='blue', label='Raw Values')
        axes[1, 1].set_title('Loss Moving Average', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Loss Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 使用明确的文件名和路径检查
    output_file = 'detailed_training_analysis.png'
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, output_file)

    plt.savefig(full_path, dpi=300, bbox_inches='tight')

    # 确认文件创建
    if os.path.exists(full_path):
        file_size = os.path.getsize(full_path)
        print(f"✅ 图表已保存: {full_path} (大小: {file_size} 字节)")
    else:
        print(f"❌ 文件保存失败: {full_path}")

def analyze_training_logs():
    """主分析函数"""

    print("训练过程详细分析")
    print("=" * 60)

    # 解析训练数据
    checkpoint_dir = "./output/alpaca_zh_model/checkpoint-30510"
    training_data = parse_trainer_state(checkpoint_dir)

    if not training_data['steps']:
        print("无法获取训练数据，分析终止")
        return

    # 显示基础信息
    print(f"\n训练概况:")
    print(f"  总步数: {training_data['steps'][-1]}")
    print(f"  数据点数量: {len(training_data['steps'])}")
    print(f"  最终损失: {training_data['losses'][-1]:.4f}")

    # 显示关键点的损失
    print(f"\n关键训练点损失:")
    key_points = [0, len(training_data['steps']) // 4, len(training_data['steps']) // 2,
                  3 * len(training_data['steps']) // 4, -1]

    for idx in key_points:
        if idx < len(training_data['steps']):
            step = training_data['steps'][idx]
            loss = training_data['losses'][idx]
            percentage = (idx / (len(training_data['steps']) - 1)) * 100
            print(f"  {percentage:5.1f}% (步骤 {step:5d}): 损失 {loss:.4f}")

    # 详细分析
    analyze_training_progress(training_data)

    # 效率分析
    total_steps = training_data['steps'][-1]
    total_time = total_steps  # 假设每步时间相同
    total_improvement = training_data['losses'][0] - training_data['losses'][-1]

    print(f"\n训练效率:")
    print(f"  总改进/总步数: {total_improvement / total_steps:.6f}")
    print(f"  最终损失/初始损失: {training_data['losses'][-1] / training_data['losses'][0]:.3f}")

    # 绘制图表
    plot_detailed_analysis(training_data)

    # 保存结果
    save_comprehensive_results(training_data, checkpoint_dir)


def save_comprehensive_results(training_data, checkpoint_dir):
    """保存综合分析结果"""
    results = {
        'checkpoint_info': {
            'checkpoint_dir': checkpoint_dir,
            'final_step': training_data['steps'][-1],
            'final_loss': training_data['losses'][-1]
        },
        'training_stats': {
            'initial_loss': training_data['losses'][0],
            'final_loss': training_data['losses'][-1],
            'total_improvement': training_data['losses'][0] - training_data['losses'][-1],
            'improvement_percentage': ((training_data['losses'][0] - training_data['losses'][-1]) /
                                       training_data['losses'][0]) * 100,
            'total_data_points': len(training_data['steps'])
        },
        'convergence_analysis': {
            'early_stage_rate': None,
            'late_stage_rate': None,
            'convergence_status': '未知'
        }
    }

    # 计算收敛率（如果有足够数据）
    if len(training_data['losses']) > 10:
        early_cutoff = len(training_data['losses']) // 4
        late_start = -len(training_data['losses']) // 4

        early_rate = (training_data['losses'][0] - training_data['losses'][early_cutoff]) / training_data['steps'][
            early_cutoff]
        late_rate = (training_data['losses'][late_start] - training_data['losses'][-1]) / (
                    training_data['steps'][-1] - training_data['steps'][late_start])

        results['convergence_analysis'].update({
            'early_stage_rate': early_rate,
            'late_stage_rate': late_rate,
            'convergence_ratio': late_rate / early_rate if early_rate > 0 else 0
        })

    with open('training_analysis_comprehensive.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n综合分析结果已保存到: training_analysis_comprehensive.json")


if __name__ == "__main__":
    analyze_training_logs()