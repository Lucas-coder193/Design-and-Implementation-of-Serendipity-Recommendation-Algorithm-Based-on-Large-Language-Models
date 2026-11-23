import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置中文和学术风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号


def draw_conceptual_diagram():
    # 创建画布 (16:9 比例)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis('off')  # 隐藏所有坐标轴

    # 1. 设置颜色和字体配置
    c_base = '#333333'  # 基础黑色
    c_theory = '#b3cde3'  # 理论框颜色 (学术蓝)
    c_analyst = '#ffffcc'  # 专家框颜色 (学术黄)
    c_explorer = '#ccebc5'  # 专家框颜色 (学术绿)
    fontsize_title = 20
    fontsize_factor = 14
    fontsize_desc = 10

    # 2. 绘制标题
    ax.text(0.5, 0.95, "从文献理论到双智能体解耦评估体系的构建逻辑",
            ha='center', va='top', fontsize=fontsize_title, fontweight='bold', color=c_base)

    # ================= 3. 绘制理论基础层 (顶部) =================
    # 理论框矩形
    theory_box = patches.FancyBboxPatch((0.15, 0.8), 0.7, 0.1, boxstyle="round,pad=0.03",
                                        ec=c_base, fc=c_theory, alpha=0.3)
    ax.add_patch(theory_box)
    ax.text(0.5, 0.85, "多维机缘巧合（Serendipity）理论基础\n( 基于 Ziarani 等[20]对 2013-2019 年文献的统计分类结果 )",
            ha='center', va='center', fontsize=12, fontweight='bold', color=c_base)

    # ================= 4. 绘制你解构出的 5 个因子 (中部) =================
    factors = [
        ("相关性 (Rel)", "基础底线 (认知范畴内)"),
        ("实用性 (Util)", "现实意义 (情绪/实用价值)"),
        ("新颖性 (Nov)", "稀缺程度 (非同质化)"),
        ("意外性 (Unexp)", "语义跳跃 (意料之外,情理之中)"),
        ("多样性 (Div)", "全局视野 (离散度最大化)")
    ]

    factor_xs = np.linspace(0.1, 0.9, len(factors))
    for i, (name, desc) in enumerate(factors):
        # 因子框
        ax.add_patch(patches.Rectangle((factor_xs[i] - 0.06, 0.6), 0.12, 0.15, fc='white', ec=c_base))
        ax.text(factor_xs[i], 0.69, name, ha='center', va='center', fontsize=fontsize_factor, fontweight='bold')
        ax.text(factor_xs[i], 0.63, desc, ha='center', va='top', fontsize=fontsize_desc, alpha=0.8)

    # ================= 5. 绘制你设计的解耦评估层 (底部) =================
    # 绘制 Analyst 专家
    ax.add_patch(patches.FancyBboxPatch((0.1, 0.3), 0.35, 0.2, boxstyle="round,pad=0.05",
                                        ec=c_base, fc=c_analyst))
    ax.text(0.275, 0.45, "Analyst 智能体 (分析师)", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.275, 0.38, "核心任务：深度语义对齐\n( 保守分析 )", ha='center', va='top', fontsize=11, alpha=0.9)

    # 绘制 Explorer 专家
    ax.add_patch(patches.FancyBboxPatch((0.55, 0.3), 0.35, 0.2, boxstyle="round,pad=0.05",
                                        ec=c_base, fc=c_explorer))
    ax.text(0.725, 0.45, "Explorer 智能体 (探索者)", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.725, 0.38, "核心任务：打破同质化边界\n( 特征对比推理 )", ha='center', va='top', fontsize=11, alpha=0.9)

    # ================= 6. 绘制所有箭头链接逻辑 (解耦与量化) =================
    # 从理论到因子 (聚合箭头)
    for fx in factor_xs:
        ax.arrow(0.5, 0.8, fx - 0.5, -0.05, head_width=0.01, head_length=0.01, fc=c_base, ec=c_base, alpha=0.2)

    # 从因子到专家 (解耦箭头)
    # Analyst 接 Rel, Util
    ax.arrow(factor_xs[0], 0.6, 0, -0.1, head_width=0.01, head_length=0.01, fc=c_base, ec=c_base)
    ax.arrow(factor_xs[1], 0.6, 0, -0.1, head_width=0.01, head_length=0.01, fc=c_base, ec=c_base)
    # Explorer 接 Nov, Unexp, Div
    ax.arrow(factor_xs[2], 0.6, 0, -0.1, head_width=0.01, head_length=0.01, fc=c_base, ec=c_base)
    ax.arrow(factor_xs[3], 0.6, 0, -0.1, head_width=0.01, head_length=0.01, fc=c_base, ec=c_base)
    ax.arrow(factor_xs[4], 0.6, 0, -0.1, head_width=0.01, head_length=0.01, fc=c_base, ec=c_base)

    # 添加解耦的标注
    ax.text(0.5, 0.55, "将模糊的主观感受解构并解耦...", ha='center', va='center', fontsize=12, fontweight='bold',
            color=c_base)

    plt.tight_layout()
    plt.show()


# 运行绘图
if __name__ == "__main__":
    draw_conceptual_diagram()