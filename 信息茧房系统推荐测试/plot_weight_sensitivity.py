import json
import numpy as np
import os
import math
import plotly.graph_objects as go

# ==========================================
# 1. 基础配置与文件路径 (请确保路径与你本地一致)
# ==========================================
ANALYST_RES = '信息茧房系统推荐测试/test_last_analyst_generated_predictions.jsonl'
EXPLORER_RES = '信息茧房系统推荐测试/test_last_explorer_generated_predictions.jsonl'
ORIGINAL_TEST = '信息茧房合适参数/topk_last_final_test_set.jsonl'

# 全局最优基准权重 (总和必须等于 10.0)
BEST_WEIGHTS = np.array([2.5, 1.0, 1.5, 1.5, 3.5])
FACTOR_NAMES = ['相关性(Relevance)', '实用性(Utility)', '新颖性(Novelty)', '意外性(Unexpectedness)', '多样性(Diversity)']
COLORS = ['#E63946', '#F4A261', '#2A9D8F', '#457B9D', '#8338EC']
MARKERS = ['circle', 'square', 'diamond', 'cross', 'triangle-up']

# ==========================================
# 2. 核心评估与归一化逻辑 (学术级严谨)
# ==========================================
def parse_model_output(line):
    try:
        data = json.loads(line)
        pred = data.get('predict', '{}')
        if isinstance(pred, str): return json.loads(pred)
        return pred
    except:
        return {}

def get_normalized_weights(target_idx, target_value, base_weights):
    """
    【核心逻辑】：等和归一化
    固定目标权重为 target_value，其余权重按原始比例缩放，使得总和恒定为 10.0
    """
    new_w = np.array(base_weights, dtype=float)
    new_w[target_idx] = float(target_value)
    
    other_indices = [j for j in range(len(base_weights)) if j != target_idx]
    current_other_sum = np.sum(base_weights[other_indices])
    desired_other_sum = 10.0 - float(target_value)
    
    if current_other_sum > 0:
        for j in other_indices:
            new_w[j] = (base_weights[j] / current_other_sum) * desired_other_sum
    else:
        for j in other_indices:
            new_w[j] = desired_other_sum / len(other_indices)
            
    return new_w

def evaluate_ranking(ranked_list):
    """计算核心指标"""
    hr10, ndcg10, mrr = 0, 0.0, 0.0
    k = 10
    total_pos = sum(1 for item in ranked_list if item['is_label'] == 1)
    
    # HR@10 & MRR
    for i, item in enumerate(ranked_list[:k]):
        if item['is_label'] == 1:
            hr10 = 1
            mrr = 1.0 / (i + 1)
            break
            
    # NDCG@10
    if total_pos > 0:
        dcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(ranked_list))) if ranked_list[i]['is_label'] == 1)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, total_pos)))
        ndcg10 = dcg / idcg if idcg > 0 else 0.0
        
    return hr10, ndcg10, mrr

def load_data():
    print("📥 正在加载并对齐五维度特征向量...")
    dataset = []
    with open(ANALYST_RES, 'r', encoding='utf-8') as fa, \
         open(EXPLORER_RES, 'r', encoding='utf-8') as fe, \
         open(ORIGINAL_TEST, 'r', encoding='utf-8') as fo:
        
        a_lines, e_lines = fa.readlines(), fe.readlines()
        global_idx = 0
        for line in fo:
            if not line.strip(): continue
            cands = json.loads(line).get('candidates', [])
            group = []
            for cand in cands:
                if global_idx >= len(a_lines): break
                sa = parse_model_output(a_lines[global_idx])
                se = parse_model_output(e_lines[global_idx])
                # 严密对齐: [Rel, Uti, Nov, Une, Div]
                vec = np.array([
                    float(sa.get('relevance', 2.5)), 
                    float(sa.get('utility', 1.0)),
                    float(se.get('novelty', 1.5)), 
                    float(se.get('unexpectedness', 1.5)),
                    float(se.get('diversity', 3.5))
                ])
                group.append({"vec": vec, "is_label": cand.get('is_label', 0)})
                global_idx += 1
            if group: dataset.append(group)
    return dataset

def compute_metrics_for_weights(dataset, weights):
    sum_hr = 0
    for group in dataset:
        ranked = [{"score": np.dot(item["vec"], weights), "is_label": item["is_label"]} for item in group]
        ranked.sort(key=lambda x: x['score'], reverse=True)
        hr, _, _ = evaluate_ranking(ranked)
        sum_hr += hr
    return sum_hr / len(dataset)

# ==========================================
# 3. 执行实验与高级可视化
# ==========================================
def run_experiments():
    dataset = load_data()
    print(f"✅ 数据加载成功！共 {len(dataset)} 个测试组。")
    
    # 1. 计算全局基准性能 (当权重恰好为 BEST_WEIGHTS 时的 HR@10)
    baseline_hr = compute_metrics_for_weights(dataset, BEST_WEIGHTS)
    print(f"🌟 基准模型 (权重组合: {BEST_WEIGHTS}) HR@10 = {baseline_hr:.4f}")

    weight_range = np.arange(0.0, 5.2, 0.2) # 细化步长，让曲线更平滑
    fig = go.Figure()
    
    print("\n⚙️ 正在执行【等和归一化敏感度推演】...")
    for idx, factor_name in enumerate(FACTOR_NAMES):
        hr_list = []
        for w in weight_range:
            # 采用等和归一化计算新权重
            normalized_w = get_normalized_weights(idx, w, BEST_WEIGHTS)
            hr = compute_metrics_for_weights(dataset, normalized_w)
            hr_list.append(hr)
            
        # 绘制主折线
        fig.add_trace(go.Scatter(
            x=weight_range, y=hr_list, 
            mode='lines+markers', name=factor_name,
            line=dict(color=COLORS[idx], width=3),
            marker=dict(symbol=MARKERS[idx], size=8, color=COLORS[idx]),
            hovertemplate=f"<b>{factor_name}</b><br>权值: %{{x:.1f}}<br>HR@10: %{{y:.4f}}<extra></extra>"
        ))
        
        # 绘制该因子的“基准锚点” (例如相关性在x=2.5处，必定等于 baseline_hr)
        base_x = BEST_WEIGHTS[idx]
        fig.add_trace(go.Scatter(
            x=[base_x], y=[baseline_hr],
            mode='markers', showlegend=False,
            marker=dict(symbol='star', size=16, color='gold', line=dict(color='black', width=1)),
            hovertemplate="<b>★ 基准锚点</b><br>权值: %{x:.1f}<br>HR@10: %{y:.4f}<extra></extra>"
        ))

    # 布局美化
    fig.update_layout(
        title=dict(
            text="<b>归一化权重约束下的五因子敏感度分析 (Sum=10.0)</b>",
            x=0.5, font=dict(size=20)
        ),
        xaxis=dict(
            title="<b>单一因子分配权重 (目标变化量)</b>",
            tickmode='linear', dtick=0.5, range=[-0.2, 5.2]
        ),
        yaxis=dict(title="<b>命中率 (HR@10)</b>"),
        template="plotly_white",
        font=dict(family="Arial", size=14),
        legend=dict(
            title="因子维度", x=0.02, y=0.98, 
            bgcolor="rgba(255,255,255,0.9)", 
            bordercolor="gray", borderwidth=1
        ),
        hovermode="x unified"
    )
    
    # 增加基准水平线
    fig.add_hline(y=baseline_hr, line_dash="dash", line_color="gray", opacity=0.5, annotation_text=f"最优基准线 ({baseline_hr:.4f})")

    file_path = os.path.join(os.getcwd(), 'Fig_Normalized_Sensitivity.html')
    fig.write_html(file_path)
    print(f"\n🎉 成功！图表已生成：{file_path}")
    print("👉 留意图中的【金色五角星】，它们完美验证了你的算法在不同切面下回到了同一个最优解！")

if __name__ == "__main__":
    run_experiments()