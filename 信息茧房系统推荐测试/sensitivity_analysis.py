import json
import numpy as np
import os
import plotly.graph_objects as go
import plotly.io as pio

# ==========================================
# 1. 配置路径 (请确保与你的实际路径一致)
# ==========================================
ANALYST_RES = '信息茧房系统推荐测试/test_last_analyst_generated_predictions.jsonl'
EXPLORER_RES = '信息茧房系统推荐测试/test_last_explorer_generated_predictions.jsonl'
ORIGINAL_TEST = '信息茧房合适参数/topk_last_final_test_set.jsonl'

def parse_model_output(line):
    try:
        data = json.loads(line)
        predict_content = data.get('predict', '{}')
        if isinstance(predict_content, str):
            return json.loads(predict_content)
        return predict_content
    except:
        return {}

def calculate_ndcg(ranked_list, k):
    dcg = 0.0
    idcg = 0.0
    total_positives = sum(1 for item in ranked_list if item['is_label'] == 1)
    if total_positives == 0: return 0.0
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i]['is_label'] == 1:
            dcg += 1.0 / np.log2(i + 2)
    for i in range(min(k, total_positives)):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_with_weights(weight_array, test_groups):
    """给定一组权重，评估整个测试集的 HR@10 和 NDCG@10"""
    hr10_sum = 0
    ndcg10_sum = 0
    total_users = 0

    for group in test_groups:
        group_scores = []
        for item in group:
            rel = float(item['analyst'].get('relevance', 2.5))
            util = float(item['analyst'].get('utility', 1.0))
            nov = float(item['explorer'].get('novelty', 1.5))
            unexp = float(item['explorer'].get('unexpectedness', 1.5))
            div = float(item['explorer'].get('diversity', 3.5))
            
            score_vector = np.array([rel, util, nov, unexp, div])
            final_score = np.dot(score_vector, weight_array)
            
            group_scores.append({
                "score": final_score,
                "is_label": item['is_label']
            })
            
        ranked = sorted(group_scores, key=lambda x: x['score'], reverse=True)
        if any(it['is_label'] == 1 for it in ranked[:10]): 
            hr10_sum += 1
        ndcg10_sum += calculate_ndcg(ranked, 10)
        total_users += 1

    return hr10_sum / total_users, ndcg10_sum / total_users

def load_all_data():
    """提前把数据加载进内存，避免重复读文件"""
    print("[INFO] 正在加载测试数据到内存...")
    test_groups = []
    
    # 先读取所有预测结果
    analyst_results = []
    explorer_results = []
    with open(ANALYST_RES, 'r', encoding='utf-8') as fa:
        for line in fa:
            analyst_results.append(parse_model_output(line))
    with open(EXPLORER_RES, 'r', encoding='utf-8') as fe:
        for line in fe:
            explorer_results.append(parse_model_output(line))
    
    # 按用户组织数据
    pred_index = 0
    with open(ORIGINAL_TEST, 'r', encoding='utf-8') as fo:
        for line in fo:
            orig_data = json.loads(line)
            uid = orig_data.get('user_id')
            candidates = orig_data.get('candidates', [])
            
            current_group = []
            for candidate in candidates:
                if pred_index >= len(analyst_results) or pred_index >= len(explorer_results):
                    break
                    
                item_data = {
                    "is_label": candidate.get('is_label', 0),
                    "analyst": analyst_results[pred_index],
                    "explorer": explorer_results[pred_index]
                }
                current_group.append(item_data)
                pred_index += 1
            
            if current_group:
                test_groups.append(current_group)
            
    print(f"[INFO] 数据加载完成，共 {len(test_groups)} 个评测组。")
    return test_groups

def plot_sensitivity(x_values, hr_values, ndcg_values, param_name, save_name):
    """使用 plotly 绘制精美的双轴折线图"""
    fig = go.Figure()
    
    # 添加 HR@10 曲线
    fig.add_trace(go.Scatter(
        x=x_values,
        y=hr_values,
        name='HR@10',
        mode='lines+markers',
        marker=dict(color='red', symbol='circle'),
        line=dict(color='red', width=2)
    ))
    
    # 添加 NDCG@10 曲线
    fig.add_trace(go.Scatter(
        x=x_values,
        y=ndcg_values,
        name='NDCG@10',
        mode='lines+markers',
        marker=dict(color='blue', symbol='square'),
        line=dict(color='blue', width=2, dash='dash'),
        yaxis='y2'
    ))
    
    # 配置布局
    fig.update_layout(
        title=f'Sensitivity Analysis of {param_name} Weight',
        xaxis=dict(
            title=f'Weight of {param_name}',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='HR@10',
            tickfont=dict(color='red'),
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis2=dict(
            title='NDCG@10',
            tickfont=dict(color='blue'),
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(
            x=0.85,
            y=0.95,
            borderwidth=1
        ),
        width=800,
        height=500,
        font=dict(size=12)
    )
    
    # 保存为 PNG 图片
    full_path = os.path.join(os.getcwd(), save_name)
    pio.write_image(fig, full_path, format='png', scale=2)
    print(f"[INFO] 图表已保存为: {full_path}")

def compute_balanced_weights(test_index, test_weight):
    """
    计算权重使得总和保持为10
    test_index: 要测试的因素索引 (0:Rel, 1:Util, 2:Nov, 3:Unexp, 4:Div)
    test_weight: 测试因素的权重值
    """
    base_weights = np.array([2.5, 1.0, 1.5, 1.5, 3.5])
    result = base_weights.copy()
    result[test_index] = test_weight
    
    delta = test_weight - base_weights[test_index]
    sum_others = base_weights.sum() - base_weights[test_index]
    
    for i in range(5):
        if i != test_index:
            proportion = base_weights[i] / sum_others
            result[i] = base_weights[i] - delta * proportion
    
    return result

def main():
    test_groups = load_all_data()
    
    weight_range = np.arange(0.0, 5.1, 0.5)
    
    # ==========================================
    # 实验一：探索意外性(Unexpectedness)的影响
    # 权重之和恒定为10，其他因子按比例分摊
    # ==========================================
    print("\n[EXP] 实验一：正在测试 Unexpectedness 权重敏感性 (权重和=10)...")
    hr_unexp, ndcg_unexp = [], []
    for w in weight_range:
        current_w = compute_balanced_weights(3, w)
        hr, ndcg = evaluate_with_weights(current_w, test_groups)
        hr_unexp.append(hr)
        ndcg_unexp.append(ndcg)
        print(f"   w_une={w:.1f} | weights=[{current_w[0]:.2f},{current_w[1]:.2f},{current_w[2]:.2f},{current_w[3]:.2f},{current_w[4]:.2f}] | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")
    plot_sensitivity(weight_range, hr_unexp, ndcg_unexp, 'Unexpectedness', 'sensitivity_unexpectedness.png')

    # ==========================================
    # 实验二：探索相关性(Relevance)的影响
    # 权重之和恒定为10，其他因子按比例分摊
    # ==========================================
    print("\n[EXP] 实验二：正在测试 Relevance 权重敏感性 (权重和=10)...")
    hr_rel, ndcg_rel = [], []
    for w in weight_range:
        current_w = compute_balanced_weights(0, w)
        hr, ndcg = evaluate_with_weights(current_w, test_groups)
        hr_rel.append(hr)
        ndcg_rel.append(ndcg)
        print(f"   w_rel={w:.1f} | weights=[{current_w[0]:.2f},{current_w[1]:.2f},{current_w[2]:.2f},{current_w[3]:.2f},{current_w[4]:.2f}] | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")
    plot_sensitivity(weight_range, hr_rel, ndcg_rel, 'Relevance', 'sensitivity_relevance.png')

if __name__ == "__main__":
    main()