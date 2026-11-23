import json
import itertools
import numpy as np
import os

# ==========================================
# 1. 配置路径 (指向你困难采样的文件夹)
# ==========================================
FOLDER = '信息茧房合适参数'
ANALYST_FILE = os.path.join(FOLDER, 'analyst_last_generated_predictions.jsonl')
EXPLORER_FILE = os.path.join(FOLDER, 'explorer_last_generated_predictions.jsonl')
ORIGINAL_FILE = os.path.join(FOLDER, 'topk_last_val_set.jsonl') 

# ==========================================
# 2. 数据对齐与预加载 (保持不变)
# ==========================================
def parse_score(data):
    try:
        if isinstance(data, str):
            return json.loads(data)
        return data
    except:
        return {}

def load_and_align_data():
    print("📦 正在加载并严格对齐数据...")
    
    with open(ANALYST_FILE, 'r', encoding='utf-8') as fa, \
         open(EXPLORER_FILE, 'r', encoding='utf-8') as fe, \
         open(ORIGINAL_FILE, 'r', encoding='utf-8') as fo:
        
        a_lines = fa.readlines()
        e_lines = fe.readlines()
        o_lines = [json.loads(l) for l in fo if l.strip()]
        
        all_groups = []
        pointer = 0
        
        for user_data in o_lines:
            candidates = user_data.get('candidates', [])
            group = []
            
            for cand in candidates:
                if pointer >= len(a_lines): break
                
                # 解析分值
                a_pred = parse_score(json.loads(a_lines[pointer]).get('predict', '{}'))
                e_pred = parse_score(json.loads(e_lines[pointer]).get('predict', '{}'))
                
                # 提取 5 维向量: [相关性, 实用性, 新颖性, 意外性, 多样性]
                vec = np.array([
                    float(a_pred.get('relevance', 3.0)),
                    float(a_pred.get('utility', 3.0)),
                    float(e_pred.get('novelty', 3.0)),
                    float(e_pred.get('unexpectedness', 3.0)),
                    float(e_pred.get('diversity', 3.0))
                ])
                
                group.append({
                    "vec": vec,
                    "is_label": cand.get('is_label', 0)
                })
                pointer += 1
            
            if group:
                all_groups.append(group)
                
    print(f"✅ 对齐完成！共计 {len(all_groups)} 组用户，总计 {pointer} 条打分记录。")
    return all_groups

# ==========================================
# 3. 核心搜索算法 (支持消融控制)
# ==========================================
def run_ablation_search(all_groups, active_dims, step=0.5, target_total=10.0, min_w=1.0):
    """
    active_dims: 激活的维度索引列表。
                 例如 [0, 1] 表示只有 Analyst (相关性, 实用性) 生效，其它为 0。
    """
    best_h10 = -1
    best_w = None
    best_metrics = (0, 0, 0)

    num_active = len(active_dims)
    residual = target_total - (min_w * num_active) 
    
    # 如果剩余权重小于0，说明基础分配都不够
    if residual < 0:
        raise ValueError(f"目标总分 {target_total} 太小，无法为 {num_active} 个维度分配最小权重 {min_w}")

    # 构建搜索步长
    possible_extras = np.arange(0, residual + 0.1, step)
    
    count = 0
    for extra in itertools.product(possible_extras, repeat=num_active):
        # 仅保留总和恰好等于 residual 的组合
        if abs(sum(extra) - residual) > 1e-7:
            continue
            
        count += 1
        
        # 构建完整的 5D 权重向量，未激活的维度默认为 0
        current_w = np.zeros(5)
        for i, dim_idx in enumerate(active_dims):
            current_w[dim_idx] = min_w + extra[i]
        
        # 计算当前指标
        h3, h5, h10 = 0, 0, 0
        for group in all_groups:
            scores = []
            for item in group:
                final_score = np.dot(item['vec'], current_w)
                scores.append((final_score, item['is_label']))
            
            # 按分值排序
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # 检查命中
            if any(s[1] == 1 for s in scores[:3]): h3 += 1
            if any(s[1] == 1 for s in scores[:5]): h5 += 1
            if any(s[1] == 1 for s in scores[:10]): h10 += 1
        
        curr_h10 = h10 / len(all_groups)
        
        # 以 HR@10 作为第一优先级
        if curr_h10 > best_h10:
            best_h10 = curr_h10
            best_w = current_w
            best_metrics = (h3/len(all_groups), h5/len(all_groups), curr_h10)
            
    return best_w, best_metrics

# ==========================================
# 4. 执行与结果输出
# ==========================================
def print_results(title, weights, res):
    print("\n" + "═"*50)
    print(f" {title} ")
    print("═"*50)
    names = ["Relevance", "Utility", "Novelty", "Unexpectedness", "Diversity"]
    for i, name in enumerate(names):
        status = "(已消融)" if weights[i] == 0 else ""
        print(f" 🔹 {name:<16}: {weights[i]:>4.1f} {status}")
    
    print("-" * 50)
    print(f" 🎯 HR@3  : {res[0]*100:.2f}%")
    print(f" 🎯 HR@5  : {res[1]*100:.2f}%")
    print(f" 🎯 HR@10 : {res[2]*100:.2f}% (优化目标)")
    print("═"*50)


if __name__ == "__main__":
    try:
        data = load_and_align_data()
        
        # 实验 1：完整系统 (Baseline)
        print("\n⏳ 正在运行 完整系统 搜索...")
        w_full, res_full = run_ablation_search(data, active_dims=[0, 1, 2, 3, 4])
        print_results("🏆 实验1: 完整多智能体系统 (Full System)", w_full, res_full)
        
        # 实验 2：消融探索者，仅保留分析师 (w/o Explorer)
        print("\n⏳ 正在运行 仅分析师 搜索 (权重分配给: 相关性, 实用性)...")
        w_analyst, res_analyst = run_ablation_search(data, active_dims=[0, 1])
        print_results("🛡️ 实验2: 仅分析师智能体 (w/o Explorer)", w_analyst, res_analyst)
        
        # 实验 3：消融分析师，仅保留探索者 (w/o Analyst)
        print("\n⏳ 正在运行 仅探索者 搜索 (权重分配给: 新颖, 意外, 多样)...")
        w_explorer, res_explorer = run_ablation_search(data, active_dims=[2, 3, 4])
        print_results("🚀 实验3: 仅探索者智能体 (w/o Analyst)", w_explorer, res_explorer)

    except Exception as e:
        print(f"❌ 运行失败: {e}")