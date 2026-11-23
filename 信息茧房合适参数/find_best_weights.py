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
# 2. 数据对齐与预加载
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
                
                # 提取 5 维向量
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
# 3. 核心搜索算法 (修改为最低权重1.0)
# ==========================================
def run_search(all_groups, step=0.5, target_total=10.0):
    best_h10 = -1
    best_w = None
    best_metrics = (0, 0, 0)

    # 【核心修改区】: 生成全域组合：每个权重最小 1.0，总和为 10
    # 先分配 1.0 * 5 = 5.0，还剩 5.0 分可以自由分配
    min_w = 1.0 
    num_dims = 5
    residual = target_total - (min_w * num_dims) 
    
    # 构建搜索步长
    possible_extras = np.arange(0, residual + 0.1, step)
    
    print(f"🔍 开启全域网格搜索 (目标总分={target_total}, 最低权重={min_w}, 步长={step})...")
    
    count = 0
    for extra in itertools.product(possible_extras, repeat=num_dims):
        if abs(sum(extra) - residual) > 1e-7:
            continue
            
        count += 1
        current_w = np.array([min_w] * num_dims) + np.array(extra)
        
        # 计算当前指标
        h3, h5, h10 = 0, 0, 0
        for group in all_groups:
            # 计算这一组内所有 candidate 的最终分
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
        
        # 以 HR@10 作为第一优先级，HR@5 作为第二优先级
        if curr_h10 > best_h10:
            best_h10 = curr_h10
            best_w = current_w
            best_metrics = (h3/len(all_groups), h5/len(all_groups), curr_h10)
            
    return best_w, best_metrics

# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    try:
        data = load_and_align_data()
        
        # 开始搜索
        weights, res = run_search(data, step=0.5)
        
        print("\n" + "═"*50)
        print(" 🏆 困难负样本(Target相似) - 全域最优权重结果 🏆")
        print("═"*50)
        names = ["Relevance", "Utility", "Novelty", "Unexpectedness", "Diversity"]
        for i, name in enumerate(names):
            print(f" 🔹 {name:<16}: {weights[i]:.1f}")
        
        print("-" * 50)
        print(f" 🎯 HR@3  : {res[0]*100:.2f}%")
        print(f" 🎯 HR@5  : {res[1]*100:.2f}%")
        print(f" 🎯 HR@10 : {res[2]*100:.2f}% (优化目标)")
        print("═"*50)
        print("💡 在高度同质化的候选池中，期待系统能凭借 Utility 或 Unexpectedness 突围！")

    except Exception as e:
        print(f"❌ 运行失败: {e}")