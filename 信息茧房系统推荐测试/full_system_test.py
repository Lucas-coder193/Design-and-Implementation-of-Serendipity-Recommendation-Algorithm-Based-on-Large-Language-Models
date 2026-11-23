import json
import numpy as np
import os
import math

# ==========================================
# 1. 配置路径与最优权重
# ==========================================
# 刚才寻优得到的最佳权重 (Sum=10)
# 顺序: [Relevance, Utility, Novelty, Unexpectedness, Diversity]
BEST_WEIGHTS = np.array([2.5, 1.0, 1.5, 1.5, 3.5])

# 文件路径
ANALYST_RES = '信息茧房系统推荐测试/test_last_analyst_generated_predictions.jsonl'
EXPLORER_RES = '信息茧房系统推荐测试/test_last_explorer_generated_predictions.jsonl'
ORIGINAL_TEST = '信息茧房合适参数/topk_last_final_test_set.jsonl'

def parse_model_output(line):
    """解析模型输出行中的 predict 字典"""
    try:
        data = json.loads(line)
        predict_content = data.get('predict', '{}')
        if isinstance(predict_content, str):
            return json.loads(predict_content)
        return predict_content
    except:
        return {}

def calculate_ndcg(ranked_list, k):
    """计算 NDCG@K"""
    dcg = 0.0
    idcg = 0.0
    total_positives = sum(1 for item in ranked_list if item['is_label'] == 1)
    
    if total_positives == 0:
        return 0.0
        
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i]['is_label'] == 1:
            dcg += 1.0 / math.log2(i + 2)
            
    for i in range(min(k, total_positives)):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg

def calculate_mrr(ranked_list):
    """计算 MRR (Mean Reciprocal Rank)"""
    for i, item in enumerate(ranked_list):
        if item['is_label'] == 1:
            return 1.0 / (i + 1)
    return 0.0

def calculate_auc(ranked_list):
    """
    计算 AUC (Area Under Curve)
    通过统计所有 (正样本, 负样本) 对中，正样本得分大于负样本得分的比例。
    """
    pos_scores = [x['score'] for x in ranked_list if x['is_label'] == 1]
    neg_scores = [x['score'] for x in ranked_list if x['is_label'] == 0]
    
    if not pos_scores or not neg_scores:
        return None # 如果没有正样本或负样本，该组无法计算AUC
        
    correct_pairs = 0.0
    total_pairs = len(pos_scores) * len(neg_scores)
    
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                correct_pairs += 1.0
            elif p == n:
                correct_pairs += 0.5 # 得分相同时记为 0.5
                
    return correct_pairs / total_pairs

def run_final_evaluation():
    if not all(os.path.exists(p) for p in [ANALYST_RES, EXPLORER_RES, ORIGINAL_TEST]):
        print("❌ 错误：请检查路径，确保推理结果和原始测试集文件存在。")
        return

    print(f"🎯 开始最终评估...")
    # 注意打印的权重也帮你同步成了代码里的真实权重
    print(f"⚖️ 使用权重分配 [Rel, Util, Nov, Unexp, Div]: {BEST_WEIGHTS.tolist()}")

    with open(ANALYST_RES, 'r', encoding='utf-8') as fa, \
         open(EXPLORER_RES, 'r', encoding='utf-8') as fe, \
         open(ORIGINAL_TEST, 'r', encoding='utf-8') as fo:
        
        a_lines = fa.readlines()
        e_lines = fe.readlines()
        
        # 指标累加器
        hr10 = 0
        ndcg10 = 0.0
        mrr = 0.0
        auc_sum = 0.0
        
        total_users = 0
        valid_auc_users = 0
        global_idx = 0

        for line_o in fo:
            if not line_o.strip(): continue
            user_data = json.loads(line_o)
            candidates = user_data.get('candidates', [])
            
            group_scores = []
            
            for cand in candidates:
                if global_idx >= len(a_lines): break
                
                scores_a = parse_model_output(a_lines[global_idx])
                scores_e = parse_model_output(e_lines[global_idx])
                
                vec = np.array([
                    float(scores_a.get('relevance', 2.5)),
                    float(scores_a.get('utility', 1.0)),
                    float(scores_e.get('novelty', 1.5)),
                    float(scores_e.get('unexpectedness', 1.5)),
                    float(scores_e.get('diversity', 3.5))
                ])
                
                final_score = np.dot(vec, BEST_WEIGHTS)
                
                group_scores.append({
                    "score": final_score,
                    "is_label": cand.get('is_label', 0)
                })
                global_idx += 1
            
            ranked = sorted(group_scores, key=lambda x: x['score'], reverse=True)
            
            # 核心 4 指标计算
            if any(it['is_label'] == 1 for it in ranked[:10]): hr10 += 1
            ndcg10 += calculate_ndcg(ranked, 10)
            mrr += calculate_mrr(ranked)
            
            auc_val = calculate_auc(ranked)
            if auc_val is not None:
                auc_sum += auc_val
                valid_auc_users += 1
                
            total_users += 1

    # ==========================================
    # 2. 打印最终结果报告 (统一为4大指标表格)
    # ==========================================
    if total_users == 0:
        return
        
    final_hr10 = hr10 / total_users
    final_ndcg10 = ndcg10 / total_users
    final_mrr = mrr / total_users
    final_auc = auc_sum / valid_auc_users if valid_auc_users > 0 else 0.0

    print("\n" + "🏁 " + "="*50)
    print(" 🚀 毕设最终测试集 (Test Set) 多维评估报告 🚀")
    print("="*52)
    print(f" 📊 测试样本总数 (User Groups): {total_users}")
    print(f" 🎞️ 推理条目总数 (Total Items): {global_idx}")
    print("-" * 52)
    print(f" | 指标名称   | 最终得分  | 指标含义简述")
    print("-" * 52)
    print(f" | HR@10      |  {final_hr10:.4f} | 召回宽容度 (是否进前10)")
    print(f" | NDCG@10    |  {final_ndcg10:.4f} | 列表排序质量 (越高越靠前)")
    print(f" | MRR        |  {final_mrr:.4f} | 首中倒数排名 (用户第一印象)")
    print(f" | AUC        |  {final_auc:.4f} | 全局排序能力 (正负样本区分度)")
    print("="*52)

if __name__ == "__main__":
    run_final_evaluation()