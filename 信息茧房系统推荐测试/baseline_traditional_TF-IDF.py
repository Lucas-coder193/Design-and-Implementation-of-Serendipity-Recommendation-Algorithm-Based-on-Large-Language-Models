import json
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEST_FILE = 'lzc/topk_hard_final_test_set.jsonl'

def calculate_ndcg(ranked_list, k):
    """计算 NDCG@K"""
    dcg = 0.0
    idcg = 0.0
    total_positives = sum(1 for item in ranked_list if item['is_label'] == 1)
    
    if total_positives == 0:
        return 0.0
        
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i]['is_label'] == 1:
            dcg += 1.0 / math.log2(i + 2)  # i 从 0 开始，公式为 log2(i + 1 + 1)
            
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
    """计算 AUC (Area Under Curve)"""
    pos_scores = [x['score'] for x in ranked_list if x['is_label'] == 1]
    neg_scores = [x['score'] for x in ranked_list if x['is_label'] == 0]
    
    if not pos_scores or not neg_scores:
        return None
        
    correct_pairs = 0.0
    total_pairs = len(pos_scores) * len(neg_scores)
    
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                correct_pairs += 1.0
            elif p == n:
                correct_pairs += 0.5
                
    return correct_pairs / total_pairs

def run_fair_traditional_baseline():
    print("🚀 启动传统基线模型 (TF-IDF 纯内容推荐)...")
    print("⏳ 正在计算 HR@10, NDCG@10, MRR, AUC...")

    hr10 = 0
    ndcg10 = 0.0
    mrr = 0.0
    auc_sum = 0.0
    
    total_users = 0
    valid_auc_users = 0

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            user_data = json.loads(line.strip())
            
            # 保持原始逻辑：上来就先 +1
            total_users += 1

            # 1. 收集历史文本
            history_docs = []
            for item in user_data['user_history']:
                title = item.get('corrected_title', item.get('title', ''))
                desc = item.get('description', '')
                history_docs.append(f"{title}. {desc}")

            # 2. 收集候选文本和标签
            candidate_docs = []
            candidate_labels = []
            for cand in user_data['candidates']:
                title = cand['title']
                desc = cand['description']
                candidate_docs.append(f"{title}. {desc}")
                candidate_labels.append(cand.get('is_label', 0))

            # 3. TF-IDF 向量化
            vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

            try:
                all_docs = history_docs + candidate_docs
                tfidf_matrix = vectorizer.fit_transform(all_docs)

                history_matrix = tfidf_matrix[:len(history_docs)]
                candidate_matrix = tfidf_matrix[len(history_docs):]

                # 4. 计算余弦相似度并完全恢复你原来的 NumPy 排序规则
                similarities = cosine_similarity(candidate_matrix, history_matrix)
                avg_similarities = np.mean(similarities, axis=1)
                
                # 💥 核心修复：完全使用原始的 numpy 倒序逻辑！
                ranked_indices = np.argsort(avg_similarities)[::-1]
                
                # 组装成标准列表以供评估函数调用 (严格按照 ranked_indices 的顺序)
                ranked = []
                for idx in ranked_indices:
                    ranked.append({
                        "score": float(avg_similarities[idx]),
                        "is_label": candidate_labels[idx]
                    })
                
                # 5. 计算 4 大核心指标
                if any(it['is_label'] == 1 for it in ranked[:10]): 
                    hr10 += 1
                ndcg10 += calculate_ndcg(ranked, 10)
                mrr += calculate_mrr(ranked)
                
                auc_val = calculate_auc(ranked)
                if auc_val is not None:
                    auc_sum += auc_val
                    valid_auc_users += 1

            except ValueError:
                # 保持原始逻辑：如果抛出异常，这一轮的指标就相当于全加了 0
                continue

    # ==========================================
    # 打印最终结果报告
    # ==========================================
    final_hr10 = hr10 / total_users if total_users > 0 else 0
    final_ndcg10 = ndcg10 / total_users if total_users > 0 else 0
    final_mrr = mrr / total_users if total_users > 0 else 0
    final_auc = auc_sum / valid_auc_users if valid_auc_users > 0 else 0.0

    print("\n" + "🏁 " + "="*50)
    print(" 🚀 传统基线模型 (TF-IDF) 多维评估报告 🚀")
    print("="*52)
    print(f" 📊 测试样本总数 (User Groups): {total_users}")
    print("-" * 52)
    print(f" | 指标名称   | 最终得分  | 指标含义简述")
    print("-" * 52)
    print(f" | HR@10      |  {final_hr10:.4f} | 召回宽容度 (是否进前10)")
    print(f" | NDCG@10    |  {final_ndcg10:.4f} | 列表排序质量 (越高越靠前)")
    print(f" | MRR        |  {final_mrr:.4f} | 首中倒数排名 (用户第一印象)")
    print(f" | AUC        |  {final_auc:.4f} | 全局排序能力 (正负样本区分度)")
    print("="*52)

if __name__ == "__main__":
    run_fair_traditional_baseline()