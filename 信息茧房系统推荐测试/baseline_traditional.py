import json
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

TEST_FILE = 'lzc/topk_last_final_test_set.jsonl'

# ==========================================
# 评估指标计算模块 (保持你的原逻辑完全不变)
# ==========================================
def calculate_ndcg(ranked_list, k):
    dcg = 0.0
    idcg = 0.0
    total_positives = sum(1 for item in ranked_list if item['is_label'] == 1)
    if total_positives == 0: return 0.0
        
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i]['is_label'] == 1:
            dcg += 1.0 / math.log2(i + 2)
            
    for i in range(min(k, total_positives)):
        idcg += 1.0 / math.log2(i + 2)
    return dcg / idcg

def calculate_mrr(ranked_list):
    for i, item in enumerate(ranked_list):
        if item['is_label'] == 1: return 1.0 / (i + 1)
    return 0.0

def calculate_auc(ranked_list):
    pos_scores = [x['score'] for x in ranked_list if x['is_label'] == 1]
    neg_scores = [x['score'] for x in ranked_list if x['is_label'] == 0]
    if not pos_scores or not neg_scores: return None
        
    correct_pairs = 0.0
    total_pairs = len(pos_scores) * len(neg_scores)
    for p in pos_scores:
        for n in neg_scores:
            if p > n: correct_pairs += 1.0
            elif p == n: correct_pairs += 0.5
    return correct_pairs / total_pairs


# ==========================================
# 基线方法 1：BM25 (经典概率检索)
# ==========================================
def run_bm25_baseline():
    print("\n🚀 启动传统基线 1: BM25 (经典稀疏概率检索)...")
    
    hr10, ndcg10, mrr, auc_sum = 0, 0.0, 0.0, 0.0
    total_users, valid_auc_users = 0, 0

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            user_data = json.loads(line.strip())
            total_users += 1

            # 收集文本并进行简单的分词 (按空格切分)
            history_docs = []
            for item in user_data['user_history']:
                text = f"{item.get('corrected_title', item.get('title', ''))} {item.get('description', '')}"
                history_docs.append(text.lower().split())

            candidate_labels = []
            candidate_docs = []
            for cand in user_data['candidates']:
                text = f"{cand['title']} {cand['description']}"
                candidate_docs.append(text.lower().split())
                candidate_labels.append(cand.get('is_label', 0))

            # 初始化 BM25 (以用户历史作为语料库)
            if not history_docs: continue
            bm25 = BM25Okapi(history_docs)

            # 计算每个候选物品对历史序列的 BM25 平均得分
            avg_similarities = []
            for cand_doc in candidate_docs:
                # get_scores 返回该候选 query 与所有历史 doc 的得分数组
                scores = bm25.get_scores(cand_doc)
                avg_similarities.append(np.mean(scores))
            
            avg_similarities = np.array(avg_similarities)
            ranked_indices = np.argsort(avg_similarities)[::-1]
            
            ranked = [{"score": float(avg_similarities[idx]), "is_label": candidate_labels[idx]} for idx in ranked_indices]
            
            # 计算指标
            if any(it['is_label'] == 1 for it in ranked[:10]): hr10 += 1
            ndcg10 += calculate_ndcg(ranked, 10)
            mrr += calculate_mrr(ranked)
            auc_val = calculate_auc(ranked)
            if auc_val is not None:
                auc_sum += auc_val
                valid_auc_users += 1

    print_report("BM25", total_users, hr10, ndcg10, mrr, auc_sum, valid_auc_users)


# ==========================================
# 基线方法 2：Sentence-BERT (经典稠密语义)
# ==========================================
def run_sbert_baseline():
    print("\n🚀 启动传统基线 2: Sentence-BERT (经典稠密语义检索)...")
    # 加载轻量级的高效多语种/英文模型
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    hr10, ndcg10, mrr, auc_sum = 0, 0.0, 0.0, 0.0
    total_users, valid_auc_users = 0, 0

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            user_data = json.loads(line.strip())
            total_users += 1

            history_docs = []
            for item in user_data['user_history']:
                history_docs.append(f"{item.get('corrected_title', item.get('title', ''))}. {item.get('description', '')}")

            candidate_docs = []
            candidate_labels = []
            for cand in user_data['candidates']:
                candidate_docs.append(f"{cand['title']}. {cand['description']}")
                candidate_labels.append(cand.get('is_label', 0))

            if not history_docs or not candidate_docs: continue

            # 向量化
            history_embeddings = model.encode(history_docs, show_progress_bar=False)
            candidate_embeddings = model.encode(candidate_docs, show_progress_bar=False)

            # 计算候选集与历史集的余弦相似度矩阵
            similarities = cosine_similarity(candidate_embeddings, history_embeddings)
            avg_similarities = np.mean(similarities, axis=1)
            
            ranked_indices = np.argsort(avg_similarities)[::-1]
            ranked = [{"score": float(avg_similarities[idx]), "is_label": candidate_labels[idx]} for idx in ranked_indices]
            
            # 计算指标
            if any(it['is_label'] == 1 for it in ranked[:10]): hr10 += 1
            ndcg10 += calculate_ndcg(ranked, 10)
            mrr += calculate_mrr(ranked)
            auc_val = calculate_auc(ranked)
            if auc_val is not None:
                auc_sum += auc_val
                valid_auc_users += 1

    print_report("Sentence-BERT", total_users, hr10, ndcg10, mrr, auc_sum, valid_auc_users)


# ==========================================
# 打印报告辅助函数
# ==========================================
def print_report(model_name, total_users, hr10, ndcg10, mrr, auc_sum, valid_auc_users):
    final_hr10 = hr10 / total_users if total_users > 0 else 0
    final_ndcg10 = ndcg10 / total_users if total_users > 0 else 0
    final_mrr = mrr / total_users if total_users > 0 else 0
    final_auc = auc_sum / valid_auc_users if valid_auc_users > 0 else 0.0

    print("="*52)
    print(f" 📊 {model_name} 测试样本总数 (User Groups): {total_users}")
    print("-" * 52)
    print(f" | 指标名称   | 最终得分  |")
    print("-" * 52)
    print(f" | HR@10      |  {final_hr10:.4f} |")
    print(f" | NDCG@10    |  {final_ndcg10:.4f} |")
    print(f" | MRR        |  {final_mrr:.4f} |")
    print(f" | AUC        |  {final_auc:.4f} |")
    print("="*52)

if __name__ == "__main__":
    run_bm25_baseline()
    run_sbert_baseline()