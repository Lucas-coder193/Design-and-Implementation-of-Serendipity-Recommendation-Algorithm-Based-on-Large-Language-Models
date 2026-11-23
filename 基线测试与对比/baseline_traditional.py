import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 文件路径：你的 Top-K 测试集
TEST_FILE = 'lzc/topk_final_test_set.jsonl'

def run_fair_traditional_baseline():
    print("🚀 启动传统基线模型 (TF-IDF 纯内容推荐) - 绝对公平对比版...")
    print("⏳ 正在处理全量测试集并计算 HR@10...")
    
    hits_at_10 = 0
    total_users = 0
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            user_data = json.loads(line.strip())
            total_users += 1
            
            # 1. 收集用户历史的文本描述（严格对齐 LLM：只用 Title 和 Description）
            history_docs = []
            for item in user_data['user_history']:
                title = item.get('corrected_title', item.get('title', ''))
                desc = item.get('description', '')
                # 【关键点】：这里绝对不加 review，保证和微调大模型输入的信息量完全一致
                combined_text = f"{title}. {desc}"
                history_docs.append(combined_text)
            
            # 2. 收集候选集的文本描述及标签
            candidate_docs = []
            candidate_labels = []
            for cand in user_data['candidates']:
                title = cand['title']
                desc = cand['description']
                candidate_docs.append(f"{title}. {desc}")
                candidate_labels.append(cand.get('is_label', 0)) # 1 表示真实的惊喜目标
            
            # 3. 使用传统的 TF-IDF 向量化
            vectorizer = TfidfVectorizer(stop_words='english', max_features=10000) 
            
            try:
                # 将历史和候选集放在一起训练词频矩阵
                all_docs = history_docs + candidate_docs
                tfidf_matrix = vectorizer.fit_transform(all_docs)
                
                # 分离历史矩阵和候选矩阵
                history_matrix = tfidf_matrix[:len(history_docs)]
                candidate_matrix = tfidf_matrix[len(history_docs):]
                
                # 4. 计算候选电影与用户历史的平均余弦相似度
                similarities = cosine_similarity(candidate_matrix, history_matrix)
                avg_similarities = np.mean(similarities, axis=1)
                
                # 5. 排序找 Top-10
                ranked_indices = np.argsort(avg_similarities)[::-1]
                top_10_indices = ranked_indices[:10]
                
                # 6. 判断命中 (HR@10)
                for idx in top_10_indices:
                    if candidate_labels[idx] == 1:
                        hits_at_10 += 1
                        break 
                        
            except ValueError:
                # 极端情况：文本全空导致报错，跳过
                continue

    # 计算最终指标
    hr_10 = hits_at_10 / total_users if total_users > 0 else 0
    
    print("\n" + "="*50)
    print("🏆 传统基线方法 (TF-IDF 内容推荐) 测试完成！")
    print(f"👥 测试总用户数: {total_users}")
    print(f"🎯 命中用户数 (Top-10): {hits_at_10}")
    print(f"📊 传统 Baseline HR@10: {hr_10:.4f} ({hr_10*100:.2f}%)")
    print("="*50)
    print("💡 请将此 HR@10 数据与你双智能体的 HR@10 (54.84%) 放在论文的同一张表格中对比！")

if __name__ == "__main__":
    run_fair_traditional_baseline()