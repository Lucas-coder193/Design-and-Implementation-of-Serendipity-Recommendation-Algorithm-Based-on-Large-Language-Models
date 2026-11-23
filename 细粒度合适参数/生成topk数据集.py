import json
import sqlite3
import random
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === 配置路径 ===
TEST_DATA_PATH = '构建电影数据集/test_movies_data.json'
DB_PATH = '细粒度合适参数/movies_demo.db'
OUTPUT_FILE = '细粒度合适参数/topk_hard_data_set.jsonl' # 改个名字，标记为 hard 难度

def get_movie_info_map(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT item_id, title, description FROM movies")
    rows = cursor.fetchall()
    
    movie_map = {}
    for row in rows:
        movie_map[str(row[0])] = {
            "item_id": str(row[0]),
            "title": row[1],
            "description": row[2]
        }
    conn.close()
    return movie_map

def build_topk_hard_test_set():
    if not os.path.exists(DB_PATH):
        print(f"❌ 错误：找不到数据库 {DB_PATH}")
        return
    
    movie_info_map = get_movie_info_map(DB_PATH)
    all_ids = set(movie_info_map.keys())
    print(f"🎬 数据库加载完成，共有 {len(all_ids)} 部电影信息。")

    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    final_test_set = []
    skipped_count = 0

    print("🚀 开始生成 Top-K 困难测试样本 (Hard Negative Mining)...")
    
    # 初始化一个 TF-IDF 向量化器
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    for entry in test_data:
        target = entry['window']['target']
        
        if target.get('label') != 1:
            skipped_count += 1
            continue
            
        user_id = entry['user_id']
        history = entry['window']['history']
        
        seen_ids = {item['item_id'] for item in history}
        seen_ids.add(target['item_id'])
        
        available_negative_ids = list(all_ids - seen_ids)
        
        if len(available_negative_ids) < 19:
            continue
    
        # === 【核心改造区】：困难负样本挖掘 ===
        # 1. 准备文本：目标电影的文本
        target_text = f"{target.get('corrected_title', target.get('title'))}. {target['description']}"
        
        # 2. 准备文本：所有可用负样本的文本
        neg_texts = [f"{movie_info_map[nid]['title']}. {movie_info_map[nid]['description']}" for nid in available_negative_ids]
        
        # 3. 将 target 和所有 neg 放在一起计算 TF-IDF
        all_texts = [target_text] + neg_texts
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            # 计算 target (第0个) 与所有负样本 (第1个及之后) 的余弦相似度
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # 获取相似度最高的前 19 个负样本的索引
            top_19_indices = np.argsort(similarities)[::-1][:19]
            selected_neg_ids = [available_negative_ids[i] for i in top_19_indices]
        except ValueError:
            # 极端情况：文本全空，回退到随机采样
            selected_neg_ids = random.sample(available_negative_ids, 19)
        # ==========================================

        # 正样本信息
        pos_info = {
            "item_id": target['item_id'],
            "title": target.get('corrected_title', target.get('title')),
            "description": target['description'],
            "is_label": 1
        }
        
        candidates_with_info = [pos_info]
        for neg_id in selected_neg_ids:
            neg_info = movie_info_map[neg_id].copy()
            neg_info["is_label"] = 0
            candidates_with_info.append(neg_info)
            
        random.shuffle(candidates_with_info)
        
        test_packet = {
            "user_id": user_id,
            "user_history": history,
            "candidates": candidates_with_info
        }
        
        final_test_set.append(test_packet)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in final_test_set:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ 困难样本集构建完成！")
    print(f"📊 筛选结果：保留了 {len(final_test_set)} 条惊喜样本，过滤了 {skipped_count} 条非惊喜样本。")
    print(f"📂 文件保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_topk_hard_test_set()