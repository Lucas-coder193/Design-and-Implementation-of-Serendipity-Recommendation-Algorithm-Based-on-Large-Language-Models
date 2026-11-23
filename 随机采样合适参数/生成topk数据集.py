import json
import sqlite3
import random
import os

# === 配置路径 ===
TEST_DATA_PATH = '构建电影数据集/test_movies_data.json'
DB_PATH = '随机采样合适参数/movies_demo.db'
OUTPUT_FILE = '随机采样合适参数/topk_random_test_set.jsonl' # 改回随机版本的名字

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

def build_topk_random_test_set():
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

    print("🚀 开始生成 Top-K 随机测试样本 (Random Negative Sampling)...")

    for entry in test_data:
        target = entry['window']['target']
        
        # 只保留惊喜样本 (label == 1)
        if target.get('label') != 1:
            skipped_count += 1
            continue
            
        user_id = entry['user_id']
        history = entry['window']['history']
        
        # 记录用户看过的电影和目标电影，防止把它们抽作负样本
        seen_ids = {item['item_id'] for item in history}
        seen_ids.add(target['item_id'])
        
        available_negative_ids = list(all_ids - seen_ids)
        
        if len(available_negative_ids) < 19:
            continue
    
        # === 【核心还原区】：最纯粹的随机抽取 ===
        # 直接从剩余的电影池子里随机抓取 19 部作为负样本
        selected_neg_ids = random.sample(available_negative_ids, 19)
        # ==========================================

        # 组装正样本信息
        pos_info = {
            "item_id": target['item_id'],
            "title": target.get('corrected_title', target.get('title')),
            "description": target['description'],
            "is_label": 1
        }
        
        # 组装候选池：1 正 + 19 负
        candidates_with_info = [pos_info]
        for neg_id in selected_neg_ids:
            neg_info = movie_info_map[neg_id].copy()
            neg_info["is_label"] = 0
            candidates_with_info.append(neg_info)
            
        # 打乱候选池顺序，防止模型死记硬背第一个是正样本
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

    print(f"✅ 随机样本集构建完成！")
    print(f"📊 筛选结果：保留了 {len(final_test_set)} 条惊喜样本，过滤了 {skipped_count} 条非惊喜样本。")
    print(f"📂 文件保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_topk_random_test_set()