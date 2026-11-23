import pandas as pd
import json
import time
import os

def build_enriched_user_sequences():
    print("=== 🚀 强化版用户兴趣序列构建启动 ===")
    start_time = time.time()
    
    # ================= 配置区域 =================
    # 读取我们刚刚生成的完美干净的数据集
    csv_path = r'标题简介补全\SerenLens_Movies_Enriched.csv' 
    output_path = '构建电影数据集/user_sequences_enriched.json'
    # ===========================================

    if not os.path.exists(csv_path):
        print(f"❌ 错误：找不到文件 {csv_path}")
        return

    # 1. 加载数据
    print(f"📊 [1/4] 正在加载 Enriched 强化数据...")
    df = pd.read_csv(csv_path)

    # 2. 核心清洗逻辑
    print("🧹 [2/4] 正在清洗无效数据并保留核心字段...")
    
    # 时间戳转数字
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    # ⚠️ 关键修改：过滤掉那些根本没有 title 也没有 review 的极度残缺数据
    # 但由于我们刚刚补全了所有的 title 和 description，这里其实非常安全
    df = df.dropna(subset=['timestamp', 'user_id', 'item_id', 'corrected_title']) 
    
    df['timestamp'] = df['timestamp'].astype(int)
    df['user_id'] = df['user_id'].astype(str)
    
    # 保留有效 label
    df = df[df['label'].isin([0, 1])]
    
    # （可选）保留长度过滤：剔除毫无营养的超短评论，保留用户的真实偏好表达
    df['review_len'] = df['review'].astype(str).str.len()
    df = df[df['review_len'] > 15] 
    
    print(f"   ✅ 清洗完成，剩余有效交互数: {len(df)}")

    # 3. 排序与序列构建
    print("🔄 [3/4] 正在按时间轴编排历史记录...")
    
    df_sorted = df.sort_values(by=['user_id', 'timestamp'], ascending=[True, True])
    user_sequences = {}
    grouped = df_sorted.groupby('user_id')
    
    for user_id, group in grouped:
        sequence = []
        for _, row in group.iterrows():
            interaction = {
                'item_id': str(row['item_id']),
                # 💎 核心升级：加回我们洗干净的完美电影名和简介！
                'corrected_title': str(row['corrected_title']), 
                'description': str(row['description']),         
                'timestamp': int(row['timestamp']),
                'review': str(row['review']).strip(),
                'rating': float(row['rating']),
                'label': int(row['label'])
            }
            sequence.append(interaction)
        
        user_sequences[user_id] = sequence

    # 4. 写入 JSON (保留缩进)
    print(f"💾 [4/4] 正在写入 JSON (Indent=2)...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(user_sequences, f, ensure_ascii=False, indent=2)

    # 统计
    total_users = len(user_sequences)
    total_ints = sum(len(s) for s in user_sequences.values())
    
    print("\n" + "="*40)
    print(f"🎉 Enriched 版序列处理报告")
    print(f"⏱️  耗时: {time.time() - start_time:.2f}s")
    print(f"👥 总独立用户数: {total_users:,}")
    print(f"🎬 总含元数据交互记录: {total_ints:,}")
    if total_users > 0:
        print(f"📏 平均序列长度: {total_ints/total_users:.2f}")
    print(f"💾 文件地址: {os.path.abspath(output_path)}")
    print("="*40)

if __name__ == "__main__":
    build_enriched_user_sequences()