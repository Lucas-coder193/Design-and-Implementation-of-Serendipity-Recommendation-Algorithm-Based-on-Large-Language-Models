import pandas as pd
import json
import os
from tqdm import tqdm

def build_item_centric_api_tasks_full():
    print("=== 🚀 启动：全量 23950 部电影数据库逆向构建任务 ===")
    
    # ⚠️ 替换为你的原始 CSV 路径
    csv_path = r'SerenLens\SerenLens\Dataset\SerenLens_Movies.csv' 
    output_file = '标题简介补全\item_metadata_api_tasks.jsonl'
    
    MAX_REVIEWS_PER_ITEM = 8 
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    print("📊 [1/3] 正在加载和清洗数据...")
    df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
    df = df.dropna(subset=['item_id', 'review'])
    
    # === 🛡️ 绝对不能删的安全门禁：清理 2 个 CSV 错位的幽灵 ID ===
    df['item_id'] = df['item_id'].astype(str).str.strip()
    df = df[df['item_id'].str.len() <= 15] 
    df = df[~df['item_id'].str.contains(" ")] 
    # =======================================================
    
    # 🧹 移除了长度过滤！哪怕只有 1 个字的评论也全部保留！
    df['review_str'] = df['review'].astype(str).str.strip()
    df['review_len'] = df['review_str'].str.len()
    
    print(f"   ✅ 清洗后剩余有效交互数: {len(df)}")

    print("🔄 [2/3] 正在按 item_id 聚合评论并生成 Prompt...")
    tasks = []
    
    grouped = df.groupby('item_id')
    for item_id, group in tqdm(grouped, desc="Processing Items"):
        # 💡 依然按长度降序排列，保证有长评的优先取长评，全是短评的就全取短评
        top_reviews = group.sort_values('review_len', ascending=False).head(MAX_REVIEWS_PER_ITEM)['review_str'].tolist()
        
        reviews_str = ""
        for idx, rev in enumerate(top_reviews):
            clean_rev = rev[:400].replace('\r', ' ').replace('\n', ' ')
            reviews_str += f"[Review {idx+1}]: {clean_rev}\n"
            
        prompt = f"""
你是一名专业的推荐系统元数据修复专家，任务是重建丢失的【电影/电视标准数据库】。
以下是多位真实用户对【同一部】未知电影/电视剧的评论。请交叉比对这些评论中的线索（角色名、剧情、年代、演员、甚至语气和情感），逆向推导出这部影视作品的真实身份。

【用户评论集合】:
{reviews_str}

【输出要求】:
必须且只能输出一个合法的 JSON 对象，包含以下两个字段：
1. "corrected_title": 真实的影视名称。如果确定，给出标准原名；如果线索极少实在无法锁定具体名字，必须根据评论推断出一个【极其具体的影片类型或氛围】（如 "1990s Action Thriller", "Classic Romantic Drama", "Highly Acclaimed Family Movie"）。**绝对禁止在输出中使用 "Unknown", "Unidentified", "N/A" 或任何表示“未知”的消极词汇！**
2. "description": 60字以内的客观剧情简介。请务必像 IMDB 官方词条一样，客观描述时代背景、主要角色和核心冲突。如果实在没有明确的剧情线索，请结合评论的情感倾向写一段专业的客观类型介绍（如 "A critically acclaimed feature film celebrated for its captivating storytelling and strong performances."）。**同样绝对禁止出现任何“未知/无法推断”的描述，绝不可打破 IMDB 词条的客观专业感！**
"""
        tasks.append({
            "custom_id": str(item_id),
            "prompt": prompt.strip()
        })

    print(f"💾 [3/3] 正在写入 API 任务列表...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
            
    print("\n" + "="*40)
    print(f"🎉 任务构建完成！")
    print(f"🎬 共提取独立 Item 数量 (应为23950): {len(tasks):,}")
    print(f"💾 输出文件: {os.path.abspath(output_file)}")
    print("="*40)

if __name__ == "__main__":
    build_item_centric_api_tasks_full()