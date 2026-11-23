import json
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. 核心路径配置 (请确保路径与你本地一致)
# ==========================================
TEST_SET_PATH = r"构建书籍测试集\books_test_set_1_19.json"
CSV_PATH = r"SerenLens\SerenLens\Dataset\SerenLens_Books.csv"
OUTPUT_API_FILE = r"构建书籍测试集\book_api_prompts.jsonl"

MAX_REVIEWS_PER_BOOK = 10 # 限制每本书最多抽取 10 条评论，防止 Token 超出 API 限制

# ==========================================
# 2. 提取测试集中的所有唯一商品 ID (已修复 target 遗漏问题)
# ==========================================
print("正在读取测试集并提取所有唯一的书籍 ID...")
with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

unique_item_ids = set()
for entry in test_data:
    # 1. 提取历史记录中的书籍 (history 是列表，使用 update)
    if "history" in entry:
        unique_item_ids.update(entry["history"])
        
    # 2. 提取要预测的目标书籍 (target 是字符串，必须使用 add)
    if "target" in entry:
        unique_item_ids.add(entry["target"])

print(f"共提取到 {len(unique_item_ids)} 本需要修复元数据的独立书籍。")

# ==========================================
# 3. 从 CSV 中提取这些书籍对应的所有评论
# ==========================================
print("正在从 CSV 中读取并聚合读者评论...")
df = pd.read_csv(CSV_PATH)
df_filtered = df[df['item_id'].isin(unique_item_ids)]

# 按 item_id 分组，提取非空的 review 组成列表
reviews_dict = df_filtered.groupby('item_id')['review'].apply(
    lambda x: [str(r) for r in x.dropna().tolist()]
).to_dict()

# ==========================================
# 4. Prompt 模板 (极其直接，只要书名和简介)
# ==========================================
PROMPT_TEMPLATE = """你是一个专业的图书信息提取助手。请阅读以下多位真实读者对【同一本书】的评论，提取或总结出这本书的书名和内容简介。

【读者评论集合】:
{reviews_text}

【输出要求】:
请务必且只能输出一个合法的 JSON 对象，包含以下两个字段：
1. "title": 书籍名称。请根据评论中的线索提取真实书名（中英文均可）。如果评论中实在没有提到明确的书名，请根据评论内容总结一个最贴切的书名（例如："Guide to [主题]" 或 "The Story of [主角]"）。
2. "description": 书籍简介。请用 60 字以内的客观语言，概括这本书的核心剧情、主题思想或主要内容。

注意：绝对不能输出多余的解释，只能输出纯 JSON 格式！"""

# ==========================================
# 5. 生成供 API 调用的 JSONL 数据
# ==========================================
print("正在生成 API 请求 Prompt...")
api_requests = []

for item_id in tqdm(unique_item_ids):
    # 获取该书的评论，如果没有则给一个默认空提示
    all_reviews = reviews_dict.get(item_id, [])
    
    # 截取前 MAX_REVIEWS_PER_BOOK 条评论
    selected_reviews = all_reviews[:MAX_REVIEWS_PER_BOOK]
    
    if not selected_reviews:
        reviews_formatted = "[Review 1]: (No text reviews available for this item, please provide a general description based on common highly-rated books.)"
    else:
        # 拼接成 [Review 1]: xxx \n [Review 2]: xxx 的格式
        reviews_formatted = "\n".join([f"[Review {i+1}]: {rev}" for i, rev in enumerate(selected_reviews)])
        
    # 填充模板
    final_prompt = PROMPT_TEMPLATE.replace("{reviews_text}", reviews_formatted)
    
    # 组装一条请求数据
    request_obj = {
        "custom_id": item_id,
        "prompt": final_prompt
    }
    api_requests.append(request_obj)

# 写入 JSONL 文件 (每行一个 JSON，方便批量读取/并发请求)
with open(OUTPUT_API_FILE, "w", encoding="utf-8") as f:
    for req in api_requests:
        f.write(json.dumps(req, ensure_ascii=False) + "\n")

print(f"\n大功告成！已将 {len(api_requests)} 条 API Prompt 写入 {OUTPUT_API_FILE}。")