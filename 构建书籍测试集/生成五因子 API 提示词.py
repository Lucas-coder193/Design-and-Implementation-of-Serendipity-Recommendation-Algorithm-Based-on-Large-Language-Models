import json
import pandas as pd
import os
from tqdm import tqdm

# ==========================================
# 1. 配置文件路径 (请确保路径正确)
# ==========================================
TEST_SET_FILE = r"构建书籍测试集\books_test_set_1_19.json"             
METADATA_FILE = r"构建书籍测试集\book_metadata_api_results.jsonl"       
CSV_PATH = r"SerenLens\SerenLens\Dataset\SerenLens_Books.csv"           
OUTPUT_PROMPT_FILE = r"构建书籍测试集\book_serendipity_api_prompts.jsonl" 

# ==========================================
# 2. 加载数据源 A：书籍标题与简介 (增加 JSON 清洗和统一小写)
# ==========================================
print("1. 正在加载书籍元数据字典...")
metadata_dict = {}
error_count = 0

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # 统一转成字符串、去除空格、转小写，防止 X 和 x 不匹配
                    item_id = str(data['custom_id']).strip().lower() 
                    
                    # 提取 response_text 并清洗潜在的 markdown 符号
                    raw_text = data['response_text'].strip()
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[7:]
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]
                        
                    meta_json = json.loads(raw_text.strip())
                    metadata_dict[item_id] = {
                        "title": meta_json.get("title", "Unknown Book"),
                        "description": meta_json.get("description", "No description available.")
                    }
                except Exception as e:
                    error_count += 1
                    # 不再使用 pass，如果有解析失败的，至少能看到报错原因
                    # print(f"解析错误 ID {data.get('custom_id')}: {e}") 

print(f"-> 成功提取了 {len(metadata_dict)} 本书的名称和简介！(解析失败: {error_count} 条)")

# ==========================================
# 3. 加载数据源 B：用户真实评论 (强制指定类型为 string 防止吞 0)
# ==========================================
print("2. 正在读取 SerenLens 原始 CSV 并提取评论...")
# 极其关键：强制 user_id 和 item_id 为字符串，防止 "0061148512" 变成 "61148512"
df = pd.read_csv(CSV_PATH, dtype={'user_id': str, 'item_id': str})
reviews_dict = {}

for _, row in tqdm(df.iterrows(), total=len(df), desc="构建评论字典"):
    # 同样统一去空格、转小写
    u_id = str(row['user_id']).strip().lower()
    i_id = str(row['item_id']).strip().lower()
    
    review_text = str(row['review']) if pd.notna(row['review']) else "该用户未留下详细文本评论。"
    reviews_dict[(u_id, i_id)] = review_text

# ==========================================
# 4. 惊喜度五因子 Prompt 模板
# ==========================================
FACTOR_PROMPT_TEMPLATE = """你是一位专业的推荐系统与心理学分析专家。请结合【书籍信息】、【用户评论】以及【用户的真实惊喜反馈】，逆向评估该次推荐在“惊喜度（Serendipity）五大核心维度”上的量化表现。

【1. 客观书籍信息】
书名：{title}
简介：{description}

【2. 主观评价反馈】
该用户的真实评论：“{review}”
该用户的最终惊喜度打分标签（Label）：{label_text}

【任务要求】
请结合上述客观内容与主观反馈，给出该推荐在以下五个维度上的打分（1.0 到 5.0，保留一位小数）。
1. "unexpectedness" (意外性)：该书是否偏离了读者的常规口味？
2. "novelty" (新颖性)：题材、观点或表达方式对该读者是否具有新鲜感？
3. "relevance" (相关性)：书籍内容是否依然命中了读者的潜在兴趣？
4. "utility" (有用性)：读者是否从阅读中获得了实际收益、知识或高质量的体验？
5. "diversity" (多样性)：该书在题材或风格上，与读者过往阅读习惯的差异化程度？

必须且只能输出一个合法的 JSON 对象，严禁包含任何其他解释性文字！"""

# ==========================================
# 5. 遍历数据源 C (测试集) 融合生成任务
# ==========================================
print("3. 正在三表融合，生成 API 请求 Prompt...")
with open(TEST_SET_FILE, 'r', encoding='utf-8') as f:
    test_cases = json.load(f)

api_tasks = []

for case in tqdm(test_cases, desc="生成任务进度"):
    # 测试集的数据也统一转小写进行匹配
    u_id = str(case['user_id']).strip().lower()
    t_id = str(case['target']).strip().lower()
    
    # 注意：确保 test_set 里的 key 是 expected_label
    label = case.get('expected_label', 0.0) 
    
    label_text = "惊喜 (Unexpected & Surprising, Label=1.0)" if label == 1.0 else "平淡/符合预期 (Expected/Not Surprising, Label=0.0)"
    
    # 提取 元数据，如果没匹配上会使用 Unknown
    meta = metadata_dict.get(t_id, {"title": "Unknown", "description": "Unknown"})
    
    review = reviews_dict.get((u_id, t_id), "该用户未留下详细文本评论。")
    
    final_prompt = FACTOR_PROMPT_TEMPLATE.format(
        title=meta['title'],
        description=meta['description'],
        review=review,
        label_text=label_text
    )
    
    api_tasks.append({
        "custom_id": f"{u_id}_{t_id}", 
        "prompt": final_prompt
    })

# ==========================================
# 6. 保存为 JSONL
# ==========================================
with open(OUTPUT_PROMPT_FILE, 'w', encoding='utf-8') as f_out:
    for task in api_tasks:
        f_out.write(json.dumps(task, ensure_ascii=False) + '\n')

print(f"\n✅ 数据链路打通！成功融合生成了 {len(api_tasks)} 条五因子评估任务！")
print(f"文件已保存至: {OUTPUT_PROMPT_FILE}")