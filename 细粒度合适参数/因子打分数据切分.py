import json
import os

# 配置路径
INPUT_PATH = '细粒度合适参数/topk_hard_val_set.jsonl'
BASE_DIR = '细粒度合适参数'
ANALYST_OUTPUT = os.path.join(BASE_DIR, 'val_hard_predict_analyst.json')
EXPLORER_OUTPUT = os.path.join(BASE_DIR, 'val_hard_predict_explorer.json')

# 提示词模板
ANALYST_PROMPT = "You are an Analyst Agent in a serendipity recommender system. Your task is to perform deep semantic alignment and mine emotional resonance. Based on the user's history and the target item, predict the Relevance and Utility scores (1.0-5.0)."
EXPLORER_PROMPT = "You are an Explorer Agent in a serendipity recommender system. Your task is to break user information boundaries. Based on the user's history and the target item, predict the Novelty, Unexpectedness, and Diversity scores (1.0-5.0)."

def process_data():
    analyst_data = []
    explorer_data = []
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误：找不到输入文件 {INPUT_PATH}")
        return

    print(f"🚀 开始读取并解析: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            user_data = json.loads(line)
            user_history = user_data.get('user_history', [])
            
            # --- 修改点 1：构建用户历史文本 (兼容 corrected_title 和 title) ---
            history_lines = []
            for item in user_history:
                # 按照优先级获取标题：corrected_title > title > Unknown
                title = item.get('corrected_title') or item.get('title') or 'Unknown'
                desc = item.get('description', 'No description available.')
                history_lines.append(f"《{title}》 - Description: {desc}")
            history_text = "\n".join(history_lines)
            
            # 遍历候选项目
            for cand in user_data.get('candidates', []):
                # --- 修改点 2：候选集标题获取 (同理兼容) ---
                target_title = cand.get('corrected_title') or cand.get('title') or 'Unknown'
                target_desc = cand.get('description', 'No description available.')
                
                input_str = (
                    f"User History:\n{history_text}\n\n"
                    f"Target Item: 《{target_title}》\n"
                    f"Target Description: {target_desc}"
                )
                
                # 为 Analyst 构建条目
                analyst_data.append({
                    "instruction": ANALYST_PROMPT,
                    "input": input_str,
                    "output": "" 
                })
                
                # 为 Explorer 构建条目
                explorer_data.append({
                    "instruction": EXPLORER_PROMPT,
                    "input": input_str,
                    "output": ""
                })

    # 保存文件
    with open(ANALYST_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(analyst_data, f, ensure_ascii=False, indent=2)
    
    with open(EXPLORER_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(explorer_data, f, ensure_ascii=False, indent=2)

    print(f"✨ 处理完成！")
    print(f"📊 统计信息：")
    print(f" - Analyst 条目数: {len(analyst_data)}")
    print(f" - Explorer 条目数: {len(explorer_data)}")
    print(f"📁 文件已保存至 {BASE_DIR} 目录下。")

if __name__ == "__main__":
    process_data()