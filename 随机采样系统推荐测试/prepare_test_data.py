import json
import os

# ==========================================
# 1. 配置路径 (指向你的测试集文件)
# ==========================================
INPUT_PATH = '随机采样合适参数/topk_final_test_set.jsonl'
BASE_DIR = '随机采样系统推荐测试'
ANALYST_OUTPUT = os.path.join(BASE_DIR, 'test_predict_analyst.json')
EXPLORER_OUTPUT = os.path.join(BASE_DIR, 'test_predict_explorer.json')

# 提示词模板 (必须与训练/验证时保持完全一致)
ANALYST_PROMPT = "You are an Analyst Agent in a serendipity recommender system. Your task is to perform deep semantic alignment and mine emotional resonance. Based on the user's history and the target item, predict the Relevance and Utility scores (1.0-5.0)."
EXPLORER_PROMPT = "You are an Explorer Agent in a serendipity recommender system. Your task is to break user information boundaries. Based on the user's history and the target item, predict the Novelty, Unexpectedness, and Diversity scores (1.0-5.0)."

def process_test_data():
    analyst_data = []
    explorer_data = []
    
    if not os.path.exists(INPUT_PATH):
        print(f"❌ 错误：找不到输入文件 {INPUT_PATH}")
        return

    print(f"🚀 开始处理最终测试集: {INPUT_PATH}")
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            if not line.strip():
                continue
            
            user_data = json.loads(line)
            user_history = user_data.get('user_history', [])
            
            # 1. 构建用户历史文本 (兼容多种标题字段)
            history_lines = []
            for item in user_history:
                title = item.get('corrected_title') or item.get('title') or 'Unknown'
                desc = item.get('description', 'No description available.')
                history_lines.append(f"《{title}》 - Description: {desc}")
            history_text = "\n".join(history_lines)
            
            # 2. 遍历候选项目并打平 (每行 20 个候选)
            candidates = user_data.get('candidates', [])
            for cand in candidates:
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
                    "output": ""  # 测试推理时输出留空
                })
                
                # 为 Explorer 构建条目
                explorer_data.append({
                    "instruction": EXPLORER_PROMPT,
                    "input": input_str,
                    "output": ""
                })
            
            line_count += 1

    # 3. 保存为 JSON 格式 (LLaMA-Factory 推理推荐格式)
    with open(ANALYST_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(analyst_data, f, ensure_ascii=False, indent=2)
    
    with open(EXPLORER_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(explorer_data, f, ensure_ascii=False, indent=2)

    print(f"✨ 处理完成！")
    print(f"📊 统计信息：")
    print(f" - 处理用户数: {line_count}")
    print(f" - Analyst 推理条目总数: {len(analyst_data)}")
    print(f" - Explorer 推理条目总数: {len(explorer_data)}")
    print(f"📁 文件已保存：")
    print(f"   - {ANALYST_OUTPUT}")
    print(f"   - {EXPLORER_OUTPUT}")

if __name__ == "__main__":
    process_test_data()