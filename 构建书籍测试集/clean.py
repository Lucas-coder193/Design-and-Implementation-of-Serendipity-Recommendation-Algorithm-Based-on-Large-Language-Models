import json
import os

# ==========================================
# 1. 配置文件路径
# ==========================================
# 👈 你刚刚跑出来的原始结果文件
INPUT_FILE = r"构建书籍测试集\book_serendipity_api_results.jsonl"
# 👈 清洗展平后生成的新文件（之后用这个文件做最终拼接）
OUTPUT_FILE = r"构建书籍测试集\book_serendipity_api_results_cleaned.jsonl"

# ==========================================
# 2. 开始清洗
# ==========================================
print("🧹 开始清洗并统一 JSON 格式...")
total_count = 0
fixed_count = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
     open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        if not line.strip(): 
            continue
            
        try:
            # 读取外层 JSON
            data = json.loads(line)
            total_count += 1
            
            # 读取里层的 response_text JSON
            response_json = json.loads(data['response_text'])
            
            # 🎯 核心逻辑：如果发现嵌套了 "serendipity_scores"，就把内层提出来
            if "serendipity_scores" in response_json:
                response_json = response_json["serendipity_scores"]
                fixed_count += 1
            
            # 把清洗后的标准字典重新转成字符串，塞回 response_text
            data['response_text'] = json.dumps(response_json, ensure_ascii=False)
            
            # 写入新文件
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            
        except Exception as e:
            print(f"❌ 警告：行解析出错 {e} -> {line.strip()}")

print("\n✅ 清洗完成！")
print(f"📊 总共处理了 {total_count} 条记录。")
print(f"🔧 成功将 {fixed_count} 条带有 'serendipity_scores' 的嵌套格式展平！")
print(f"🚀 请在后续步骤中使用完美对齐的新文件：{OUTPUT_FILE}")