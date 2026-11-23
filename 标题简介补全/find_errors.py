import json
import re

def clean_llm_json_text(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None
    return None

jsonl_file_path = 'item_metadata_api_results.jsonl'

print("🔍 正在抓捕解析失败的 15 条数据...\n" + "="*50)

error_count = 0
with open(jsonl_file_path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        data = json.loads(line)
        item_id = data['custom_id']
        response_text = data['response_text']
        
        parsed_info = clean_llm_json_text(response_text)
        
        if not parsed_info:
            error_count += 1
            print(f"❌ 异常 ID: {item_id}")
            print(f"🤖 大模型原始回复:\n{response_text}")
            print("-" * 50)

print(f"🎯 抓捕完毕，共找到 {error_count} 条。")