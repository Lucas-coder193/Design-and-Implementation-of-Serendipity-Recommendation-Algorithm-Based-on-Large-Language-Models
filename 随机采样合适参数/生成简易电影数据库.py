import json
import sqlite3
import os
import re

# === 配置输入输出 ===
INPUT_FILE = '标题简介补全/item_metadata_api_results.jsonl'
OUTPUT_DIR = '随机采样合适参数'
OUTPUT_DB = os.path.join(OUTPUT_DIR, 'movies_demo.db')

def clean_llm_json(text):
    """专门用来清理大模型生成的带有 ```json ... ``` 的字符串"""
    # 移除外层的 markdown 符号
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()

def build_demo_db():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已创建目录: {OUTPUT_DIR}")

    conn = sqlite3.connect(OUTPUT_DB)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            item_id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            genre TEXT,
            poster_url TEXT
        )
    ''')

    count = 0
    error_count = 0
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到源文件 {INPUT_FILE}")
        return

    print("🚀 开始解析数据并写入数据库...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 第一层解析：解开最外层的 JSON
                outer_data = json.loads(line.strip())
                
                # 坑1：真正的 ID 叫 custom_id
                item_id = str(outer_data.get('custom_id', '')).strip()
                if not item_id:
                    continue

                # 坑2：数据藏在 response_text 里的 markdown 字符串中
                response_text = outer_data.get('response_text', '')
                clean_text = clean_llm_json(response_text)
                
                # 第二层解析：把字符串转回真实的字典
                inner_data = json.loads(clean_text)
                
                # 提取数据
                title = inner_data.get('corrected_title', 'Unknown Title')
                description = inner_data.get('description', 'No description available.')
                
                # 插入数据
                cursor.execute('''
                    INSERT OR REPLACE INTO movies (item_id, title, description)
                    VALUES (?, ?, ?)
                ''', (item_id, title, description))
                
                count += 1
            except json.JSONDecodeError:
                error_count += 1
                # print(f"⚠️ JSON解析失败跳过: {item_id}")
            except Exception as e:
                error_count += 1
                # print(f"⚠️ 其他错误跳过: {e}")

    conn.commit()
    conn.close()
    
    print(f"✅ 数据库构建完成！")
    print(f"📊 数据库位置: {os.path.abspath(OUTPUT_DB)}")
    print(f"🎬 成功导入电影: {count} 部")
    if error_count > 0:
        print(f"⚠️ 格式异常未能导入的数量: {error_count} 部")

if __name__ == "__main__":
    build_demo_db()