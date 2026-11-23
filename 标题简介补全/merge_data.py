import pandas as pd
import json
import re

def clean_llm_json_text(text):
    """
    清洗大模型返回的带有 markdown 格式的 JSON 字符串
    提取出纯正的 JSON 字典对象
    """
    # 使用正则表达式提取 { ... } 之间的内容，忽略外面的 ```json 和 ```
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        clean_text = match.group(0)
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return None
    return None

def main():
    # 1. 设置文件路径（请根据你的实际文件名修改）
    jsonl_file_path = 'item_metadata_api_results.jsonl'
    original_csv_path = 'SerenLens\SerenLens\Dataset\SerenLens_Movies.csv'
    output_csv_path = '找真实名称\SerenLens_Movies_Enriched.csv'

    print("🚀 步骤 1: 正在解析大模型生成的 JSONL 字典库...")
    movie_metadata_dict = {}
    success_count = 0
    error_count = 0
    
    # 逐行读取 JSONL
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            
            data = json.loads(line)
            item_id = data['custom_id']
            response_text = data['response_text']
            
            # 清洗并解析大模型生成的文本
            parsed_info = clean_llm_json_text(response_text)
            
            if parsed_info:
                movie_metadata_dict[item_id] = {
                    'corrected_title': parsed_info.get('corrected_title', 'Unknown Title'),
                    'description': parsed_info.get('description', 'No description available')
                }
                success_count += 1
            else:
                # 记录解析失败的 item_id（极少数情况）
                movie_metadata_dict[item_id] = {
                    'corrected_title': 'Parsing Error',
                    'description': 'Parsing Error'
                }
                error_count += 1
                
    print(f"✅ 字典库构建完成！成功解析: {success_count} 条，解析失败: {error_count} 条。")

    print("🚀 步骤 2: 正在加载原始 CSV 并进行数据缝合 (Merge)...")
    # 加载原始数据
    df = pd.read_csv(original_csv_path)
    
    # 使用 map 函数，根据 item_id 快速匹配对应的片名和简介
    # 如果某个 item_id 在字典里没找到，就填入空值
    df['corrected_title'] = df['item_id'].map(
        lambda x: movie_metadata_dict.get(x, {}).get('corrected_title', None)
    )
    df['description'] = df['item_id'].map(
        lambda x: movie_metadata_dict.get(x, {}).get('description', None)
    )

    print("🚀 步骤 3: 正在保存全新的强化版数据集...")
    # 保存为新的 CSV 文件
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    
    print(f"🎉 缝合大功告成！全新数据集已保存为: {output_csv_path}")
    print("\n新数据前 3 行预览：")
    # 打印前几行展示一下效果（只展示核心列）
    print(df[['user_id', 'item_id', 'corrected_title', 'description']].head(3))

if __name__ == "__main__":
    main()