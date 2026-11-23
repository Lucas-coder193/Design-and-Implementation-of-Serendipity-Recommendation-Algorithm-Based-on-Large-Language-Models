import json
import os

# 文件路径
TEST_SET_FILE = 'books_test_set_1_19.json'
SERENDIPITY_RESULTS_FILE = 'book_serendipity_api_results.jsonl'
METADATA_RESULTS_FILE = 'book_metadata_api_results.jsonl'

# 输出文件
EXPLORER_OUTPUT_FILE = 'books_test_set_explorer.json'
ANALYST_OUTPUT_FILE = 'books_test_set_analyst.json'

def load_test_set():
    """加载测试集数据"""
    with open(TEST_SET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_serendipity_results():
    """加载惊喜度评分结果"""
    scores_map = {}
    with open(SERENDIPITY_RESULTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            custom_id = data['custom_id']
            response_text = json.loads(data['response_text'])
            scores_map[custom_id] = response_text
    return scores_map

def load_metadata_results():
    """加载书籍元数据结果"""
    metadata_map = {}
    with open(METADATA_RESULTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            book_id = data['custom_id']
            try:
                response_text = json.loads(data['response_text'])
                # 处理可能的列表情况
                if isinstance(response_text, list):
                    # 取第一个结果
                    if response_text:
                        metadata_map[book_id] = response_text[0]
                else:
                    metadata_map[book_id] = response_text
            except json.JSONDecodeError:
                # 如果解析失败，跳过
                pass
    return metadata_map

def get_book_info(book_id, metadata_map):
    """获取书籍信息"""
    if book_id in metadata_map:
        book_info = metadata_map[book_id]
        title = book_info.get('title', 'Unknown Title')
        description = book_info.get('description', 'No description available')
        return title, description
    return 'Unknown Title', 'No description available'

def generate_test_set():
    """生成测试集"""
    test_set = load_test_set()
    scores_map = load_serendipity_results()
    metadata_map = load_metadata_results()
    
    explorer_data = []
    analyst_data = []
    
    for item in test_set:
        user_id = item['user_id']
        history = item['history']
        target = item['target']
        
        # 构建用户历史文本
        history_text = ""
        for book_id in history:
            title, description = get_book_info(book_id, metadata_map)
            history_text += f"《{title}》 - Description: {description}\n"
        
        # 获取目标书籍信息
        target_title, target_description = get_book_info(target, metadata_map)
        
        # 构建输入文本
        input_text = f"User History:\n{history_text.strip()}\n\nTarget Item: 《{target_title}》\nTarget Description: {target_description}"
        
        # 生成 Explorer Agent 的输出
        scores_key = f"{user_id}_{target}"
        explorer_output = {
            "novelty": 3.0,
            "unexpectedness": 3.0,
            "diversity": 3.0
        }
        
        if scores_key in scores_map:
            scores = scores_map[scores_key]
            # 确保 scores 是字典
            if isinstance(scores, dict):
                explorer_output = {
                    "novelty": scores.get("novelty", 3.0),
                    "unexpectedness": scores.get("unexpectedness", 3.0),
                    "diversity": scores.get("diversity", 3.0)
                }
        
        # 生成 Analyst Agent 的输出
        analyst_output = {
            "relevance": 3.0,
            "usefulness": 3.0
        }
        
        if scores_key in scores_map:
            scores = scores_map[scores_key]
            # 确保 scores 是字典
            if isinstance(scores, dict):
                analyst_output = {
                    "relevance": scores.get("relevance", 3.0),
                    "usefulness": scores.get("utility", 3.0)
                }
        
        # 添加到结果列表
        explorer_item = {
            "instruction": "You are an Explorer Agent in a serendipity recommender system. Your task is to break user information boundaries. Based on the user's history and the target item, predict the Novelty, Unexpectedness, and Diversity scores (1.0-5.0).",
            "input": input_text,
            "output": json.dumps(explorer_output, ensure_ascii=False)
        }
        
        analyst_item = {
            "instruction": "You are an Analyst Agent in a serendipity recommender system. Your task is to evaluate user information boundaries. Based on the user's history and the target item, predict the Relevance and Usefulness scores (1.0-5.0).",
            "input": input_text,
            "output": json.dumps(analyst_output, ensure_ascii=False)
        }
        
        explorer_data.append(explorer_item)
        analyst_data.append(analyst_item)
    
    # 保存结果
    with open(EXPLORER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(explorer_data, f, ensure_ascii=False, indent=2)
    
    with open(ANALYST_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(analyst_data, f, ensure_ascii=False, indent=2)
    
    print(f"生成完成！")
    print(f"Explorer 测试集已保存到: {EXPLORER_OUTPUT_FILE}")
    print(f"Analyst 测试集已保存到: {ANALYST_OUTPUT_FILE}")

if __name__ == "__main__":
    generate_test_set()