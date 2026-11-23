import json
import time
import os
import requests
import urllib3
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 禁用安全警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === ⚙️ 核心配置 ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise RuntimeError("请先设置环境变量 OPENROUTER_API_KEY。")
MODEL_NAME = "google/gemini-2.0-flash-001"
INPUT_TASK_FILE = "标题简介补全\item_metadata_api_tasks.jsonl"   # 上一步生成的 Item 聚合任务文件
OUTPUT_RESULT_FILE = "标题简介补全\item_metadata_api_results.jsonl"

# === 🌐 代理配置 ===
PROXY_URL = os.environ.get("OPENROUTER_PROXY") or None
# PROXY_URL = None 

# === 🚀 初始化 Session ===
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))
if PROXY_URL:
    session.proxies = {"http": PROXY_URL, "https": PROXY_URL}

def call_gemini_for_item(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1, # 保持极低温度，保证逆向推导的客观性和稳定性
    }
    
    for attempt in range(3):
        try:
            # Item 任务只生成片名和简介，输出极短，20秒超时足够
            response = session.post(
                url, headers=headers, data=json.dumps(payload), 
                timeout=20, verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
            elif response.status_code == 429:
                time.sleep(5) # 触发频控，睡 5 秒
            else:
                pass 
                
        except Exception as e:
            time.sleep(1)
            
    return None

def run_item_metadata_extraction():
    print(f"=== 🚀 Stage 1: Item 元数据盲猜提取启动 ===")
    
    if not os.path.exists(INPUT_TASK_FILE): 
        print(f"❌ 找不到输入文件: {INPUT_TASK_FILE}，请先运行上一步的任务构建脚本！")
        return

    # 1. 断点续传逻辑 (核心！省钱必备)
    processed_ids = set()
    if os.path.exists(OUTPUT_RESULT_FILE):
        with open(OUTPUT_RESULT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    processed_ids.add(json.loads(line)['custom_id'])
                except: 
                    pass
    
    print(f"🔄 已完成: {len(processed_ids)} 条 | 准备继续执行...")

    with open(INPUT_TASK_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        
    tasks_to_run = [json.loads(line) for line in all_lines if json.loads(line)['custom_id'] not in processed_ids]

    if not tasks_to_run:
        print("🎉 所有电影的元数据恢复任务已全部完成！")
        return

    with open(OUTPUT_RESULT_FILE, 'a', encoding='utf-8') as f_out:
        for task in tqdm(tasks_to_run, desc="🚀 Extracting Titles"):
            cid = task['custom_id'] # 这里的 cid 就是 item_id
            prompt = task['prompt']
            
            response_text = call_gemini_for_item(prompt)
            
            if response_text:
                result_obj = {
                    "custom_id": cid, # 对应 item_id
                    "response_text": response_text
                }
                f_out.write(json.dumps(result_obj, ensure_ascii=False) + '\n')
                f_out.flush()
            
            # 动态休眠防封，0.2秒比较安全
            time.sleep(0.2) 

    print(f"✅ 完成！结果已保存至 {OUTPUT_RESULT_FILE}")

if __name__ == "__main__":
    run_item_metadata_extraction()