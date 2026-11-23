import json
import time
import os
import requests
import urllib3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 禁用安全警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === ⚙️ 核心配置 ===
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise RuntimeError("请先设置环境变量 OPENROUTER_API_KEY。")
MODEL_NAME = "google/gemini-2.0-flash-001"
INPUT_TASK_FILE = "构建电影数据集/api_tasks_movies.json" 
OUTPUT_RESULT_FILE = "构建电影数据集/five_factors_results.jsonl" 

# === 🌐 代理配置 ===
PROXY_URL = os.environ.get("OPENROUTER_PROXY") or None
# PROXY_URL = None 

# === 🚀 初始化 Session ===
def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    if PROXY_URL:
        session.proxies = {"http": PROXY_URL, "https": PROXY_URL}
    return session

# === 🧠 终极版 System Prompt：专注五因子，强制参考 Label ===
SYSTEM_PROMPT = """
You are an expert evaluator in Serendipity Recommendation systems. Your task is to extract the scores for 5 core factors based on a User-Item interaction.

[Task]
Based on the User's History, the Target Item's info, and strictly validated by the User's Actual Review and the Binary Surprise Label (the Ground Truth), score the following 5 factors from 1.0 to 5.0 (decimals allowed, e.g., 3.5). 

CRITICAL RULE: A positive review DOES NOT automatically mean high serendipity. Pay strict attention to the "Surprise Label" (1=Surprised, 0=Not Surprised). If the label is 0, the item might be highly satisfying but expected, so factors related to unexpectedness or novelty should be scored lower.

The 5 Factors are:
1. Novelty (新颖性): How new, unfamiliar, or conceptually fresh the item is to the user.
2. Relevance (相关性): How well the item aligns with the user's latent needs and historical preferences.
3. Unexpectedness (意外性): How much the item deviates from the user's predictable or recent historical pattern.
4. Usefulness (有用性): How much practical value, deep satisfaction, or emotional resonance the user gained, as expressed in their review.
5. Diversity (多样性): How much this specific item diversifies the user's overall consumption profile.

[Output Format]
Output ONLY a valid JSON object. Do not include markdown formatting like ```json.
{
  "novelty": float,
  "relevance": float,
  "unexpectedness": float,
  "usefulness": float,
  "diversity": float,
  "rationale": "One short sentence explaining the scores."
}
"""

def build_user_prompt(task):
    return f"""
[Input Data]
- User's History (Recently interacted items): {task['history_watched']}
- Target Item Title: {task['target_title']}
- Target Item Description: {task['target_description']}
- User's Actual Review: {task['target_review']}
- User's Binary Surprise Label: {task['label']} (1=Surprised, 0=Not Surprised)

Analyze the above and output the JSON.
"""

def call_gemini_api(task, session):
    # 🌟 修复 1：纯净的 URL 字符串
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    user_prompt = build_user_prompt(task)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1, 
        "response_format": {"type": "json_object"} 
    }
    
    try:
        response = session.post(url, headers=headers, json=payload, timeout=25, verify=False)
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            # 清洗 markdown 符号，防止 JSON 解析报错
            content = content.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(content) 
            return task['custom_id'], parsed_json
        else:
            # 🌟 修复 2：如果状态码不是 200，把错误信息打印出来，方便排查
            print(f"❌ API 报错，状态码: {response.status_code}, 详情: {response.text}")
            return task['custom_id'], None
    except Exception as e:
        # 🌟 修复 3：如果网络直接断了，把异常打印出来
        print(f"⚠️ 网络或代码异常: {e}")
        return task['custom_id'], None

def run_distillation():
    print(f"=== 🚀 知识蒸馏启动：五因子 API 打分 (多线程纯净版) ===")
    
    if not os.path.exists(INPUT_TASK_FILE):
        print(f"❌ 找不到输入文件 {INPUT_TASK_FILE}，请检查路径。")
        return

    with open(INPUT_TASK_FILE, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)

    # 📌 断点续传逻辑
    processed_ids = set()
    if os.path.exists(OUTPUT_RESULT_FILE):
        with open(OUTPUT_RESULT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['custom_id'])
                except:
                    pass
                    
    tasks_to_run = [t for t in all_tasks if t['custom_id'] not in processed_ids]
    
    print(f"📊 任务总数: {len(all_tasks)} | 已完成: {len(processed_ids)} | 本次待跑: {len(tasks_to_run)}")
    if not tasks_to_run:
        print("🎉 所有打分任务已全部完成！")
        return

    # ⚡ 开启 20 线程并发
    MAX_WORKERS = 20 
    session = get_session()
    
    print(f"⚡ 开启 {MAX_WORKERS} 线程狂飙模式，正在猛烈请求 API...")
    
    with open(OUTPUT_RESULT_FILE, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(call_gemini_api, task, session): task for task in tasks_to_run}
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc="🤖 五因子提炼中"):
                cid, result_json = future.result()
                
                if result_json:
                    save_obj = {
                        "custom_id": cid,
                        "five_factors": result_json
                    }
                    f_out.write(json.dumps(save_obj, ensure_ascii=False) + '\n')
                    f_out.flush() 

    print("\n" + "="*40)
    print("🎉 五因子蒸馏完毕！结果已安全存入: ", OUTPUT_RESULT_FILE)
    print("="*40)

if __name__ == "__main__":
    run_distillation()