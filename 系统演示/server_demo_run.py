import json
import os
import torch
import gc
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. 路径与核心配置
# ==========================================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
ANALYST_ADAPTER = os.environ.get("ANALYST_ADAPTER", "analyst_v1")
EXPLORER_ADAPTER = os.environ.get("EXPLORER_ADAPTER", "explorer_v1")

# 输入与输出文件路径
RAW_DATA_FILE = 'lzc/topk_last_final_test_set.jsonl'  # 请确保相对路径正确
ANALYST_INPUT_JSON = 'test_predict_analyst.json'
EXPLORER_INPUT_JSON = 'test_predict_explorer.json'
ANALYST_OUTPUT_JSONL = 'test_analyst_generated_predictions.jsonl'
EXPLORER_OUTPUT_JSONL = 'test_explorer_generated_predictions.jsonl'

# 重排权重: Rel(2.5), Util(1.0), Nov(1.5), Unexp(1.5), Div(3.5)
WEIGHTS = np.array([2.5, 1.0, 1.5, 1.5, 3.5])

# System Prompt
SYS_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
ANALYST_INST = "You are an Analyst Agent in a serendipity recommender system. Your task is to perform deep semantic alignment and mine emotional resonance. Based on the user's history and the target item, predict the Relevance and Utility scores (1.0-5.0)."
EXPLORER_INST = "You are an Explorer Agent in a serendipity recommender system. Your task is to break user information boundaries. Based on the user's history and the target item, predict the Novelty, Unexpectedness, and Diversity scores (1.0-5.0)."

# ==========================================
# 工具函数
# ==========================================
def clear_memory():
    """强制清理显存，防止 OOM"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_base_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return model, tokenizer

def get_json_from_response(response_text):
    """从模型输出中安全提取 JSON 字符串"""
    try:
        start = response_text.rfind('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end != -1:
            return response_text[start:end]
    except:
        pass
    return "{}"

# ==========================================
# Step 1: 数据处理 (生成输入 JSON)
# ==========================================
def step1_prepare_data():
    print("\n" + "="*50)
    print("📦 [Step 1] 开始处理原始数据，生成模型输入 JSON...")
    
    with open(RAW_DATA_FILE, 'r', encoding='utf-8') as f:
        # 为了演示，我们只读取第一行（第一个用户）
        first_line = f.readline()
    
    user_data = json.loads(first_line)
    
    # 构造历史记录字符串
    history_str = "User History:\n"
    for item in user_data['user_history'][-5:]: # 取最近5个
        history_str += f"《{item.get('corrected_title', item.get('title', 'Unknown'))}》 - Description: {item.get('description', '')}\n"
    
    analyst_inputs = []
    explorer_inputs = []
    
    # 为该用户的每一个候选电影生成 prompt
    for cand in user_data['candidates']:
        target_str = f"\nTarget Item: 《{cand['title']}》\nTarget Description: {cand['description']}"
        input_text = history_str + target_str
        
        analyst_inputs.append({
            "instruction": ANALYST_INST,
            "input": input_text,
            "output": "",
            "title": cand['title'] # 附加字段，方便后续步骤读取
        })
        
        explorer_inputs.append({
            "instruction": EXPLORER_INST,
            "input": input_text,
            "output": "",
            "title": cand['title']
        })
        
    with open(ANALYST_INPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(analyst_inputs, f, ensure_ascii=False, indent=2)
    with open(EXPLORER_INPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(explorer_inputs, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 成功生成 {ANALYST_INPUT_JSON} 和 {EXPLORER_INPUT_JSON}，共 {len(analyst_inputs)} 条推理任务。")
    return analyst_inputs, explorer_inputs, history_str

# ==========================================
# Step 2: 模型推理 (生成打分 JSONL)
# ==========================================
def step2_run_inference(analyst_data, explorer_data):
    # ------------------------------------
    # A. 运行 Analyst 模型
    # ------------------------------------
    print("\n" + "="*50)
    print("🧠 [Step 2.1] 加载 Analyst 适配器，开始分析师打分...")
    model, tokenizer = load_base_model_and_tokenizer()
    model = PeftModel.from_pretrained(model, ANALYST_ADAPTER)
    
    analyst_results = []
    for data in analyst_data:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"{data['instruction']}\n{data['input']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
        
        # 截取新生成的部分
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predict_json_str = get_json_from_response(response)
        
        # 按照 LLaMA-Factory 的格式保存
        analyst_results.append({
            "prompt": prompt,
            "predict": predict_json_str,
            "label": "\n",
            "title": data["title"] # 辅助字段
        })
        print(f"   Analyst -> 《{data['title']}》: {predict_json_str}")
        
    with open(ANALYST_OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for res in analyst_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    # 彻底卸载 Analyst
    del model, tokenizer
    clear_memory()
    print("✅ Analyst 推理完成，显存已释放。")

    # ------------------------------------
    # B. 运行 Explorer 模型
    # ------------------------------------
    print("\n" + "="*50)
    print("🚀 [Step 2.2] 加载 Explorer 适配器，开始探索者打分...")
    model, tokenizer = load_base_model_and_tokenizer()
    model = PeftModel.from_pretrained(model, EXPLORER_ADAPTER)
    
    explorer_results = []
    for data in explorer_data:
        messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"{data['instruction']}\n{data['input']}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predict_json_str = get_json_from_response(response)
        
        explorer_results.append({
            "prompt": prompt,
            "predict": predict_json_str,
            "label": "\n",
            "title": data["title"]
        })
        print(f"   Explorer -> 《{data['title']}》: {predict_json_str}")
        
    with open(EXPLORER_OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for res in explorer_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # 彻底卸载 Explorer
    del model, tokenizer
    clear_memory()
    print("✅ Explorer 推理完成，显存已释放。")
    return analyst_results, explorer_results

# ==========================================
# Step 3 & 4: 重排与生成理由
# ==========================================
def step3_and_4_rerank_and_explain(analyst_results, explorer_results, history_str):
    print("\n" + "="*50)
    print("⚖️ [Step 3] 提取五因子得分，使用公式进行重排...")
    
    candidates_ranked = []
    
    for a_res, e_res in zip(analyst_results, explorer_results):
        title = a_res["title"]
        try:
            a_scores = json.loads(a_res["predict"])
            e_scores = json.loads(e_res["predict"])
        except:
            a_scores, e_scores = {}, {}
            
        vec = np.array([
            float(a_scores.get('relevance', 3.0)),
            float(a_scores.get('utility', 3.0)),
            float(e_scores.get('novelty', 3.0)),
            float(e_scores.get('unexpectedness', 3.0)),
            float(e_scores.get('diversity', 3.0))
        ])
        
        final_score = np.dot(vec, WEIGHTS)
        candidates_ranked.append({
            "title": title,
            "score": final_score,
            "vec": vec
        })
        
    # 降序排序，取 Top-3
    candidates_ranked.sort(key=lambda x: x['score'], reverse=True)
    top_3 = candidates_ranked[:3]
    
    print("\n" + "="*50)
    print("💬 [Step 4] 加载 Qwen 基座模型，为 Top-3 生成推荐理由...")
    model, tokenizer = load_base_model_and_tokenizer()
    
    for item in top_3:
        prompt = f"""你是一个电影推荐专家。
                    用户的观影历史如下：
                    {history_str}

                    系统通过多因子计算，决定为用户推荐电影：《{item['title']}》。
                    请结合用户的观影历史，给出一句简短（不超过50字）且具有吸引力的推荐理由："""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        
        reason = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        item["reason"] = reason

    # 最终结果展示
    print("\n" + "█"*60)
    print("🏆 毕设最终系统展示 - Top-3 推荐结果 🏆")
    print("█"*60)
    for i, res in enumerate(top_3, 1):
        print(f"【No.{i}】 《{res['title']}》")
        print(f"   📊 综合加权得分: {res['score']:.2f}")
        print(f"   📐 五因子雷达 [Rel, Util, Nov, Unexp, Div]: {list(res['vec'])}")
        print(f"   💡 AI 推荐理由: {res['reason']}")
        print("-" * 60)

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 1. 准备数据
    analyst_inputs, explorer_inputs, history_str = step1_prepare_data()
    
    # 2. 推理打分
    analyst_res, explorer_res = step2_run_inference(analyst_inputs, explorer_inputs)
    
    # 3 & 4. 重排并生成理由
    step3_and_4_rerank_and_explain(analyst_res, explorer_res, history_str)