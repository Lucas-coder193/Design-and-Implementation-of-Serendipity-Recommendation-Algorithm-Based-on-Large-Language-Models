import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 基础模型路径 (不挂载任何 LoRA Adapter)
BASE_MODEL_PATH = '/data/resource/models/Qwen/Qwen2.5-7B-Instruct'
TEST_FILE = 'lzc/topk_final_test_set.jsonl'

def run_sereneva_baseline():
    print("🚀 启动 SerenEva (HKBU 2025) LLM 直接惊喜度打分基线...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        # 这里仅读取第一行用于演示，实际运行需遍历所有测试数据
        user_data = json.loads(f.readline())
        
    history_str = "\n".join([f"- 《{item.get('title', 'Unknown')}》: {item.get('description', '')}" for item in user_data['user_history'][-5:]])
    
    candidates_ranked = []
    
    print("\n⏳ 正在让未经微调的基座模型进行 SerenEva 惊喜度 (Serendipity) 独立打分...")
    for cand in user_data['candidates']:
        title = cand['title']
        
        # 核心修改：遵循 SerenEva 思想，直接让大模型做综合的惊喜度评估
        prompt = f"""You are an evaluator for a serendipity-oriented recommender system. 
A serendipitous recommendation is defined as an item that is BOTH relevant to the user's historical preferences AND highly unexpected/surprising.

User History:
{history_str}

Target Item: 《{title}》 - {cand['description']}

Task: Evaluate whether the Target Item is serendipitous for the user.
Provide a "serendipity_score" strictly from 1.0 to 10.0, where 10.0 means highly serendipitous.

Output STRICTLY in JSON format: {{"serendipity_score": float}}"""

        messages = [
            {"role": "system", "content": "You are a helpful evaluator."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            # 输出极其简短，只需一个 JSON
            outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.1)
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 尝试解析单一分数输出
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            scores = json.loads(response[start:end])
            s_score = float(scores.get('serendipity_score', 0.0))
        except:
            s_score = 0.0 
            print(f"   ⚠️ 模型格式遵循失败，输出: {response.strip()}")
            
        candidates_ranked.append({"title": title, "score": s_score})
        print(f"   《{title}》 Serendipity 得分: {s_score}")

    # 根据大模型给出的惊喜度分数直接进行倒序排名
    candidates_ranked.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n🏆 SerenEva Baseline (Direct Serendipity) Top-3 推荐结果：")
    for i, res in enumerate(candidates_ranked[:3], 1):
        print(f"【No.{i}】 《{res['title']}》 | 惊喜度得分: {res['score']:.2f}")
        
    print("-" * 50)
    print("💡 实验对比意义：你可以用这个脚本跑出来的 HR@10 去对比你之前双智能体的 54.84%。")

if __name__ == "__main__":
    run_sereneva_baseline()