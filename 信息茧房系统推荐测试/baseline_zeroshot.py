import json
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_PATH = '/data/resource/models/Qwen/Qwen2.5-7B-Instruct'
TEST_FILE = 'lzc/topk_hard_final_test_set.jsonl'

def calculate_ndcg(ranked_list, k):
    """计算 NDCG@K"""
    dcg = 0.0
    idcg = 0.0
    total_positives = sum(1 for item in ranked_list if item['is_label'] == 1)
    
    if total_positives == 0:
        return 0.0
        
    for i in range(min(k, len(ranked_list))):
        if ranked_list[i]['is_label'] == 1:
            dcg += 1.0 / math.log2(i + 2)
            
    for i in range(min(k, total_positives)):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg

def calculate_mrr(ranked_list):
    """计算 MRR (Mean Reciprocal Rank)"""
    for i, item in enumerate(ranked_list):
        if item['is_label'] == 1:
            return 1.0 / (i + 1)
    return 0.0

def calculate_auc(ranked_list):
    """计算 AUC (Area Under Curve)"""
    pos_scores = [x['score'] for x in ranked_list if x['is_label'] == 1]
    neg_scores = [x['score'] for x in ranked_list if x['is_label'] == 0]
    
    if not pos_scores or not neg_scores:
        return None
        
    correct_pairs = 0.0
    total_pairs = len(pos_scores) * len(neg_scores)
    
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                correct_pairs += 1.0
            elif p == n:
                correct_pairs += 0.5
                
    return correct_pairs / total_pairs

def run_sereneva_full_eval():
    print("🚀 启动 SerenEva (HKBU 2025) LLM 直接惊喜度打分基线 (全量评估版)...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    # 使用 bfloat16 或 float16 加速推理
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # 初始化核心 4 指标
    hr10 = 0
    ndcg10 = 0.0
    mrr = 0.0
    auc_sum = 0.0
    
    total_users = 0
    valid_auc_users = 0
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"\n⏳ 共加载 {len(lines)} 个测试用户，开始漫长的打分过程 (预计需要一点时间)...")
    
    for idx, line in enumerate(lines, 1):
        if not line.strip(): continue
        user_data = json.loads(line.strip())
        
        history_str = "\n".join([f"- 《{item.get('title', 'Unknown')}》: {item.get('description', '')}" for item in user_data['user_history'][-5:]])
        
        candidates_ranked = []
        
        for cand in user_data['candidates']:
            title = cand['title']
            is_label = cand.get('is_label', 0)
            
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
                outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.1)
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 解析得分
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                scores = json.loads(response[start:end])
                s_score = float(scores.get('serendipity_score', 0.0))
            except:
                s_score = 0.0 
                
            candidates_ranked.append({"title": title, "score": s_score, "is_label": is_label})
            
        # 根据打分进行降序排列
        candidates_ranked.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算当前用户的 4 大核心指标
        if any(it['is_label'] == 1 for it in candidates_ranked[:10]): hr10 += 1
        ndcg10 += calculate_ndcg(candidates_ranked, 10)
        mrr += calculate_mrr(candidates_ranked)
        
        auc_val = calculate_auc(candidates_ranked)
        if auc_val is not None:
            auc_sum += auc_val
            valid_auc_users += 1
            
        total_users += 1
        
        # 实时打印进度
        current_hr10 = hr10 / total_users
        hit_rank = next((i + 1 for i, res in enumerate(candidates_ranked) if res['is_label'] == 1), -1)
        print(f"✅ [{idx}/{len(lines)}] 处理完毕 | 目标排第 {hit_rank} 名 | 当前 HR@10: {current_hr10*100:.2f}%")

    # ==========================================
    # 打印最终结果报告
    # ==========================================
    if total_users == 0:
        return
        
    final_hr10 = hr10 / total_users
    final_ndcg10 = ndcg10 / total_users
    final_mrr = mrr / total_users
    final_auc = auc_sum / valid_auc_users if valid_auc_users > 0 else 0.0

    print("\n" + "🏁 " + "="*50)
    print(" 🚀 SerenEva Baseline (Direct Serendipity) 最终评测结果 🚀")
    print("="*52)
    print(f" 📊 测试样本总数 (User Groups): {total_users}")
    print("-" * 52)
    print(f" | 指标名称   | 最终得分  | 指标含义简述")
    print("-" * 52)
    print(f" | HR@10      |  {final_hr10:.4f} | 召回宽容度 (是否进前10)")
    print(f" | NDCG@10    |  {final_ndcg10:.4f} | 列表排序质量 (越高越靠前)")
    print(f" | MRR        |  {final_mrr:.4f} | 首中倒数排名 (用户第一印象)")
    print(f" | AUC        |  {final_auc:.4f} | 全局排序能力 (正负样本区分度)")
    print("="*52)
    print("💡 完美！直接把这个表的数据填进你的论文里！")

if __name__ == "__main__":
    run_sereneva_full_eval()