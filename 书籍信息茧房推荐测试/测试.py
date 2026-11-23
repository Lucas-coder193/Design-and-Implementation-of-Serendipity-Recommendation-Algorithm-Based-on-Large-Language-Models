import json
import re

# ==========================================
# 1. 路径与权重配置
# ==========================================
DATASET_FILE = r'书籍信息茧房推荐测试\topk_books_dataset.jsonl'
ANALYST_FILE = r'测试结果评估\books_analyst_generated_predictions.jsonl'
EXPLORER_FILE = r'测试结果评估\books_explorer_generated_predictions.jsonl'
BOOKS_METADATA = r'构建书籍测试集\book_metadata_api_results.jsonl'
RAW_TEST_SET = r'构建书籍测试集\books_test_set_1_19.json'

# 核心：五因子权重
WEIGHTS = {
    "relevance": 2.5,
    "utility": 1.0,
    "novelty": 1.5,
    "unexpectedness": 1.5,
    "diversity": 3.5
}
K = 10  # 评测 Top-10

def evaluate_cocoon_breaking():
    print("🚀 开始端到端评测：寻找乱序打分 -> 加权计算 -> 输出指标")
    
    # 1. 提取元数据书名
    metadata_titles = {}
    with open(BOOKS_METADATA, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            iid = data.get("custom_id")
            try:
                inner = json.loads(data.get("response_text", "{}"))
                if isinstance(inner, list): inner = inner[0]
                metadata_titles[iid] = inner.get("title", "").strip()
            except:
                pass

    # 2. 提取用户历史书名（辅助精准匹配）
    user_histories = {}
    with open(RAW_TEST_SET, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for item in raw_data:
            uid = item.get('user_id')
            user_histories[uid] = [metadata_titles.get(hid, "") for hid in item.get('history', []) if hid in metadata_titles]

    # 3. 加载预测文件
    with open(ANALYST_FILE, 'r', encoding='utf-8') as f:
        analyst_preds = [json.loads(line) for line in f if line.strip()]
    with open(EXPLORER_FILE, 'r', encoding='utf-8') as f:
        explorer_preds = [json.loads(line) for line in f if line.strip()]

    def get_scores(pred_obj, keys):
        if not pred_obj: return {k: 0.0 for k in keys}
        try:
            p_dict = json.loads(pred_obj.get("predict", "{}"))
            return {k: float(p_dict.get(k, 0.0)) for k in keys}
        except:
            return {k: 0.0 for k in keys}

    def find_match(pred_list, cand_title, user_h_titles):
        """乱序匹配核心逻辑"""
        if not cand_title: return None
        target_str = f"Target Item: 《{cand_title}》"
        
        # 第一轮：精准匹配
        for i, pred in enumerate(pred_list):
            prompt = pred.get("prompt", "")
            if target_str in prompt:
                if not user_h_titles: return pred_list.pop(i)
                for ht in user_h_titles:
                    if ht and f"《{ht}》" in prompt:
                        return pred_list.pop(i)
                        
        # 第二轮：兜底匹配
        for i, pred in enumerate(pred_list):
            prompt = pred.get("prompt", "")
            if target_str in prompt: return pred_list.pop(i)
        return None

    # 4. 开始遍历评估
    hits = 0
    rr_sum = 0.0
    total_users = 0
    missing_scores = 0

    print(f"⚖️ 当前权重配置: {WEIGHTS}")
    print("⚙️ 正在匹配分数并计算排名，请稍候...")

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            user_data = json.loads(line)
            uid = user_data["user_id"]
            h_titles = user_histories.get(uid, [])
            
            candidates_with_scores = []
            
            for cand in user_data["candidates"]:
                iid = cand["item_id"]
                is_label = cand["is_label"]
                cand_title = metadata_titles.get(iid, "")
                
                # 去乱序列表里捞分数
                a_match = find_match(analyst_preds, cand_title, h_titles)
                e_match = find_match(explorer_preds, cand_title, h_titles)
                
                if not a_match or not e_match:
                    missing_scores += 1
                
                a_scores = get_scores(a_match, ["relevance", "utility"])
                e_scores = get_scores(e_match, ["unexpectedness", "novelty", "diversity"])
                
                # === 核心加权计算 ===
                final_score = (
                    a_scores["relevance"] * WEIGHTS["relevance"] +
                    a_scores["utility"] * WEIGHTS["utility"] +
                    e_scores["novelty"] * WEIGHTS["novelty"] +
                    e_scores["unexpectedness"] * WEIGHTS["unexpectedness"] +
                    e_scores["diversity"] * WEIGHTS["diversity"]
                )
                
                candidates_with_scores.append({
                    "item_id": iid,
                    "is_label": is_label,
                    "final_score": final_score
                })
            
            # 5. 排序并计算指标
            # 按 final_score 降序排列
            candidates_with_scores.sort(key=lambda x: x["final_score"], reverse=True)
            
            # 取 Top-K 进行评判
            top_k_items = candidates_with_scores[:K]
            
            # 检查正样本是否在 Top-K 中
            hit = False
            rank = 0
            for idx, item in enumerate(top_k_items):
                if item["is_label"] == 1:
                    hit = True
                    rank = idx + 1
                    break
            
            if hit:
                hits += 1
                rr_sum += 1.0 / rank
                
            total_users += 1

    # ==========================================
    # 5. 打印最终结果
    # ==========================================
    hr_at_k = hits / total_users if total_users > 0 else 0
    mrr_at_k = rr_sum / total_users if total_users > 0 else 0

    print("\n================ 📊 评测结果 ================")
    print(f"总测试用户数 : {total_users}")
    print(f"命中目标数   : {hits}")
    if missing_scores > 0:
        print(f"⚠️ 警告: 有 {missing_scores} 个候选项未能找到匹配的打分，被记为0分。")
    print("---------------------------------------------")
    print(f"🏆 HR@{K}  (Hit Ratio)        : {hr_at_k:.4f}")
    print(f"🏅 MRR@{K} (Mean Reciprocal Rank) : {mrr_at_k:.4f}")
    print("=============================================")

if __name__ == "__main__":
    evaluate_cocoon_breaking()