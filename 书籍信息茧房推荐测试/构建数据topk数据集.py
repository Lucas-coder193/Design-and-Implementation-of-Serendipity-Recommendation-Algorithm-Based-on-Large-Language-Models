import json

# ==========================================
# 1. 路径配置 (请确认路径完全准确)
# ==========================================
DATASET_FILE = r'书籍信息茧房推荐测试\topk_books_dataset.jsonl'
ANALYST_FILE = r'测试结果评估\books_analyst_generated_predictions.jsonl'
EXPLORER_FILE = r'测试结果评估\books_explorer_generated_predictions.jsonl'
BOOKS_METADATA = r'构建书籍测试集\book_metadata_api_results.jsonl'
RAW_TEST_SET = r'构建书籍测试集\books_test_set_1_19.json'

OUTPUT_FILE = r'书籍信息茧房推荐测试\topk_books_dataset_with_scores.jsonl'

def run_score_injector():
    print("🚀 开始执行乱序分数的精准匹配与注入...")

    # 1. 加载 Metadata，建立 item_id -> 书名的映射
    print("📚 正在加载元数据提取书名...")
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

    # 2. 加载用户的历史记录，建立 user_id -> 历史书名列表的映射（用于防止重名书匹配错人）
    print("👤 正在加载用户历史画像...")
    user_histories = {}
    with open(RAW_TEST_SET, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for item in raw_data:
            uid = item.get('user_id')
            h_ids = item.get('history', [])
            user_histories[uid] = [metadata_titles.get(hid, "") for hid in h_ids if hid in metadata_titles]

    # 3. 加载乱序的预测数据
    print("🌀 正在加载乱序的预测打分文件...")
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
        """核心乱序匹配逻辑：通过 prompt 里的 Target Item 书名找回对应的打分"""
        if not cand_title: return None
        target_str = f"Target Item: 《{cand_title}》"
        
        # 第一轮：精准匹配 (目标书名一致，且用户的历史书名也出现在 prompt 里)
        for i, pred in enumerate(pred_list):
            prompt = pred.get("prompt", "")
            if target_str in prompt:
                if not user_h_titles:
                    return pred_list.pop(i)
                for ht in user_h_titles:
                    if ht and f"《{ht}》" in prompt:
                        return pred_list.pop(i) # 找到了就弹出，加快后续查找速度
                        
        # 第二轮：兜底匹配 (如果历史书名没对上，只要目标书名对了就认)
        for i, pred in enumerate(pred_list):
            prompt = pred.get("prompt", "")
            if target_str in prompt:
                return pred_list.pop(i)
                
        return None

    # 4. 遍历数据集，拼装分数
    print("🧩 正在执行跨文件匹配拼装，这可能需要几秒钟...")
    new_dataset = []
    missing_count = 0
    total_count = 0

    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            user_data = json.loads(line)
            uid = user_data["user_id"]
            h_titles = user_histories.get(uid, [])
            
            for cand in user_data["candidates"]:
                total_count += 1
                iid = cand["item_id"]
                cand_title = metadata_titles.get(iid, "")
                
                # 寻找对应的预测数据
                a_match = find_match(analyst_preds, cand_title, h_titles)
                e_match = find_match(explorer_preds, cand_title, h_titles)
                
                if not a_match or not e_match:
                    missing_count += 1
                
                # 提取分数
                a_scores = get_scores(a_match, ["relevance", "utility"])
                e_scores = get_scores(e_match, ["unexpectedness", "novelty", "diversity"])
                
                cand["scores"] = {**a_scores, **e_scores}
                
            new_dataset.append(user_data)

    # 5. 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for data in new_dataset:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("========================================")
    print(f"🎉 拼装彻底完成！总共处理了 {total_count} 个候选项。")
    if missing_count > 0:
        print(f"⚠️ 有 {missing_count} 个候选项没能在预测文件里找到匹配的书名。")
    else:
        print("✅ 完美匹配！没有遗漏任何一个分数！")
    print(f"📁 最终带分数的测试集已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_score_injector()