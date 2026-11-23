import json
import random
import os

def build_train_test_and_api_tasks():
    print("=== 🚀 滑动窗口切分 & API 任务去重启动 (保留用户版) ===")
    
    # 固定随机种子，保证每次划分的 Train/Test 是一致的
    random.seed(42) 

    input_path = '构建电影数据集/user_sequences_enriched.json'
    if not os.path.exists(input_path):
        print(f"❌ 找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        user_seqs = json.load(f)

    window_size = 5
    user_windows = {}

    print("🔄 [1/4] 正在生成滑动窗口 (保留所有合规用户)...")
    for uid, seq in user_seqs.items():
        # 只要用户的交互记录大于等于 6（5个历史+1个目标），就保留这个用户！
        if len(seq) < window_size + 1:
            continue
            
        windows = []
        # 生成该用户所有的滑动窗口
        for i in range(len(seq) - window_size):
            history = seq[i : i+window_size]
            target = seq[i+window_size]
            
            sample = {'history': history, 'target': target}
            windows.append(sample)
            
        user_windows[uid] = windows

    # 规则 2：用户隔离 (Seen vs Unseen) - 80% 训练用户, 20% 测试用户
    uids = list(user_windows.keys())
    random.shuffle(uids)
    split_idx = int(len(uids) * 0.8)
    train_uids = uids[:split_idx]
    test_uids = uids[split_idx:]
    
    print(f"👥 [2/4] 用户隔离完成: 训练集包含 {len(train_uids)} 名用户，测试集包含 {len(test_uids)} 名用户。")

    # 规则 1 & 3：全局 1:19 抽样函数
    def sample_1_to_19(uids_list, dataset_name):
        pos_samples = []
        neg_samples = []
        
        # 将该集合内所有用户的窗口放入全局池子
        for uid in uids_list:
            for w in user_windows[uid]:
                if w['target']['label'] == 1:
                    pos_samples.append((uid, w))
                else:
                    neg_samples.append((uid, w))
                    
        print(f"   🔎 {dataset_name} 初始池子: 惊喜样本={len(pos_samples)} 个，不惊喜样本={len(neg_samples)} 个")
        
        # 严格计算 1:19 比例
        target_neg_count = len(pos_samples) * 19
        
        if len(neg_samples) >= target_neg_count:
            # 不惊喜样本充足，直接随机抽取 19 倍
            sampled_negs = random.sample(neg_samples, target_neg_count)
            final_pos = pos_samples
        else:
            # 万一不惊喜样本不够 19 倍（几乎不可能，但防患于未然），反向裁剪正样本
            print(f"⚠️ {dataset_name} 不惊喜样本不足 19 倍，正在微调数量...")
            max_pos_allowed = len(neg_samples) // 19
            final_pos = random.sample(pos_samples, max_pos_allowed)
            sampled_negs = random.sample(neg_samples, max_pos_allowed * 19)
            
        final_dataset = final_pos + sampled_negs
        random.shuffle(final_dataset) # 打乱顺序，防止微调时模型产生偏见
        
        print(f"📊 {dataset_name} 最终抽样 -> 惊喜(1): {len(final_pos)}, 不惊喜(19): {len(sampled_negs)} -> 总计: {len(final_dataset)}")
        return final_dataset

    print("⚖️ [3/4] 正在进行全局 1:19 严格抽样...")
    train_data = sample_1_to_19(train_uids, "训练集 (Train)")
    test_data = sample_1_to_19(test_uids, "测试集 (Test)")

    # 规则 4 & 5：提取不重复的数据，准备去调 API
    print("🔍 [4/4] 正在提取去重后的 API 打分任务...")
    api_tasks = {}
    
    # 扫描刚刚抽样出来的最终数据集
    for uid, w in train_data + test_data:
        target_item = w['target']
        # 唯一键：用户ID + 目标电影ID
        unique_key = f"{uid}_{target_item['item_id']}"
        
        if unique_key not in api_tasks:
            # 提取 API 需要的“因”与“果”
            # 把前 5 部历史电影名拼成一句话给大模型看
            history_summary = " | ".join([f"《{h['corrected_title']}》" for h in w['history']])
            api_tasks[unique_key] = {
                "custom_id": unique_key,
                "user_id": uid,
                "history_watched": history_summary,
                "target_title": target_item['corrected_title'],
                "target_description": target_item['description'],
                "target_review": target_item['review'], # 🌟 果：用户的真实长评
                "label": target_item['label']
            }

    print(f"🎯 API 任务去重完毕！从总计 {len(train_data) + len(test_data)} 个样本中，提炼出 {len(api_tasks)} 个独立的打分请求！")

    # 💾 保存全部文件
    with open('构建电影数据集/train_movies_data.json', 'w', encoding='utf-8') as f:
        json.dump([{"user_id": u, "window": w} for u, w in train_data], f, ensure_ascii=False, indent=2)
        
    with open('构建电影数据集/test_movies_data.json', 'w', encoding='utf-8') as f:
        json.dump([{"user_id": u, "window": w} for u, w in test_data], f, ensure_ascii=False, indent=2)
        
    with open('构建电影数据集/api_tasks_movies.json', 'w', encoding='utf-8') as f:
        json.dump(list(api_tasks.values()), f, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print("🎉 完美！数据集划分与任务剥离完成！")
    print("="*40)

if __name__ == "__main__":
    build_train_test_and_api_tasks()