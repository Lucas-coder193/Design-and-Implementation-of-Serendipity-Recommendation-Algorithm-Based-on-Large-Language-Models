import json
import os

# === ⚙️ 配置文件 ===
INPUT_TRAIN = "构建电影数据集/train_movies_data.json"  
INPUT_TEST = "构建电影数据集/test_movies_data.json"    
INPUT_FACTORS = "构建电影数据集/five_factors_results.jsonl" 

# 输出的 4 个 LLaMA-Factory 数据集 (改名为 explorer 和 analyst)
OUT_EXPLORER_TRAIN = "构建电影数据集/llama_explorer_train.json"
OUT_EXPLORER_TEST  = "构建电影数据集/llama_explorer_test.json"
OUT_ANALYST_TRAIN  = "构建电影数据集/llama_analyst_train.json"
OUT_ANALYST_TEST   = "构建电影数据集/llama_analyst_test.json"

def load_five_factors():
    factor_dict = {}
    with open(INPUT_FACTORS, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            factors = data['five_factors']
            factor_dict[data['custom_id']] = factors
    return factor_dict

def build_agent_datasets(input_file, factor_dict, out_explorer, out_analyst, dataset_type="Train"):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    explorer_dataset, analyst_dataset = [], []
    missing_count = 0

    for row in data:
        user_id = row['user_id']
        target = row['window']['target']
        target_id = target['item_id']
        custom_id = f"{user_id}_{target_id}"

        if custom_id not in factor_dict:
            missing_count += 1
            continue

        factors = factor_dict[custom_id]
        
        # ✂️ 拆分因子给不同的智能体
        explorer_factors = {
            "novelty": factors.get("novelty", 0.0),
            "unexpectedness": factors.get("unexpectedness", 0.0),
            "diversity": factors.get("diversity", 0.0)
        }
        
        analyst_factors = {
            "relevance": factors.get("relevance", 0.0),
            "utility": factors.get("usefulness", 0.0) # 注意：这里把 usefulness 映射为你的 utility
        }

# 🌟 满血版：保留完整的历史电影标题和简介，用分隔符隔开
        history_items = []
        for h in row['window']['history']:
            title = h['corrected_title']
            desc = h.get('description', '').replace('\n', ' ').strip()
            history_items.append(f"《{title}》 - Description: {desc}")
            
        # 用换行符把历史记录拼起来，让大模型看着更清晰
        history_context = "\n".join(history_items)
        
        # 统一的 Input：极其丰满的上下文
        agent_input = f"User History:\n{history_context}\n\nTarget Item: 《{target['corrected_title']}》\nTarget Description: {target['description']}"
        
        # ---------------------------------------------------
        # 🕵️‍♂️ 构建 Explorer 智能体样本 (只预测: 新颖、意外、多样)
        # ---------------------------------------------------
        explorer_sample = {
            "instruction": "You are an Explorer Agent in a serendipity recommender system. Your task is to break user information boundaries. Based on the user's history and the target item, predict the Novelty, Unexpectedness, and Diversity scores (1.0-5.0).",
            "input": agent_input,
            "output": json.dumps(explorer_factors, ensure_ascii=False)
        }
        
        # ---------------------------------------------------
        # 👩‍🔬 构建 Analyst 智能体样本 (只预测: 相关、实用)
        # ---------------------------------------------------
        analyst_sample = {
            "instruction": "You are an Analyst Agent in a serendipity recommender system. Your task is to perform deep semantic alignment and mine emotional resonance. Based on the user's history and the target item, predict the Relevance and Utility scores (1.0-5.0).",
            "input": agent_input,
            "output": json.dumps(analyst_factors, ensure_ascii=False)
        }

        explorer_dataset.append(explorer_sample)
        analyst_dataset.append(analyst_sample)

    # 💾 保存文件
    with open(out_explorer, 'w', encoding='utf-8') as f:
        json.dump(explorer_dataset, f, ensure_ascii=False, indent=2)
    with open(out_analyst, 'w', encoding='utf-8') as f:
        json.dump(analyst_dataset, f, ensure_ascii=False, indent=2)

    print(f"📊 {dataset_type} 集处理完毕:")
    print(f"   - 成功生成 {len(explorer_dataset)} 条数据 (缺失: {missing_count} 条)")

def run_pipeline():
    print("=== 🚀 开始生成 多智能体 (Multi-Agent) 微调数据集 ===")
    factor_dict = load_five_factors()
    
    print("\n📦 正在处理训练集...")
    build_agent_datasets(INPUT_TRAIN, factor_dict, OUT_EXPLORER_TRAIN, OUT_ANALYST_TRAIN, "Train")
    
    print("\n📦 正在处理测试集...")
    build_agent_datasets(INPUT_TEST, factor_dict, OUT_EXPLORER_TEST, OUT_ANALYST_TEST, "Test")
    
    print("\n🎉 全部大功告成！四个 Multi-Agent 数据集已准备就绪！")

if __name__ == "__main__":
    run_pipeline()