import json
import random

INPUT_FILE = '信息茧房合适参数/topk_last_data_set.jsonl'
VAL_FILE = '信息茧房合适参数/topk_last_val_set.jsonl'
TEST_FILE = '信息茧房合适参数/topk_last_final_test_set.jsonl'

# 1. 读取所有数据
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    all_data = [json.loads(line) for line in f]

# 2. 随机打乱用户顺序
random.seed(42) # 固定随机种子，保证每次切分结果一致
random.shuffle(all_data)

# 3. 按照 4:6 或 5:5 比例切分 (这里假设用 40% 做验证，60% 做最终测试)
split_idx = int(len(all_data) * 0.4)
val_data = all_data[:split_idx]
test_data = all_data[split_idx:]

# 4. 保存文件
with open(VAL_FILE, 'w', encoding='utf-8') as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open(TEST_FILE, 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ 切分完成！验证集(寻优用): {len(val_data)} 条，最终测试集(论文汇报用): {len(test_data)} 条")