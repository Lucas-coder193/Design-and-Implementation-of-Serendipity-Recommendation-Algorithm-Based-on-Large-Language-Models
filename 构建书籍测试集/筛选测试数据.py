import pandas as pd
import random
import json
from tqdm import tqdm

# 1. 读取数据 (替换为你的实际路径)
csv_path = r"SerenLens\SerenLens\Dataset\SerenLens_Books.csv"
df = pd.read_csv(csv_path)

# 确保时间戳是数值型以便排序
df['timestamp'] = pd.to_numeric(df['timestamp'])

# 2. 核心参数
WINDOW_SIZE = 5      # 历史窗口大小
TARGET_POS_COUNT = 50 # 设定你想提取的“正样本”数量（控制测试规模，50个正样本对应950个负样本）
RATIO = 19           # 负样本比例

pos_windows = []
neg_windows = []

# 3. 按用户分组并按时间排序，执行真实的滑动窗口
grouped = df.groupby('user_id')

for user_id, group in tqdm(grouped, desc="Sliding Windows"):
    group = group.sort_values(by='timestamp').reset_index(drop=True)
    
    # 如果用户总交互记录不够一个窗口+1个目标，跳过
    if len(group) <= WINDOW_SIZE:
        continue
        
    # 真实的滑动窗口逻辑
    for i in range(WINDOW_SIZE, len(group)):
        # 截取 i 前面的 WINDOW_SIZE 个商品作为 history
        history_items = group.iloc[i - WINDOW_SIZE : i]['item_id'].tolist()
        target_item = group.iloc[i]['item_id']
        label = group.iloc[i]['label']
        
        record = {
            "user_id": str(user_id),
            "history": history_items,
            "target": str(target_item),
            "expected_label": float(label)
        }
        
        # 根据真实的 label 分发到不同的候选池
        if label == 1.0:
            pos_windows.append(record)
        elif label == 0.0:
            neg_windows.append(record)

# 4. 严格按照 1:19 比例抽样
# 为了控制 API 调用成本，我们限制一下正样本数量 (比如 50 个)
if len(pos_windows) > TARGET_POS_COUNT:
    pos_windows = random.sample(pos_windows, TARGET_POS_COUNT)

required_neg_count = len(pos_windows) * RATIO

if len(neg_windows) < required_neg_count:
    print(f"警告: 真实的负样本窗口数量不足！仅有 {len(neg_windows)} 个。")
    # 如果不够，就把所有的负样本都拿进来（但在你这个数据集里，0.0 的数据通常海量，不会发生）
    sampled_neg_windows = neg_windows
else:
    # 从海量的真实负样本窗口中随机抽取 19 倍的数量
    sampled_neg_windows = random.sample(neg_windows, required_neg_count)

# 合并并打乱顺序
final_test_dataset = pos_windows + sampled_neg_windows
random.shuffle(final_test_dataset)

# 5. 保存为 JSON
output_path = r"构建书籍测试集\books_test_set_1_19.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_test_dataset, f, ensure_ascii=False, indent=4)

print(f"\n成功提取！共生成了 {len(final_test_dataset)} 条真实滑动窗口数据。")
print(f"其中正样本(惊喜): {len(pos_windows)} 条，负样本(不惊喜): {len(sampled_neg_windows)} 条。")
print(f"文件已保存至: {output_path}")