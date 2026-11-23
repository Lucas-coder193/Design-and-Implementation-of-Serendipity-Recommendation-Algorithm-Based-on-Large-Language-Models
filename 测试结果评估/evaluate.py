import json
import os
import numpy as np
from scipy.stats import pearsonr

# ==========================================
# 1. 核心评估函数
# ==========================================
def evaluate_predictions(file_path):
    """
    读取并评估单个 jsonl 文件的预测结果
    指标包括：MAE, RMSE, Pearson Correlation, Prediction STD
    """
    # 存储数据：{ 维度名: {'pred': [], 'label': []} }
    metrics_data = {} 
    
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # 兼容性处理：防止模型输出带 Markdown 标签的字符串
                pred_content = data['predict'].strip()
                if pred_content.startswith("```json"):
                    pred_content = pred_content[7:-3].strip()
                
                predict_dict = json.loads(pred_content)
                label_dict = json.loads(data['label'])
                
                for key in label_dict:
                    if key not in metrics_data:
                        metrics_data[key] = {'pred': [], 'label': []}
                    
                    # 提取数值，若预测缺失则给默认值 3.0
                    p_val = float(predict_dict.get(key, 3.0))
                    l_val = float(label_dict[key])
                    
                    metrics_data[key]['pred'].append(p_val)
                    metrics_data[key]['label'].append(l_val)
                    
            except Exception as e:
                # 打印解析错误的行，方便排查格式问题
                # print(f"Skipping line {line_num+1} due to error: {e}")
                pass
                
    # 计算统计指标
    final_results = {}
    for key, values in metrics_data.items():
        preds = np.array(values['pred'])
        labels = np.array(values['label'])
        
        if len(preds) > 0:
            # A. 绝对误差 (原有)
            mae = np.mean(np.abs(preds - labels))
            rmse = np.sqrt(np.mean((preds - labels) ** 2))
            
            # B. 相对趋势 (新增：皮尔逊相关系数)
            # 处理标准差为 0 的情况（即模型只会打同一个分时无法计算相关性）
            if np.std(preds) == 0 or np.std(labels) == 0:
                corr = 0.0
            else:
                corr, _ = pearsonr(preds, labels)
            
            # C. 模型健康度 (新增：预测值的标准差)
            # 如果这个值太小（如 < 0.1），说明模型出现了“分数坍塌”
            pred_std = np.std(preds)
            
            final_results[key] = {
                'MAE': mae,
                'RMSE': rmse,
                'Pearson': corr,
                'Pred_STD': pred_std,
                'Count': len(preds)
            }
            
    return final_results

# ==========================================
# 2. 执行主程序
# ==========================================
def main():
    # 路径配置
    folder_path = './测试结果评估' 
    files_to_eval = [
        "books_analyst_generated_predictions.jsonl",
        "books_explorer_generated_predictions.jsonl",
        "movies_analyst_generated_predictions.jsonl",
        "movies_explorer_generated_predictions.jsonl"
    ]
    
    print("🚀 启动大模型组件打分能力评估（微调效果检测）...")
    print("-" * 60)

    for file_name in files_to_eval:
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.exists(file_path):
            print(f"📊 正在评估文件: {file_name}")
            results = evaluate_predictions(file_path)
            
            if not results:
                print("   ⚠️ 未能解析到有效数据。")
                continue

            for metric, scores in results.items():
                print(f"   🔹 维度: {metric.upper():<15}")
                print(f"      [准确度] MAE: {scores['MAE']:.4f} | RMSE: {scores['RMSE']:.4f}")
                print(f"      [相关性] Pearson: {scores['Pearson']:.4f} (越接近1越好)")
                print(f"      [健康度] 预测标准差: {scores['Pred_STD']:.4f}")
                print(f"      [样本数] Count: {scores['Count']}")
            print("-" * 60)
        else:
            print(f"❌ 找不到文件: {file_path}")

if __name__ == "__main__":
    main()