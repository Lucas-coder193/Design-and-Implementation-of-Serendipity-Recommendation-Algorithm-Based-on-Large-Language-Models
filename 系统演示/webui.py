import streamlit as st
import json
import numpy as np
import os
from pathlib import Path
from openai import OpenAI

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="SerenAgent 推荐系统", page_icon="🍿", layout="wide")
st.title("🍿 SerenAgent: 多智能体惊喜度推荐系统")
st.markdown("架构：离线专家双盲打分 (Local) + 在线推荐理由生成 (OpenRouter API)")
st.markdown("---")

# ==========================================
# 侧边栏配置参数
# ==========================================
st.sidebar.header("⚙️ 系统配置")
API_KEY = st.sidebar.text_input("OpenRouter API Key", type="password", value=os.environ.get("OPENROUTER_API_KEY", ""), help="可在此输入密钥，或设置环境变量 OPENROUTER_API_KEY。")
# OpenRouter 支持的模型标识符，这里使用 Qwen 2.5 7B
OPENROUTER_MODEL = st.sidebar.text_input("大模型端点", value="qwen/qwen-2.5-7b-instruct")

# 五因子权重: Rel, Util, Nov, Unexp, Div
WEIGHTS = np.array([2.5, 1.0, 1.5, 1.5, 3.5])

# 文件路径配置 (自动处理操作系统的斜杠问题)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "信息茧房系统推荐测试"
ANALYST_FILE = os.path.join(BASE_DIR, "test_last_analyst_generated_predictions.jsonl")
EXPLORER_FILE = os.path.join(BASE_DIR, "test_last_explorer_generated_predictions.jsonl")

if not (os.path.isfile(ANALYST_FILE) and os.path.isfile(EXPLORER_FILE)):
    ANALYST_FILE = str(PROJECT_ROOT / "examples" / "analyst_predictions.jsonl")
    EXPLORER_FILE = str(PROJECT_ROOT / "examples" / "explorer_predictions.jsonl")
    st.info("当前使用虚构示例评分，仅用于演示排序流程，不代表模型实验结果。")

# ==========================================
# 核心逻辑函数
# ==========================================
@st.cache_data
def load_and_merge_predictions(analyst_path, explorer_path):
    """读取本地两个智能体的 JSONL 打分文件并合并"""
    candidates = []
    history_text = ""
    CANDIDATES_PER_USER = 20  # 每个用户有20个候选电影
    TARGET_USER_INDEX = 0 if Path(analyst_path).parent.name == "examples" else 1
    
    if not os.path.exists(analyst_path) or not os.path.exists(explorer_path):
        return None, "找不到指定的打分文件，请检查路径是否正确。"
        
    try:
        with open(analyst_path, 'r', encoding='utf-8') as fa, \
             open(explorer_path, 'r', encoding='utf-8') as fe:
            
            for i, (line_a, line_e) in enumerate(zip(fa, fe)):
                # 跳过前面用户的数据
                if i < TARGET_USER_INDEX * CANDIDATES_PER_USER:
                    continue
                # 只取目标用户的20个候选电影
                if i >= (TARGET_USER_INDEX + 1) * CANDIDATES_PER_USER:
                    break
                    
                data_a = json.loads(line_a)
                data_e = json.loads(line_e)
                
                # 从 prompt 中提取目标电影名称，格式是 "Target Item: 《电影名称》"
                prompt_text = data_a.get('prompt', '')
                title = 'Unknown'
                if "Target Item: 《" in prompt_text:
                    start = prompt_text.find("Target Item: 《") + len("Target Item: 《")
                    end = prompt_text.find("》", start)
                    if end > start:
                        title = prompt_text[start:end]
                
                # 尝试从 Analyst 的 prompt 中提取用户的观影历史（只需要提取一次即可）
                if not history_text:
                    if "User History:\n" in prompt_text and "\nTarget Item:" in prompt_text:
                        # 切片截取历史记录部分
                        history_text = prompt_text.split("User History:\n")[1].split("\nTarget Item:")[0].strip()

                # 解析打分 JSON 字符串
                try:
                    score_a = json.loads(data_a.get('predict', '{}'))
                except: score_a = {}
                
                try:
                    score_e = json.loads(data_e.get('predict', '{}'))
                except: score_e = {}

                candidates.append({
                    'title': title,
                    'scores': {
                        'rel': float(score_a.get('relevance', 3.0)),
                        'util': float(score_a.get('utility', 3.0)),
                        'nov': float(score_e.get('novelty', 3.0)),
                        'unexp': float(score_e.get('unexpectedness', 3.0)),
                        'div': float(score_e.get('diversity', 3.0))
                    }
                })
        return candidates, history_text
    except Exception as e:
        return None, f"数据解析失败: {str(e)}"

def generate_reason(history, target_title, api_key, model_name):
    """调用 OpenRouter API 生成推荐理由"""
    if not api_key:
        return "⚠️ 请在左侧侧边栏配置 OpenRouter API Key。"
    
    # OpenRouter 的标准初始化方式
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://localhost:8501", # OpenRouter 建议的 header
            "X-Title": "SerenLens-Graduation-Project",
        }
    )
    
    prompt = f"""你是一个电影推荐专家。
用户最近观看的5部电影：
{history}

现在系统为用户推荐：《{target_title}》。

请综合考虑用户的全部观影历史，简要说明为什么这部电影能给用户带来惊喜和新颖感。
要求：50字以内，客观吸引人，不要过度解读历史电影。"""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful recommender assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成失败: {str(e)}"

# ==========================================
# 前端交互 UI
# ==========================================
candidates_data, history_info = load_and_merge_predictions(ANALYST_FILE, EXPLORER_FILE)

if candidates_data is None:
    st.error(history_info)
    st.stop()

st.subheader("👤 解析到的用户背景历史")
if history_info:
    st.info(history_info)
else:
    st.warning("未能从数据中提取到用户历史，将采用默认推荐策略。")

st.markdown(f"**数据状态:** 成功加载 {len(candidates_data)} 个候选物品的五因子打分记录。")

if st.button("🚀 根据权重重排并生成 Top-3", use_container_width=True, type="primary"):
    
    progress_text = "正在执行 numpy 加权矩阵计算..."
    my_bar = st.progress(0, text=progress_text)
    
    # 1. 计算总分
    ranked_candidates = []
    for cand in candidates_data:
        s = cand['scores']
        vec = np.array([s['rel'], s['util'], s['nov'], s['unexp'], s['div']])
        final_score = np.dot(vec, WEIGHTS)
        ranked_candidates.append({
            "title": cand['title'],
            "vec": vec,
            "final_score": final_score
        })
    
    # 2. 按分数排序
    ranked_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    # 3. 去重：对于重复的电影名称，只保留评分最高的那个
    seen_titles = set()
    unique_candidates = []
    for item in ranked_candidates:
        if item['title'] not in seen_titles:
            seen_titles.add(item['title'])
            unique_candidates.append(item)
    
    # 4. 取去重后的 Top 3
    top_3 = unique_candidates[:3]
    my_bar.progress(50, text=f"计算完成！正在调用 OpenRouter ({OPENROUTER_MODEL}) API 生成推荐理由...")
    
    # 3. 渲染结果
    st.markdown("### 🏆 Top-3 惊喜度推荐榜单")
    cols = st.columns(3)
    
    for i, (col, item) in enumerate(zip(cols, top_3)):
        with col:
            st.success(f"**Top {i+1}: 《{item['title']}》**")
            st.metric("综合推荐分", f"{item['final_score']:.2f}")
            
            # 显示该电影的五因子得分
            vec = item['vec']
            st.caption(f"🎯 五因子得分: Rel={vec[0]} | Util={vec[1]} | Nov={vec[2]} | Unexp={vec[3]} | Div={vec[4]}")
            
            # 动态生成理由
            with st.spinner("AI 思考中..."):
                reason = generate_reason(history_info, item['title'], API_KEY, OPENROUTER_MODEL)
            
            st.info(f"💡 **AI 推荐语:**\n\n{reason}")
            
    my_bar.progress(100, text="推荐流程执行完毕！")