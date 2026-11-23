file_path = r"构建书籍测试集\book_serendipity_api_prompts.jsonl"
problem_ids = []

with open(file_path, "r", encoding="utf-8") as f:
    import json
    for line in f:
        if "书名：Unknown" in line:
            data = json.loads(line)
            problem_ids.append(data["custom_id"])

print(f"🚨 找出 {len(problem_ids)} 条问题数据的 ID：")
for pid in problem_ids:
    print(pid)