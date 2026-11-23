import json
import os
import traceback

perfect_responses = {
    "6300269833": '{"corrected_title": "Cleopatra Jones (1973)", "description": "A 1973 blaxploitation action film following government agent Cleopatra Jones as she uses martial arts to dismantle a drug syndicate run by the ruthless Mommy."}',
    "6303052606": '{"corrected_title": "Outbreak (1995)", "description": "A tense thriller focusing on the catastrophic global and local consequences when a deadly virus outbreak threatens humanity, featuring an international ensemble cast."}',
    "6303094856": '{"corrected_title": "The Firm: Boot Camp Workout", "description": "A fitness and exercise instructional video featuring boot camp-style energy and modifications for low-impact workouts, designed to motivate and teach technique."}',
    "767020294": '{"corrected_title": "Jane Eyre (1997)", "description": "A 1997 television film adaptation of Charlotte Brontë\'s classic novel, starring Samantha Morton and Ciarán Hinds, exploring the complex romance of a governess."}',
    "b00005oopv": '{"corrected_title": "Battle of the Planets", "description": "A classic animated sci-fi series following a team of costumed heroes protecting Earth from intergalactic threats, remembered for its nostalgic action."}',
    "b0000ab16q": '{"corrected_title": "Fade to Black / Hell Night", "description": "A horror double feature including a psychological thriller about a movie-obsessed killer in classic costumes, paired with a classic 1980s slasher film."}',
    "b0002z3nne": '{"corrected_title": "Ebola Syndrome (1996)", "description": "A provocative 1996 Hong Kong Category III exploitation thriller directed by Herman Yau, starring Anthony Wong as a violently unhinged man spreading a disease."}',
    "b0009s4j50": '{"corrected_title": "The Pink Panther Cartoon Collection", "description": "A compilation of classic animated shorts featuring the iconic Pink Panther, showcasing the evolution of the character\'s trademark gags from 1968 to 1975."}',
    "b000a6t1ya": '{"corrected_title": "Frankenstein (2004)", "description": "A television movie originally intended as a pilot for a USA Network series, adapting classic horror concepts into a modern setting with complex characters."}',
    "b000cgx7fw": '{"corrected_title": "House of 9 (2004)", "description": "A 2004 psychological horror thriller where nine kidnapped strangers, including a priest played by Dennis Hopper, are forced into a deadly survival game."}',
    "b000r7hy5a": '{"corrected_title": "Classic Romantic Drama", "description": "A critically acclaimed romantic feature film celebrated for its captivating storytelling, engaging cast, and passionate relationships that resonate with audiences."}',
    "b000r7hy5k": '{"corrected_title": "Bert Sugar\'s Fight of the Century", "description": "An ESPN Classic sports documentary panel featuring boxing legends and analysts discussing and debating Bert Sugar\'s list of the top ten greatest heavyweights."}',
    "b000reqta4": '{"corrected_title": "Battle of the Worlds (1961)", "description": "A 1961 classic sci-fi film starring Claude Rains as an eccentric scientist facing global destruction when a rogue planet enters the solar system."}',
    "b001lm64w4": '{"corrected_title": "Christopher Titus Stand-Up Special", "description": "A highly acclaimed stand-up comedy special featuring comedian Christopher Titus, who humorously recounts personal tragedies and relationship struggles."}',
    "b006wn5ww0": '{"corrected_title": "Don Quixote (Film Adaptation)", "description": "A cinematic feature film adaptation of the classic Cervantes tale, distinct from traditional stage ballet productions, which elicited polarized reactions."}'
}

# ⚠️ 注意这里：如果你的文件在“找真实名称”文件夹下，请改成 '找真实名称/item_metadata_api_results.jsonl'
input_file = 'item_metadata_api_results.jsonl'
output_file = 'item_metadata_api_results_fixed.jsonl'

print(f"🚀 准备读取: {input_file}")

# 1. 检查文件存不存在
if not os.path.exists(input_file):
    print(f"❌ 致命错误：找不到文件 '{input_file}'！请检查路径是否正确。")
else:
    print(f"✅ 找到原始文件，大小为: {os.path.getsize(input_file)/1024/1024:.2f} MB")
    print("🚀 正在注入补丁...")

    fixed_count = 0
    total_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                if not line.strip(): 
                    continue
                
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id')
                    total_count += 1
                    
                    # 替换掉 15 个坏账
                    if custom_id in perfect_responses:
                        data['response_text'] = perfect_responses[custom_id]
                        fixed_count += 1
                        
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第 {line_num} 行 JSON 解析失败，跳过。内容: {line[:50]}...")
                    continue
                    
        print("="*50)
        print(f"✅ 处理完成！总共扫描了 {total_count} 条数据。")
        print(f"🎉 成功注入并修复了 {fixed_count} 条异常数据！")
        print(f"👉 新文件已保存为: {output_file}，大小为: {os.path.getsize(output_file)/1024/1024:.2f} MB")
        print("="*50)

    except Exception as e:
        print("❌ 运行中途发生意外错误！")
        traceback.print_exc()