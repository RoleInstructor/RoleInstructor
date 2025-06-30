import json
import re

def parse_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    qa_pairs = []
    pattern = re.compile(
        r'（问题）(.*?)对(.*?)说："(.*?)"\s*（回答）(.*?)对(.*?)说："(.*?)"',
        re.DOTALL
    )

    for match in pattern.findall(text):
        q_asker = match[0].strip()
        q_role = match[1].strip()
        q_content = match[2].strip()
        a_role = match[3].strip()
        a_asker = match[4].strip()
        a_content = match[5].strip()

        if q_asker == a_asker and q_role == a_role:
            qa_pairs.append({
                "角色": q_role,
                "提问人": q_asker,
                "提问内容": q_content,
                "回答": a_content
            })
    return qa_pairs

def process_data(json_path, qa_pairs):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    merged = {}
    for entry in json_data:
        answer_part = entry["回答"]
        found = None
        
        for qa in qa_pairs:
            if answer_part in qa["回答"]:
                found = qa
                break
        
        if not found:
            print(answer_part)
            print('---')
            continue

        full_answer = found["回答"]
        if full_answer not in merged:
            merged[full_answer] = {
                "角色": found["角色"],
                "提问人": found["提问人"],
                "提问内容": found["提问内容"],
                "回答": full_answer,
                "原因": [entry["原因"]],
                "结论": entry["结论"]
            }
        else:
            merged[full_answer]["原因"].append(entry["原因"])

    # # 转换评估原因为换行连接的字符串
    result = []
    for item in merged.values():
        item["原因"] = '\n'.join(item["原因"])
        result.append(item)
    return result

def main():
    # 文件路径需要根据实际情况修改
    qa_pairs = parse_txt("run4_glm.txt")
    processed = process_data("z_output_script.json", qa_pairs)
    
    with open("z_reflexion_script.json", 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()