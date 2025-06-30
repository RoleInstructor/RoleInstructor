import json

def parse_txt_to_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = [b.strip() for b in content.strip().split('---') if b.strip()]
    results = []

    for block in blocks:
        entry = {}
        for line in block.split('\n'):
            if ': ' in line:
                key, value = line.split(': ', 1)
                entry[key.strip()] = value.strip()
            elif '：' in line:
                key, value = line.split('：', 1)
                entry[key.strip()] = value.strip()
        if entry.get('结论') == '不满足':
            # 提取被提问人（从问题第一个冒号前提取）
            answer = entry.get('回答', '')

            results.append({
                '角色': entry.get('角色', '未知角色'),
                '回答': answer,
                '原因': entry.get('原因', '未说明原因'),
                '结论': entry.get('结论', '未说明结论')
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f'处理完成，共找到{len(results)}条不符合数据')

# 使用示例
parse_txt_to_json('z_style_eval2.txt', 'z_output_style2.json')