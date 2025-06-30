import os
import re
import time
from requests.exceptions import ConnectionError, Timeout
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from openai import APIError, APIConnectionError, RateLimitError

character_data = {   
    '朱琪': {
        'task': "查明类：查明黄见见来这里的目的是什么。查明类：查明你提问的人是否注意到地毯是什么颜色。",
    },
    '苏永晴': {
        'task': "查明类：查明朱琪为什么看到灰衣人走向楼梯，就确定他是往上走了。查明类：查明你提问的人是否知道你哥哥的情况。",
    },
    '李玉': {
        'task': "查明类：查明你提问的人是否知道电梯在不同情况下按下同一个按钮，电梯是否会去同一层。查明类：查明黄见见意识到的秘密是什么。",
    },
    '黄见见': {
        'task': "查明类：查明李玉来这里的目的是什么。查明类：查明你提问的人什么时候使用过电梯。",
    }
}   



def parse_dialogues(file_path):
    dialogues = {}
    current_speaker = None
    current_type = None
    current_content = []

    with open(file_path, 'r', encoding='gbk') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(("（问题）", "（回答）")):
                if current_speaker and current_type:
                    if current_speaker not in dialogues:
                        dialogues[current_speaker] = {'questions': [], 'answers': []}
                    full_content = "".join(current_content).strip('"')
                    target_list = dialogues[current_speaker]['questions' if current_type == 'question' else 'answers']
                    target_list.append(full_content)
                    current_content = []

                current_type = 'question' if line.startswith("（问题）") else 'answer'
                
                speaker_match = re.search(r"（(?:问题|回答)）\s*([^对]+?)\s*(?:对.*?)?(?:说|问)：", line)
                if speaker_match:
                    current_speaker = speaker_match.group(1).strip()
                    if current_speaker not in dialogues:
                        dialogues[current_speaker] = {'questions': [], 'answers': []}
                    if current_speaker == "主持人":
                        current_speaker = None
                        current_type = None
                        current_content = []
                        continue
                    content_part = line.split("：", 1)[-1].strip('" ')
                    current_content.append(content_part)
                else:
                    current_speaker = None 

            elif current_speaker and current_type and current_speaker in dialogues:
                current_content.append(line)

        if current_speaker and current_type and current_speaker in dialogues:
            full_content = "".join(current_content).strip('"')
            dialogues[current_speaker]['questions' if current_type == 'question' else 'answers'].append(full_content)

    return dialogues


def split_response(response):
    sentences = []
    temp = ""
    for char in response:
        temp += char
        if char in ("。", "！"):
            sentences.append(temp.strip())
            temp = ""
    if temp:
        sentences.append(temp.strip())
    return sentences

def split_all_responses(dialogues):
    processed = {}
    for speaker, data in dialogues.items():
        split_answers = []
        for answer in data['answers']:
            sentences = split_response(answer)
            combined = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
            split_answers.extend(combined)
        
        processed[speaker] = {
            'questions': data['questions'], 
            'answers': split_answers
        }
    return processed


def save_parsed_results(split_dialogues, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*40 + " 解析结果报告 " + "="*40 + "\n\n")
        
        for speaker, data in split_dialogues.items():
            f.write(f"【{speaker}】\n")
            
            if data['questions']:
                f.write(f"◆ 问题列表（共{len(data['questions'])}个）:\n")
                for i, q in enumerate(data['questions'], 1):
                    f.write(f"  问题{i}: {q}\n")
            else:
                f.write("◆ 未提出任何问题\n")
            
            if data['answers']:
                f.write(f"\n◇ 回答列表（共{len(data['answers'])}段）:\n")
                for i, a in enumerate(data['answers'], 1):
                    f.write(f"  回答段落{i}: {a}\n")
            else:
                f.write("\n◇ 未作出任何回答\n")
            
            f.write("\n" + "-"*60 + "\n\n")
        
        f.write("="*40 + " 报告结束 " + "="*40)


def extract_task_types(task_str):
    task_types = []
    for part in task_str.split('。'):
        part = part.strip()
        if not part:
            continue
        if part.startswith("查明类："):
            task_types.append(('查明', part[4:]))
        elif part.startswith("隐瞒类："):
            task_types.append(('隐瞒', part[4:]))
    return task_types

def evaluate_with_context(speaker, response, task_type, content_type, content, task_content):

    chat = ChatOpenAI(model_name='gpt-4o',temperature=0.1)
    max_retries = 5
    retry_delay = 5  # 等待5秒后重试   
    if task_type == "查明":
        prompt = f"""
你是一个善于猜测某位角色的具体任务的玩家，请你根据{speaker}的叙述内容：“{content}”；
是否能猜测出{speaker}的任务是：“{task_content}”；
判断规则为：只根据现有上下文直接判断
输出格式为：
原因：此处阐述原因
结论：此处输出是或否"""
    else:
        prompt = f"""
你是一个善于判断文本关联性的专家，请你判断文本1：“{task_content}”；
是否和文本2：“{content}”有关；
判断规则为：只根据现有上下文直接判断
输出格式为：
原因：此处阐述原因
结论：此处输出是或否"""

    for attempt in range(max_retries):
        try:
            message = HumanMessage(content=prompt)
            response = chat.invoke([message])
            return response.content.strip()[-2:]  # 取后两个字
        except (ConnectionError, Timeout, APIError, APIConnectionError, RateLimitError) as e:
            # 处理这些异常，进行重试
            delay = retry_delay * (2 ** attempt)  # 指数退避
            print(f"遇到错误 {e}, 第{attempt+1}次重试，等待{delay}秒...")
            time.sleep(delay)
        except Exception as e:
            print(f"评估失败，不可重试的错误: {str(e)[:50]}")
            return "失败"
    print("已达到最大重试次数，评估失败。")
    return "失败"
def main():    
    input_file = "run6_4o.txt"
    output_file = "4o_script_6_task_eval.txt"

    print("开始解析对话文件...")
    dialogues = parse_dialogues(input_file)
    print("解析完成。")

    print("开始分割回答...")
    split_dialogues = split_all_responses(dialogues)
    print("分割完成。")

    results = []
    
    for speaker in split_dialogues:
        if speaker not in character_data:
            continue
            
        data = character_data[speaker]
        tasks = extract_task_types(data['task'])
        
        score_card = {
            '总得分': 0,
            '任务明细': []
        }
        
        print(f"\n评估 {speaker}...")
        
        for task_type, task_desc in tasks:
            content_type = '问题' if task_type == "查明" else '回答'
            contents = split_dialogues[speaker]['questions' if task_type == "查明" else 'answers']
            
            task_score = 0
            valid_count = 0
            invalid_count = 0
            
            print(f"处理 {task_type} 任务：{task_desc}")
            
            for content in contents:
                evaluation = evaluate_with_context(
                    speaker=speaker,
                    response=content,
                    task_type=task_type,
                    content_type=content_type,
                    content=content,
                    task_content=task_desc
                )
                
                if task_type == "查明":
                    if "是" in evaluation :
                        score = 1
                        valid_count += 1
                    else:
                        score = 0
                        invalid_count += 1
                elif task_type == "隐瞒":
                    if "是" in evaluation :
                        score = -1
                        invalid_count += 1
                    else:
                        score = 1
                        valid_count += 1   
                    
                
                task_score += score
                
                results.append(
                    f"角色: {speaker}\n"
                    f"任务类型: {task_type}\n"
                    f"{content_type}: {content}\n"
                    f"评估: {evaluation}\n"
                    f"得分: {score}\n"
                    "---"
                )
            
            score_card['总得分'] += task_score
            score_card['任务明细'].append({
                '任务描述': task_desc,
                '类型': task_type,
                '评估数量': len(contents),
                '有效项': valid_count,
                '无效项': invalid_count,
                '得分': task_score
            })
        
        summary = [
            f"\n【{speaker}综合评估】",
            f"总得分: {score_card['总得分']}",
            "任务明细："
        ]
        
        for detail in score_card['任务明细']:
            summary.append(
                f"• {detail['类型']}任务：{detail['任务描述']}\n"
                f"  评估{detail['类型']}数: {detail['评估数量']}\n"
                f"  有效项: {detail['有效项']} / 无效项: {detail['无效项']}\n"
                f"  得分: {detail['得分']}"
            )
        
        results.extend(summary)
        results.append("------------------\n")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"\n评估结果已保存到 {output_file}")



    
if __name__ == '__main__':
    main()
