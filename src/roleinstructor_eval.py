import os
import re
import time
from requests.exceptions import ConnectionError, Timeout
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


character_data = {   
    '赵雨': {
        'task': "查明类：查明李浩在楼梯间对你说了什么。查明类：查明你的提问对象最后是否拿到了金块。",
    },
    '王萱': {
        'task': "查明类：查明赵雨的笔记本里有什么秘密。查明类：查明你的提问对象是否知道电梯的秘密。",
    },
    '李浩': {
        'task': "查明类：查明你和赵雨在电梯间时，你挑起的话题是什么。查明类：查明你提问的对象来医院的目的是什么。",
    },
    '张晨': {
        'task': "查明类：查明掐你脖子的人是不是同伴王萱。查明类：查明你提问的对象什么时间来到了医院。",
    }
}    



def parse_dialogues(file_path):
    dialogues = {}
    current_speaker = None
    current_type = None
    current_content = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 检测新段落开始
            if line.startswith(("（问题）", "（回答）")):
                # 保存前一个段落（增加空值检查）
                if current_speaker and current_type:
                    # 确保字典结构存在
                    if current_speaker not in dialogues:
                        dialogues[current_speaker] = {'questions': [], 'answers': []}
                    full_content = "".join(current_content).strip('"')
                    target_list = dialogues[current_speaker]['questions' if current_type == 'question' else 'answers']
                    target_list.append(full_content)
                    current_content = []

                # 解析新段落类型
                current_type = 'question' if line.startswith("（问题）") else 'answer'
                
                # 精确提取说话者（修复正则表达式）
                speaker_match = re.search(r"（(?:问题|回答)）\s*([^对]+?)\s*(?:对.*?)?(?:说|问)：", line)
                if speaker_match:
                    current_speaker = speaker_match.group(1).strip()
                    # 新增：立即初始化数据结构
                    if current_speaker not in dialogues:
                        dialogues[current_speaker] = {'questions': [], 'answers': []}
                    # 过滤主持人
                    if current_speaker == "主持人":
                        current_speaker = None
                        current_type = None
                        current_content = []
                        continue
                    # 提取内容部分
                    content_part = line.split("：", 1)[-1].strip('" ')
                    current_content.append(content_part)
                else:
                    current_speaker = None  # 无法解析时重置

            # 处理内容续行（增加空值检查）
            elif current_speaker and current_type and current_speaker in dialogues:
                current_content.append(line)

        # 处理最后一个段落（增加空值检查）
        if current_speaker and current_type and current_speaker in dialogues:
            full_content = "".join(current_content).strip('"')
            dialogues[current_speaker]['questions' if current_type == 'question' else 'answers'].append(full_content)

    return dialogues


# 保持原有函数完全不变
def split_response(response):
    """
    将回答内容按句号和感叹号分割成句子。
    
    参数:
        response (str): 回答内容。
        
    返回:
        list: 分割后的句子列表。
    """
    sentences = []
    temp = ""
    for char in response:
        temp += char
        if char in ("。", "！"):
            sentences.append(temp.strip())
            temp = ""
    # 处理最后一个未分割的句子
    if temp:
        sentences.append(temp.strip())
    return sentences

def split_all_responses(dialogues):
    """
    将每个人的所有回答按句号和感叹号分割，并按每两句话合并。
    
    参数:
        dialogues (dict): 原始对话数据（包含questions和answers）
        
    返回:
        dict: 结构不变，仅answers被处理
    """
    processed = {}
    for speaker, data in dialogues.items():
        # 只处理回答部分
        split_answers = []
        for answer in data['answers']:
            sentences = split_response(answer)
            combined = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)]
            split_answers.extend(combined)
        
        # 保持问题原样
        processed[speaker] = {
            'questions': data['questions'],  # 保持完整
            'answers': split_answers
        }
    return processed


def save_parsed_results(split_dialogues, output_file):
    """将解析和分割后的结果保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*40 + " 解析结果报告 " + "="*40 + "\n\n")
        
        for speaker, data in split_dialogues.items():
            # 角色标题
            f.write(f"【{speaker}】\n")
            
            # 问题输出
            if data['questions']:
                f.write(f"◆ 问题列表（共{len(data['questions'])}个）:\n")
                for i, q in enumerate(data['questions'], 1):
                    f.write(f"  问题{i}: {q}\n")
            else:
                f.write("◆ 未提出任何问题\n")
            
            # 回答输出
            if data['answers']:
                f.write(f"\n◇ 回答列表（共{len(data['answers'])}段）:\n")
                for i, a in enumerate(data['answers'], 1):
                    f.write(f"  回答段落{i}: {a}\n")
            else:
                f.write("\n◇ 未作出任何回答\n")
            
            f.write("\n" + "-"*60 + "\n\n")
        
        f.write("="*40 + " 报告结束 " + "="*40)


# 增强版任务类型提取
def extract_task_types(task_str):
    """从任务描述中提取结构化任务信息"""
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

def check_output_format(response):
    """检查大模型输出是否以'原因：'开头，'结论：'结尾，且结论行包含'是'或'否'"""
    # 去除首尾空白并按行分割，同时过滤空行
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    if len(lines) < 2:  # 至少需要两行有效内容
        print(f"调试信息：行数不足，实际行数={len(lines)}, 内容={repr(response)}")
        return False
    if not lines[0].startswith("原因："):
        print(f"调试信息：首行不以'原因：'开头，实际首行={lines[0]}, 内容={repr(response)}")
        return False
    if not lines[-1].startswith("结论："):
        print(f"调试信息：末行不以'结论：'开头，实际末行={lines[-1]}, 内容={repr(response)}")
        return False
    if "是" not in lines[-1] and "否" not in lines[-1]:
        print(f"调试信息：末行不包含'是'或'否'，实际末行={lines[-1]}, 内容={repr(response)}")
        return False
    return True

def generate_text(prompt, max_length=1000):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

base_model = "./models/Qwen2_5-3b-Instruct"  # 替换为原始基础模型路径
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model).to("cuda")

# # 加载LoRA适配器
# lora_path = "./models/Qwen2_5-1_5b-LoRA"  # 替换为LoRA保存路径
# model = PeftModel.from_pretrained(model, lora_path)

# # 合并LoRA权重到基础模型（可选，根据需要选择是否合并）
# model = model.merge_and_unload()

# 整合角色数据的评估函数
def evaluate_with_context(speaker, response, task_type, content_type, content, task_content):

    max_retries = 5
    retry_delay = 5  # 等待5秒后重试   
    if task_type == "查明":
        prompt = f"""
你是一个善于猜测某位角色的具体任务的玩家，请你根据{speaker}的叙述内容：“{content}”；
是否能猜测出{speaker}的任务有：“{task_content}”；
判断规则为：只根据现有上下文直接判断
输出格式为：
原因：此处阐述原因
结论：此处输出是或否"""
    else:
        prompt = f"""
你是一位精通对话分析的专家，尤其擅长评估角色是否成功完成了隐瞒类任务，请你分析角色的对话：**“{content}”**，判断该角色是否成功地完成了隐瞒任务：**“{task_content}”**；
判断规则为：
1. 只根据提供的角色对话和隐瞒任务进行分析，不引入外部假设。
2. 关注角色的言行是否与其隐瞒任务一致或矛盾,不要在意角色的语言风格（如：爱说脏话等）。
3. 如果角色的言行未暴露隐瞒类任务中的秘密，或巧妙掩饰了任务相关信息，则视为隐瞒成功；反之则失败。
输出格式为：
原因：此处阐述原因
结论：此处输出是或否"""

    #print(prompt)
    for attempt in range(max_retries):
        try:
            full_response = generate_text(prompt).strip()
            print(full_response)
            # 检查输出格式
            # if not check_output_format(full_response):
            #     print(f"格式不正确，第{attempt+1}次重试...")
            #     continue
            reason = full_response.split('\n')[0]  # 取第一行，即“原因：”部分    
            return reason,full_response,response.strip()[-2:]  # 取后两个字
        except Exception as e:
            print(f"评估失败，不可重试的错误: {str(e)[:50]}")
            return "失败"
    print("已达到最大重试次数，评估失败。")
    return "失败"
def main():
    
    
    # 输入文件路径
    input_file = "run4_4o.txt"
    output_file = "task_result_by_base_3b.txt"

    # 解析对话文件
    print("开始解析对话文件...")
    dialogues = parse_dialogues(input_file)
    print("解析完成。")

    # 分割回答
    print("开始分割回答...")
    split_dialogues = split_all_responses(dialogues)
    print("分割完成。")

    # # 3. 保存解析结果到文件
    # debug_file = "parse_debug.txt"
    # print("正在生成解析报告...")
    # save_parsed_results(split_dialogues, debug_file)
    # print(f"解析报告已保存至: {debug_file}")
    

    # # 存储评估结果
    results = []
    
    for speaker in split_dialogues:
        if speaker not in character_data:
            continue
            
        data = character_data[speaker]
        tasks = extract_task_types(data['task'])
        
        # 初始化得分统计
        score_card = {
            '总得分': 0,
            '任务明细': []
        }
        
        print(f"\n评估 {speaker}...")
        
        # 遍历每个子任务
        for task_type, task_desc in tasks:
            # 选择评估内容类型
            content_type = '问题' if task_type == "查明" else '回答'
            contents = split_dialogues[speaker]['questions' if task_type == "查明" else 'answers']
            
            task_score = 0
            valid_count = 0
            invalid_count = 0
            
            print(f"处理 {task_type} 任务：{task_desc}")
            
            for content in contents:
                reason,full_response,evaluation = evaluate_with_context(
                    speaker=speaker,
                    response=content,
                    task_type=task_type,
                    content_type=content_type,
                    content=content,
                    task_content=task_desc
                )
                
                #
                # 计算得分
                if task_type == "查明":
                    if "是" in evaluation :
                        score = 1
                        valid_count += 1
                    else:
                        score = 0
                        invalid_count += 1
                elif task_type == "隐瞒":
                    if "是" in evaluation :
                        score = 1
                        invalid_count += 1
                    else:
                        score = -1
                        valid_count += 1   
                    
                
                task_score += score
                
                # 记录明细
                results.append(
                    f"角色: {speaker}\n"
                    f"任务类型: {task_type}\n"
                    f"{content_type}: {content}\n"
                    f"输出：\n"
                    f"{full_response}\n"
                    f"---"
                )
            
            # 记录任务统计
            score_card['总得分'] += task_score
            score_card['任务明细'].append({
                '任务描述': task_desc,
                '类型': task_type,
                '评估数量': len(contents),
                '有效项': valid_count,
                '无效项': invalid_count,
                '得分': task_score
            })
        
        # 生成总结报告
        # summary = [
        #     f"\n【{speaker}综合评估】",
        #     f"总得分: {score_card['总得分']}",
        #     "任务明细："
        # ]
        
        # for detail in score_card['任务明细']:
        #     summary.append(
        #         f"• {detail['类型']}任务：{detail['任务描述']}\n"
        #         f"  评估{detail['类型']}数: {detail['评估数量']}\n"
        #         f"  有效项: {detail['有效项']} / 无效项: {detail['无效项']}\n"
        #         f"  得分: {detail['得分']}"
        #     )
        
        # results.extend(summary)
        results.append("------------------\n")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"\n评估结果已保存到 {output_file}")



    
if __name__ == '__main__':
    main()
