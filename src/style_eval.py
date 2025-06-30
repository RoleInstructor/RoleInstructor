import os
import re
import time
import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage



def parse_responses(file_path):
    """
    解析对话文件，提取每个人的完整回答。
    
    参数:
        file_path (str): 对话文件的路径。
        
    返回:
        dict: 键为说话者名字，值为其所有回答合并成的一个大段。
    """
    dialogues = {}
    current_speaker = None
    current_response = []

    with open(file_path, 'r', encoding='gbk') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 检查是否是回答行
            if line.startswith("（回答）"):
                # 如果当前已经有回答，保存当前回答并重置
                if current_speaker:
                    # 将当前回答列表合并成一个字符串
                    response_text = "".join(current_response).strip('"')
                    # 保存到对话中
                    if current_speaker in dialogues:
                        dialogues[current_speaker] += " " + response_text
                    else:
                        dialogues[current_speaker] = response_text
                    current_response = []

                # 提取说话者和回答内容
                speaker_part, response = line.split("说：", 1)
                # 移除前缀并提取说话者名字
                speaker_part_clean = speaker_part.replace("（回答）", "").strip()  # 新增清理步骤
                speaker = speaker_part_clean.split("对")[0].strip()              # 正确提取名字
                
                current_speaker = speaker
                current_response = [response]  # 重置当前回答

            # 如果是回答的后续内容
            elif current_speaker and not line.startswith("（问题）"):
                current_response.append(line)  # 继续添加回答内容

    # 处理最后一个回答
    if current_speaker:
        # 将当前回答列表合并成一个字符串
        response_text = "".join(current_response).strip('"')
        # 保存到对话中
        if current_speaker in dialogues:
            dialogues[current_speaker] += " " + response_text
        else:
            dialogues[current_speaker] = response_text

    return dialogues

def split_response(response):
    """
    将回答内容按句号和感叹号分割成句子。
    
    参数:
        response (str): 回答内容。
        
    返回:
        list: 分割后的句子列表。
    """
    # 按句号和感叹号分割句子
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
        dialogues (dict): 包含每个人回答的字典。
        
    返回:
        dict: 键为说话者名字，值为其分割后的回答列表（每两句话为一个整体）。
    """
    split_dialogues = {}
    for speaker, response in dialogues.items():
        # 按句号和感叹号分割句子
        sentences = split_response(response)
        # 按每两句话合并
        combined = []
        for i in range(0, len(sentences), 2):
            combined.append(" ".join(sentences[i:i+2]))
        split_dialogues[speaker] = combined
    return split_dialogues

def evaluate_response(response, style1, style2, retries=3):
    """
    使用 ChatGPT-4o 评估回应是否符合两种预期说话风格，并提取评估结果。
    
    参数:
        response (str): 说话者的回应。
        style1 (str): 第一种预期的说话风格。
        style2 (str): 第二种预期的说话风格。
        
    返回:
        tuple: (评估结果1, 评估结果2)
    """
    chat = ChatOpenAI(model_name='gpt-4o', temperature=0.1)  # 确保模型名称正确
    prompt = (
        "请首先将以下回应分割成单独的句子，注意句子以句号（。）结尾。然后评估回应整体是否符合指定的两种说话风格。\n"
        f"回应：{response}\n"
        f"说话风格1：{style1}\n"
        f"说话风格2：{style2}\n"
        "输出格式为：\n"
        "原因：此处阐述原因\n"
        "结论：此处输出结果，请分别用'是'或'否'回答，并用逗号分隔，不要添加任何其他文字。例如：是,否"
    )
    message = HumanMessage(content=prompt)
    for attempt in range(retries):
        try:
            # 获取模型返回结果并去除首尾空格
            evaluation = chat.invoke([message]).content.strip()
            #print(f"模型返回: {evaluation}")
            
            # 使用正则表达式提取原因和结论
            reason_match = re.search(r'原因：(.*?)结论：', evaluation, re.DOTALL)
            conclusion_match = re.search(r'结论：(.*)', evaluation, re.DOTALL)
            
            if reason_match and conclusion_match:
                reason = reason_match.group(1).strip()
                conclusion = conclusion_match.group(1).strip()
            else:
                reason = "未找到原因"
                conclusion = "未找到结论"
            
            # 从结论中提取评估结果
            parts = [part.strip() for part in conclusion.replace('，', ',').split(',')]
            eval1 = "是" if parts[0] == "是" else "否"  # 第一个风格的评估
            eval2 = "是" if len(parts) > 1 and parts[1] == "是" else "否"  # 第二个风格的评估
            
            # 打印原因和评估结果
            print(f"原因: {reason}")
            print(f"评估结果: {eval1}, {eval2}")
            
            # 只返回评估结果
            return eval1, eval2,reason
        except Exception as e:
            print(f"评估失败: {e},重试 {attempt+1}/{retries}...")
            time.sleep(2**attempt)
    return "未知", "未知","API 失败"

# 设置 OpenAI API 密钥
os.environ["OPENAI_API_KEY"] = "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"


def main():
    # 输入文件路径
    input_file = "run3_4o.txt"
    # 输出文件路径
    output_file = "4o_script_3_style_eval.txt"

    # 1. 提取每个人的所有回答，形成一个大段
    print("开始解析对话文件...")
    dialogues = parse_responses(input_file)
    print("解析完成。")

    # 2. 对每个人的回答进行分割
    print("开始分割回答...")
    split_dialogues = split_all_responses(dialogues)
    print("分割完成。")

    # 定义每个角色的两种说话风格
    speaking_styles = {
        '任风': ['每句话（以句号分隔）说话前都叹气。', '每句话（以句号分隔）都会有一个成语。'],
        '吴信刚': ['每句话（以句号分隔）都以“咳”开头。', '每句话（以句号分隔）都有不礼貌的词语。'],
        '曾聪齐': ['每句话（以句号分隔）都夹杂英文单词。', '自称时，全部以“本人”自称。'],
        '黄齐生': ['每句话（以句号分隔）都夹杂英文单词。', '在每句话（以句号分隔）开头加入“嘶...”的思考词。'],
    }

    # 存储评估结果
    results = []

# 遍历每个说话者及其回答
    for speaker, responses in split_dialogues.items():
        styles = speaking_styles.get(speaker, ['未知风格', '未知风格'])
        total_score = 0
        total_responses = len(responses)
        style1_correct = 0
        style2_correct = 0

        print(f"开始评估 {speaker} 的回答...")
        for response in responses:
            try:
                # 评估回应是否符合两种风格
                eval1, eval2, reason = evaluate_response(response, styles[0], styles[1])
                
                # 计算得分
                if eval1 == "是" and eval2 == "是":
                    score = 1
                    style1_correct += 1
                    style2_correct += 1
                elif eval1 == "是" or eval2 == "是":
                    score = 0.5
                    if eval1 == "是":
                        style1_correct += 1
                    if eval2 == "是":
                        style2_correct += 1
                else:
                    score = 0

                total_score += score
                
                # 记录结果
                results.append(f"说话人: {speaker}\n回应: {response}\n风格1评估: {eval1}\n风格2评估: {eval2}\n原因: {reason}\n得分: {score}\n---")
            except Exception as e:
                print(f"评估失败: {e}")
                results.append(f"说话人: {speaker}\n回应: {response}\n评估结果: 评估失败\n原因: 无\n得分: 0\n---")

        # 计算平均分
        average_score = total_score / total_responses if total_responses > 0 else 0
        # 添加总结信息
        results.append(f"说话人: {speaker}\n总回答次数: {total_responses}\n风格1正确次数: {style1_correct}\n风格2正确次数: {style2_correct}\n总得分: {total_score}\n平均分: {average_score:.2f}\n---")
        print(f"{speaker} 的评估完成。")

    # 将结果保存到 TXT 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"评估结果已保存到 {output_file}")

if __name__ == '__main__':
    main()