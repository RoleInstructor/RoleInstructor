import os
import re
import time
import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage



def parse_responses(file_path):
    dialogues = {}
    current_speaker = None
    current_response = []

    with open(file_path, 'r', encoding='gbk') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("（回答）"):
                if current_speaker:
                    response_text = "".join(current_response).strip('"')
                    if current_speaker in dialogues:
                        dialogues[current_speaker] += " " + response_text
                    else:
                        dialogues[current_speaker] = response_text
                    current_response = []

                speaker_part, response = line.split("说：", 1)
                speaker_part_clean = speaker_part.replace("（回答）", "").strip() 
                speaker = speaker_part_clean.split("对")[0].strip()            
                
                current_speaker = speaker
                current_response = [response]

            elif current_speaker and not line.startswith("（问题）"):
                current_response.append(line) 

    if current_speaker:
        response_text = "".join(current_response).strip('"')
        if current_speaker in dialogues:
            dialogues[current_speaker] += " " + response_text
        else:
            dialogues[current_speaker] = response_text

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
    split_dialogues = {}
    for speaker, response in dialogues.items():
        sentences = split_response(response)
        combined = []
        for i in range(0, len(sentences), 2):
            combined.append(" ".join(sentences[i:i+2]))
        split_dialogues[speaker] = combined
    return split_dialogues

def evaluate_response(response, style1, style2, retries=3):
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
            evaluation = chat.invoke([message]).content.strip()
            #print(f"模型返回: {evaluation}")
            
            reason_match = re.search(r'原因：(.*?)结论：', evaluation, re.DOTALL)
            conclusion_match = re.search(r'结论：(.*)', evaluation, re.DOTALL)
            
            if reason_match and conclusion_match:
                reason = reason_match.group(1).strip()
                conclusion = conclusion_match.group(1).strip()
            else:
                reason = "未找到原因"
                conclusion = "未找到结论"
            
            parts = [part.strip() for part in conclusion.replace('，', ',').split(',')]
            eval1 = "是" if parts[0] == "是" else "否"  # 第一个风格的评估
            eval2 = "是" if len(parts) > 1 and parts[1] == "是" else "否"  # 第二个风格的评估
            
            print(f"原因: {reason}")
            print(f"评估结果: {eval1}, {eval2}")
            
            return eval1, eval2,reason
        except Exception as e:
            print(f"评估失败: {e},重试 {attempt+1}/{retries}...")
            time.sleep(2**attempt)
    return "未知", "未知","API 失败"


def main():
    input_file = "run3_4o.txt"
    output_file = "4o_script_3_style_eval.txt"

    print("开始解析对话文件...")
    dialogues = parse_responses(input_file)
    print("解析完成。")

    print("开始分割回答...")
    split_dialogues = split_all_responses(dialogues)
    print("分割完成。")

    speaking_styles = {
        '任风': ['每句话（以句号分隔）说话前都叹气。', '每句话（以句号分隔）都会有一个成语。'],
        '吴信刚': ['每句话（以句号分隔）都以“咳”开头。', '每句话（以句号分隔）都有不礼貌的词语。'],
        '曾聪齐': ['每句话（以句号分隔）都夹杂英文单词。', '自称时，全部以“本人”自称。'],
        '黄齐生': ['每句话（以句号分隔）都夹杂英文单词。', '在每句话（以句号分隔）开头加入“嘶...”的思考词。'],
    }

    results = []

    for speaker, responses in split_dialogues.items():
        styles = speaking_styles.get(speaker, ['未知风格', '未知风格'])
        total_score = 0
        total_responses = len(responses)
        style1_correct = 0
        style2_correct = 0

        print(f"开始评估 {speaker} 的回答...")
        for response in responses:
            try:
                eval1, eval2, reason = evaluate_response(response, styles[0], styles[1])
                
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
                
                results.append(f"说话人: {speaker}\n回应: {response}\n风格1评估: {eval1}\n风格2评估: {eval2}\n原因: {reason}\n得分: {score}\n---")
            except Exception as e:
                print(f"评估失败: {e}")
                results.append(f"说话人: {speaker}\n回应: {response}\n评估结果: 评估失败\n原因: 无\n得分: 0\n---")

        average_score = total_score / total_responses if total_responses > 0 else 0
        results.append(f"说话人: {speaker}\n总回答次数: {total_responses}\n风格1正确次数: {style1_correct}\n风格2正确次数: {style2_correct}\n总得分: {total_score}\n平均分: {average_score:.2f}\n---")
        print(f"{speaker} 的评估完成。")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"评估结果已保存到 {output_file}")

if __name__ == '__main__':
    main()
