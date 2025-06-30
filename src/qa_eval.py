from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import time

client = ChatOpenAI(model_name="gpt-4o", temperature=0.1, openai_api_key="YOUR OPENAI KEY")

def evaluate_answer(question, answer):
    prompt = f"""请判断以下回答是否回答了全部的问题，且回答的内容都跟问题有关：

问题：{question}
回答：{answer}

输出格式为：
原因：此处阐述原因
结论：此处输出是或否"""
    
    try:
        response = client.invoke([SystemMessage(content="严谨的评估专家"),HumanMessage(content=prompt)])

        full_response = response.content.strip()
        print(full_response)
        last_two_chars = full_response[-2:]  # 取最后两个字
        
        return 1 if "是" in last_two_chars else 0
    except Exception as e:
        print(f"API调用出错: {str(e)}")
        return 0



def parse_dialogue(file_path):
    dialogues = []
    
    with open(file_path, "r", encoding="gbk") as f:
        lines = [line.strip() for line in f if line.strip()]
        
        i = 0
        while i < len(lines):
            if lines[i].startswith("（问题）"):
                # 解析问题行
                q_header = lines[i][4:lines[i].index("说：")]
                q_role, a_role = q_header.split("对")
                question = lines[i][lines[i].index("说：")+3:].strip('"')
                
                # 查找对应的回答行
                if i+1 < len(lines) and lines[i+1].startswith("（回答）"):
                    a_header = lines[i+1][4:lines[i+1].index("说：")]
                    answer = lines[i+1][lines[i+1].index("说：")+3:].strip('"')
                    
                    dialogues.append({
                        "q_role": q_role.strip(),
                        "a_role": a_role.strip(),
                        "question": question,
                        "answer": answer
                    })
                    i += 2  # 跳过已处理的两行
                else:
                    i += 1
            else:
                i += 1
    return dialogues

def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for role, data in results.items():
            f.write(f"【{role} 评估报告】\n")
            total = len(data["qas"])
            for idx, qa in enumerate(data["qas"], 1):
                f.write(f"\n第{idx}问（来自：{qa['q_role']}）\n")
                f.write(f" 问题：{qa['question']}\n")
                f.write(f" 回答：{qa['answer']}\n")
                f.write(f" 得分：{'✅' if qa['score'] else '❌'}\n")
                f.write("-"*60 + "\n")
            f.write(f" 最终得分：{data['total_score']}/{total}\n\n\n")

def main():
    input_file = "run6_4o.txt"
    output_file = "4o_script_6_qa_eval.txt"
    
    print(" 正在解析对话文件...")
    try:
        dialogues = parse_dialogue(input_file)
        print(f" 成功解析到 {len(dialogues)} 组问答对")
    except Exception as e:
        print(f" 文件解析失败: {str(e)}")
        return
    
    results = {}
    
    print("\n 开始评估回答质量：")
    for dialogue in dialogues:
        a_role = dialogue["a_role"]
        print(f"  正在评估 {a_role} 的回答...")
        
        try:
            score = evaluate_answer(dialogue["question"], dialogue["answer"])
        except Exception as e:
            print(f" 评估异常: {str(e)}")
            score = 0
        
        if a_role not in results:
            results[a_role] = {"total_score": 0, "qas": []}
            
        results[a_role]["total_score"] += score
        results[a_role]["qas"].append({
            "q_role": dialogue["q_role"],
            "question": dialogue["question"],
            "answer": dialogue["answer"],
            "score": score
        })
        
        time.sleep(1.5)  # API速率限制
    
    save_results(results, output_file)
    print(f"\n 评估完成！结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
