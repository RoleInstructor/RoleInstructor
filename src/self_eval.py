import os
import re
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import faiss
import json
import math
# 修改1：修改API名称以及KEY
os.environ["OPENAI_API_KEY"] = "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"
chat = ChatOpenAI(model='gpt-4o', max_tokens=8192, temperature=0.1)
def parse_dialogues(file_path):
    dialogues = {}
    current_speaker = None
    current_type = None
    current_content = []

    with open(file_path, 'r', encoding='GBK') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 检测新段落开始
            if line.startswith(("（问题）", "（回答）")):
                # 保存前一个段落（增加空值检查）
                if current_speaker and current_type:
                    # 确保字典结构存在
                    # if current_speaker not in dialogues:
                    #     dialogues[current_speaker] = {'questions': [], 'answers': []}
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
                    # 过滤主持人
                    if current_speaker == "主角玩家":
                        current_speaker = None
                        current_type = None
                        current_content = []
                        continue
                    # 新增：立即初始化数据结构
                    if current_speaker not in dialogues:
                        dialogues[current_speaker] = {'questions': [], 'answers': []}
                    # 提取内容部分
                    content_part = line.split("：", 1)[-1].strip('" ')
                    current_content.append(content_part)
                else:
                    current_speaker = None  # 无法解析时重置

            # 处理内容续行（增加空值检查）
            elif current_speaker and current_type and current_speaker in dialogues:
                current_content.append(line)

        # 添加speaker相关叙述内容到dialogues[current_speaker]
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
            combined = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
            split_answers.extend(combined)
        
        # 保持问题原样
        processed[speaker] = {
            'questions': data['questions'],  # 保持完整
            'answers': split_answers
        }
    return processed
    
def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - score / math.sqrt(2)    

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    ret = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                 search_kwargs={"score_threshold": 0.5, 
                                                "k": 3})
    return ret

def retrieve_docs(memory_retriever, query): # retrieve relevant docs of npc's answer from script
    ref = memory_retriever.get_relevant_documents(query)
    return ref

def _format_memories_to_summarize(relevant_memories: List[Document]) -> str:
    content_strs = set()
    content = []
    for mem in relevant_memories:
        if mem.page_content in content_strs:
            continue
        content_strs.add(mem.page_content)
        content.append(f"{mem.page_content.strip()}")
    return "\n\n".join([f"{mem}" for mem in content])

# 整合角色数据的评估函数（注意：如果相关性低于0.5直接得分）
def evaluate_with_context(speaker, reference, content):
    if len(reference)>0:
        reference = _format_memories_to_summarize(reference)
        prompt = f"""
你是判断大模型生成对话质量的专家, 这是大模型生成的对话内容:{speaker}说：“{content}”；
请你根据之前已经有的对话:"{reference}"；
判断生成两段对话是否矛盾。
判断规则为：判断两段对话内容是否存在明显矛盾(不需要推理就可以得出的称为明显矛盾)，此外，可以忽略信息量不匹配导致的矛盾。
输出格式为：
原因：此处阐述原因
结论：此处输出是或否
     """

        print(prompt)
        try:
            prompt_template = PromptTemplate.from_template(prompt)
            chain = LLMChain(llm=chat, prompt=prompt_template, verbose=False)
            res = chain.run({}).strip()
            print(res)
            return res.split('结论：')[1]  
        except Exception as e:
            print(f"评估失败: {str(e)}")
            return "失败"
    else:
        return "否"

def main():
    
    # 修改2：输入文件路径和输出名称
    input_file = "run6_4o.txt"
    output_file = "4o_script_6_self_eval.txt"

    # 解析对话文件
    print("开始解析对话文件...")
    dialogues = parse_dialogues(input_file)
    print("解析完成。")

    # 分割回答
    print("开始分割回答...")
    split_dialogues = split_all_responses(dialogues)
    print("分割完成。")

    agent_script_rag = {}
    play = "play6_inpc"
    with open('%s.json'%(play), encoding='utf-8') as f:
        data = json.load(f)

    for i in range(len(data['角色'])):
        print("创建",data['角色'][i]['角色名'],"的向量库")
        agent_script_rag[data['角色'][i]['角色名']] = create_new_memory_retriever()


    # 存储评估结果
    results = []
    
    for speaker in split_dialogues:

        # 初始化得分统计
        score_card = {
            '总得分': 0,
            '评分明细': []
        }
        
        print(f"\n评估 {speaker}的自我一致性...")
        contents = split_dialogues[speaker]['answers']

        script_score = 0
        valid_count = 0
        invalid_count = 0

        for content in contents:
            relevant_text = retrieve_docs(agent_script_rag[speaker], content)
            evaluation = evaluate_with_context(
                speaker=speaker,
                reference=relevant_text,
                content=content
            )
            print(evaluation)
            # 计算得分
            if "是" in evaluation:
                tmp_score = 0
                invalid_count += 1
            elif "否" in evaluation:
                tmp_score = 1
                valid_count += 1
            else:
                raise "评测失败：输出不符合要求"
            script_score += tmp_score

            # 记录明细
            results.append(
                f"角色: {speaker}\n"
                f"叙述内容: {content}\n"
                f"相关叙述: {relevant_text}\n"
                f"评估: {evaluation}\n"
                f"得分: {tmp_score}\n"
                "---"
            )

            # 当前说的话放入库中
            document = Document(page_content=content)
            agent_script_rag[speaker].add_documents([document])

        # 记录任务统计
        score_card['评分明细'].append({
            '评估数量': len(contents),
            '有效项': valid_count,
            '无效项': invalid_count,
            '得分': script_score
        })
        
        # 生成总结报告
        summary = [
            f"\n【{speaker}综合评估】",
            "评估明细："
        ]
        
        for detail in score_card['评分明细']:
            summary.append(
                f"  评估数: {detail['评估数量']}\n"
                f"  有效项: {detail['有效项']} / 无效项: {detail['无效项']}\n"
                f"  得分: {detail['得分']}"
            )
        
        results.extend(summary)
        results.append("------------------\n")

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"\n评估结果已保存到 {output_file}")



    
if __name__ == '__main__':
    main()
