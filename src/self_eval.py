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

            if line.startswith(("（问题）", "（回答）")):
                if current_speaker and current_type:
                    # 确保字典结构存在
                    # if current_speaker not in dialogues:
                    #     dialogues[current_speaker] = {'questions': [], 'answers': []}
                    full_content = "".join(current_content).strip('"')
                    target_list = dialogues[current_speaker]['questions' if current_type == 'question' else 'answers']
                    target_list.append(full_content)
                    current_content = []

                current_type = 'question' if line.startswith("（问题）") else 'answer'
                speaker_match = re.search(r"（(?:问题|回答)）\s*([^对]+?)\s*(?:对.*?)?(?:说|问)：", line)
                if speaker_match:
                    current_speaker = speaker_match.group(1).strip()
                    if current_speaker == "主角玩家":
                        current_speaker = None
                        current_type = None
                        current_content = []
                        continue
                    if current_speaker not in dialogues:
                        dialogues[current_speaker] = {'questions': [], 'answers': []}
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
            combined = [" ".join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
            split_answers.extend(combined)
        
        processed[speaker] = {
            'questions': data['questions'],
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
    
    input_file = "run6_4o.txt"
    output_file = "4o_script_6_self_eval.txt"

    print("开始解析对话文件...")
    dialogues = parse_dialogues(input_file)
    print("解析完成。")

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


    results = []
    
    for speaker in split_dialogues:

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
            if "是" in evaluation:
                tmp_score = 0
                invalid_count += 1
            elif "否" in evaluation:
                tmp_score = 1
                valid_count += 1
            else:
                raise "评测失败：输出不符合要求"
            script_score += tmp_score

            results.append(
                f"角色: {speaker}\n"
                f"叙述内容: {content}\n"
                f"相关叙述: {relevant_text}\n"
                f"评估: {evaluation}\n"
                f"得分: {tmp_score}\n"
                "---"
            )

            document = Document(page_content=content)
            agent_script_rag[speaker].add_documents([document])

        score_card['评分明细'].append({
            '评估数量': len(contents),
            '有效项': valid_count,
            '无效项': invalid_count,
            '得分': script_score
        })
        
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

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f"\n评估结果已保存到 {output_file}")



    
if __name__ == '__main__':
    main()
