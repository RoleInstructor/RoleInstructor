#-*- coding:utf-8 -*-
import time
from logging.config import listen
import os
import re
from datetime import datetime, timedelta
from tokenize import String
from typing import List, Optional, Tuple

from sympy.stats import characteristic_function
from termcolor import colored

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from langchain.base_language import BaseLanguageModel
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, CommaSeparatedListOutputParser

import random
import math
import faiss
import json
import pickle
import string
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter, defaultdict

setting = {
"Cooperation_script": True,   # 修改1：是否为合作类，是则True
"Retrieval" : True,
"Recent_observation" : True,
"Information_Supplement": True,
"Two_Step_Characterized":False,
"Selected_gpt_model" : "gpt-4o",
"Max_tokens" : 5500,
"Play_no." : 'play6_inpc',   # 修改2：评测使用的剧本
"Project_path" : os.path.abspath('.').replace("\\", "/"),
"k" : 5,
"temperature": 0.8,
"num of rounds for first free discussion": 2,
"num of rounds for second free discussion": 3,
"score_threshold" : 0.3,
"max_output_retries":10,
"Self_Examination": True

}

# 修改3：是否需要设置max_token
model_token_limits = {"gpt-4o":16385, "gpt-4o-mini":16385,'gpt-4-1106-preview':100000,'gpt-3.5-turbo-1106':16385}
# 修改4：修改API名称以及对应的API key
os.environ["OPENAI_API_KEY"] = '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'
USER_NAME = "主角玩家" # The name you want to use when interviewing the agent.
selected_gpt_model = setting['Selected_gpt_model']
# 修改5：修改要评测的模型初始化方式
LLM = ChatOpenAI(model=selected_gpt_model,max_tokens=setting['Max_tokens'],temperature = setting['temperature']) # Can be any LLM you want.
global_output_list = []
numbers = [str(i) for i in range(10)]
lowercase_letters = list(string.ascii_lowercase)
greek_letters = ['α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω']
letter_list = numbers + lowercase_letters + greek_letters



class GenerativeAgent(BaseModel):
    """A character with memory and innate characteristics."""

    name: str
    age: int
    role: str
    mission: str
    chat_history: List[str] = []
    clue_dict: Dict[str,List[str]] = {}
    clue_list: List[str] = []
    previous_chat_history: List[str] = []
    players_in_game: List[str] = []
    action_history: List[str] = []
    story_background: str
    character_story: str
    character_timeline: str
    game_rule: str
    game_rule_understanding: str = ''
    player_summaries : str = None
    timeline_summary : str = None
    player2summary : dict = None
    characterInfo_dict :Dict[str, List[Tuple[str, str]]] = {}
    otherPlayersTimeline : Dict[str,str] = {}
    characteristic : str
    # self_history : List[str] = []
    """Current activities of the character."""
    llm: BaseLanguageModel
    memory_retriever: BaseRetriever = None
    """The retriever to fetch related memories."""
    verbose: bool = False




    summary: str = ""  #: :meta private:
    last_refreshed: datetime =Field(default_factory=datetime.now)  #: :meta private:
    daily_summaries: List[str] #: :meta private:
    memory_importance: float = 0.0 #: :meta private:
    max_tokens_limit: int = 5500 #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r'\n', text.strip())
        return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]

    def _is_question_asked_for_timeline(self,content):
        speaker = content.split('对')[0]
        listener = content.split('说')[0].split('对')[1]
        if speaker == self.name:
            return
        response_schemas = [
            ResponseSchema(name="是否询问案发日时间线", description="根据内容判断里面%s是否在询问%s关于案发日时间线相关的问题，如果是的话返回True,不是的话返回False。返回值只能是True或者False。"%(speaker,listener)),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template("内容：{content}。根据内容判断里面%s是否在询问%s关于案发日时间线相关的问题。\n{format_instructions}\n"%(speaker,listener))
            ],
            partial_variables={"format_instructions": format_instructions}
        )
        _input = prompt.format_prompt(content = content)
        n = 0
        while n<=setting['max_output_retries']:
            n +=1
            output = self.llm(_input.to_messages())
            json_result = handling_output_parsing(output=output,output_parser=output_parser)
            if not json_result:
                continue
            else:
                asking_timeline  = json_result["是否询问案发日时间线"]
                break
        return str_to_bool(asking_timeline)

    def add_memory(self, memory_content: str, isclue: bool = False) -> List[str]:
        """Add an observation or memory to the agent's memory."""

        if self.memory_retriever==None:
            return None

        self.chat_history.append(memory_content)
        document = Document(page_content=memory_content)
        result = self.memory_retriever.add_documents([document])
        return result

    def fetch_memories(self, observation: str) -> List[Document]:
        """Fetch related memories."""
        if self.memory_retriever==None:
            return None
        return self.memory_retriever.get_relevant_documents(observation)


    def get_summary(self) -> str:
        """Return a descriptive summary of the agent."""
        if setting['Cooperation_script']:
            self.summary = 	f"角色在游戏中的任务: {self.mission}\n角色人物剧本：{self.character_story}\n角色案发日时间线：{self.character_timeline}\n"
        else:
            self.summary = 	f"游戏中的角色: {self.role}\n角色在游戏中的任务: {self.mission}\n角色人物剧本：{self.character_story}\n角色案发日时间线：{self.character_timeline}\n"

        return (
            f"角色名: {self.name} (年龄: {self.age})"
            +f"\n{self.summary}")




    def _format_memories_to_summarize(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content in content_strs:
                continue
            content_strs.add(mem.page_content)
            content.append(f"{mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])

    def summarize_relationship_with_interlocutor(self, observation: str, inquirer: str) -> str:
        """Summarize memories that are most relevant to an observation."""

        if inquirer !='主角玩家':
            q1 = f"{self.name}和{inquirer}的关系是什么"

            context_str = ''

            context_str2 = '%s的人物剧本：'%self.name + self.character_story + '\n' + context_str
            prompt = PromptTemplate.from_template(
                "{q1}?\n记忆中的上下文：\n{context_str2}\n "
            )
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)


            return "对话人与你的关系是：" + chain.run(q1=q1, context_str2=context_str2.strip()).strip() +  '\n'
        else:
            return "对话人与你的关系是：对话人是主角玩家，本身不在剧本中，他负责引导你们完成游戏。" + '\n'

    def _generate_reaction(
        self,
        observation: str,
        inquirer: str,
        suffix: str,
        require_all_clues: bool = False
    ) -> str:

        agent_summary_description = self.get_summary()

        relevant_memories_str = ''

        clues_str = ''
        if require_all_clues and len(self.clue_list)!=0:
            clues_str +='以下是本案相关线索：\n' + '\n'.join(["“" + clue.split('：“')[1] for clue in self.clue_list])+'\n'

        if self.memory_retriever!=None:
            relevant_memories = self.fetch_memories(observation)
            relevant_memories_str = self._format_memories_to_summarize(relevant_memories)
            
        if setting['Cooperation_script']:
            relationship_with_interlocutor = "未知"
        else:
            relationship_with_interlocutor = self.summarize_relationship_with_interlocutor(observation,inquirer)


        chat_history_str =  '\n' + '\n'.join([observation] + self.chat_history[-1:-4:-1]  if observation not in self.chat_history[-1::] else self.chat_history[-1:-5:-1])

            # relevant_memories_str += chat_history_str

        kwargs = dict(agent_summary_description=agent_summary_description,
                      relevant_memories=relevant_memories_str,
                      agent_name=self.name,
                      observation=observation,
                      story_background=self.story_background,
                      game_rule=self.game_rule,
                      relationship_with_interlocutor = relationship_with_interlocutor,
                      mission=self.mission
                      )

        kwargs["recent_observations"] = chat_history_str if inquirer!='主角玩家' and setting["Recent_observation"] == True else '无\n。'


        """React to a given observation."""
        prompt = PromptTemplate.from_template(
                "{agent_summary_description}"
                + "\n{game_rule}"
                + "\n{story_background}"
                + "\n{agent_name}在之前的对话中与游戏里正在进行的对话相关的内容："
                +"\n{relevant_memories}"
                +"\n游戏过去几轮发生的对话（其中包含来自于其他角色所分享的信息）：{recent_observations}"
                + "\n游戏里正在进行的对话：{observation}"
                + "\n对话人和你的关系：{relationship_with_interlocutor}"
                + "\n着重参考你的任务：{mission}"
                + clues_str
                + "\n\n" + suffix
        )
        consumed_tokens = self.llm.get_num_tokens(prompt.format(**kwargs))

        model_max_tokens = model_token_limits[self.llm.model_name]
        old_max_tokens = self.llm.max_tokens
        self.llm.max_tokens = min(model_max_tokens - consumed_tokens-10,old_max_tokens)
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)

        result = action_prediction_chain.run(**kwargs)

        self.llm.max_tokens = old_max_tokens
        return result.strip()

    def _information_supplement(self,question,previous_reply, victim):
        question_list = question_decomposition(question,self)
        question_list = [remove_numeric_prefix(q.strip()) for q in question_list]
        question_asking_timeline_list  = [q for q in question_list if self._is_question_asked_for_timeline(question.split('："')[0]+ "：\"" +q)]
        players = '，'.join(self.players_in_game)
        numofplayers  = len(self.players_in_game)
        if setting['Cooperation_script']:
            q1 = f"你正在进行一场剧本杀游戏，游戏里一共有{numofplayers}位角色。他们分别是：{players}。每位角色需要通过互相交流还原事件真相并完成自己的任务；以下是角色：{self.name}在游戏里的案发日时间线原文：{self.character_timeline}；请按案发日时间线原文的顺序依次列出{self.name}这个人物角色案发日时间线的信息，每个时间线信息必须是简短又完整（self-contained）的一段话，格式是 什么时间，你做了什么事（越详细越好）。\n每个时间线信息之间用\n隔开。"
        else:
            q1 = f"你正在进行一场剧本杀游戏，游戏里一共有{numofplayers}位角色。他们分别是：{players}。每个角色需要通过互相交流找到杀害{victim}的凶手；以下是角色：{self.name}在游戏里的案发日时间线原文：{self.character_timeline}；请按案发日时间线原文的顺序依次列出{self.name}这个人物角色案发日时间线的信息，每个时间线信息必须是简短又完整（self-contained）的一段话，格式是 什么时间，你做了什么事（越详细越好）。\n每个时间线信息之间用\n隔开。"

        prompt = PromptTemplate.from_template(
            "{q1}\n\n"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        info_list = chain.run(q1=q1).strip().split('\n')
        info_list = [remove_numeric_prefix(i.strip()) for i in info_list if i.strip()]
        info_list = [i for i in info_list if i!='']
        useful_info_list = []
        for sub_ques in question_asking_timeline_list:
            response_schemas = [
                ResponseSchema(name="第%s个时间线信息是否可以用于回答问题的判断结果"%(j), description="请判断时间线信息：%s是否可以用来回答问题。如果可以的话返回True，不可以的话返回False。返回结果只能是 True，False中的一个。"%(info_list[j])) for j in range(len(info_list))


            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            prompt = ChatPromptTemplate(
            messages=[
                    HumanMessagePromptTemplate.from_template("你是一个做阅读理解题目的专家，尤其擅长做是非判断题。给定时间线信息还有一个问题，你需要判断该时间线信息是否可以用于回答该问题。如果可以的话返回True，不可以的话返回False。返回结果只能是 True，False中的一个。判断的时候要遵循以下几个原则：1. 如果问题只是问案发日时间线，而没有提到案发日的某个特定的时间段（比如19:00-21:00）或者时间点（比如16:30）的话，则所有的时间线信息都可以用于回答该问题。2. 如果问题提到了具体案发日的某个时间段，比如15：00-21:00，则只有满足这个时间段的时间线信息可以用于回答问题。3. 如果问题提到了具体案发日的某个时间点，比如18：00，则只有满足这个时间点的时间线信息可以用于回答问题 4. 为了避免遗漏重要时间线信息，你如果不是很确定一个时间线信息是否可以用于回答问题，也返回True而不是False。以下是给定的问题：{question}，请遵循上面的原则对时间线信息做出判断。\n{format_instructions}")
                ],
                input_variables=["question"],
                partial_variables={"format_instructions": format_instructions}
            )
            _input = prompt.format_prompt(question=sub_ques)


            chat_model = self.llm

            info_usefulness_checking_result = None
            n = 0
            while n<=setting['max_output_retries']:
                n +=1
                output = chat_model(_input.to_messages())
                json_result = handling_output_parsing(output=output,output_parser=output_parser)
                if json_result == False:
                    continue
                else:
                    info_usefulness_checking_result  = json_result
                    break


            useful_info_list.extend([idx for idx, info in enumerate(info_list) if str_to_bool(info_usefulness_checking_result["第%s个时间线信息是否可以用于回答问题的判断结果"%(idx)])==True])
        useful_info_list = [info_list[i] for i  in sorted(set(useful_info_list))]
        response_schemas = [
            ResponseSchema(name="回复是否包含第%s个时间线信息的判断结果"%(j), description="请判断时间线信息：%s是否被包含在了角色之前的回答之中。如果包含的话返回True，没有包含的话返回False。返回结果只能是 True，False中的一个。"%(useful_info_list[j])) for j in range(len(useful_info_list))


        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        prompt = ChatPromptTemplate(
        messages=[
                HumanMessagePromptTemplate.from_template("你是一个做阅读理解题目的专家，尤其擅长做是非判断题。给定时间线信息还有角色之前的回答，你需要判断该时间线信息是否被包含在了角色的回答之中。如果包含的话返回True，没有包含的话返回False。返回结果只能是 True，False中的一个。你判断的风格一向非常严格，只有在时间线的所有信息（包括时间点和做的事）都包含在了角色的回答中你才会判定为包含。以下是角色之前的回答：{previous_reply}\n{format_instructions}")
            ],
            input_variables=["previous_reply"],
            partial_variables={"format_instructions": format_instructions}
        )
        _input = prompt.format_prompt(previous_reply=previous_reply)


        chat_model = self.llm

        info_containment_checking_result = None
        n = 0
        while n<=setting['max_output_retries']:
            n +=1
            output = chat_model(_input.to_messages())
            json_result = handling_output_parsing(output=output,output_parser=output_parser)
            if json_result == False:
                continue
            else:
                info_containment_checking_result  = json_result
                break

        missing_info_list = [info for idx, info in enumerate(useful_info_list) if str_to_bool(info_containment_checking_result["回复是否包含第%s个时间线信息的判断结果"%(idx)])==False]
        missing_info_str = '\n'.join(missing_info_list)

        response_schemas = [
            ResponseSchema(name="改进后的回答", description="你补充遗漏的重要信息并修复原有时间线描述后的回答"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        if setting['Cooperation_script']:
            prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template("你正在玩一场剧本杀游戏，游戏规则如下：{game_rule}；以下是你的角色名：{name}，以下是你的人物角色剧本：{story}；还有你的案发日时间线：{timeline}；有人向你提问：{question}，"
                                                         "你之前回答过这个问题，以下是你之前的回答：{previous_reply}；"
                                                         "但是，你之前的回答遗漏了数条重要信息：{missing_info_str}；"
                                                         "请根据给定的问题，并着重参考你的个人任务：{mission}"
                                                         "修改你之前的回答，将遗漏的重要信息补充进你的回答里，并包含所有重要的时间点；"
                                                         "补充后的回答语言整体要通顺流畅。凡是涉及到你的角色：{name}的事，要记得用第一人称的口吻写出来。\n{format_instructions}\n")
            ],
            input_variables=["game_rule","name","question","previous_reply","story","timeline","missing_info_str","mission"],
            partial_variables={"format_instructions": format_instructions}
            )
        else:
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template("你正在玩一场剧本杀游戏，游戏规则如下：{game_rule}；以下是你的角色名：{name}，以下是你的人物角色剧本：{story}；还有你的案发日时间线：{timeline}；有人向你提问：{question}，"
                                                            "你之前回答过这个问题，以下是你之前的回答：{previous_reply}；"
                                                            "但是，你之前的回答遗漏了数条重要信息：{missing_info_str}；"
                                                            "请根据给定的问题，并着重参考你的个人任务：{mission}"
                                                            "修改你之前的回答，修改原则如下："
                                                            "首先判断某条遗漏的信息是否对个人任务有利，若有利则将遗漏的重要信息补充进你的回答里，并包含所有重要的时间点；"
                                                            "但是若某条遗漏的信息对个人任务不利，则通过隐瞒或说谎的方式把它变得有利。"
                                                            "如果你决定隐瞒，可以选择隐瞒整条时间线或时间线内发生的某些情节；"
                                                            "如果你决定说谎，那么只能对事件的目的或发生情节说谎，千万不能对事件的时间、地点和人物撒谎。"
                                                            "补充后的回答语言整体要通顺流畅。凡是涉及到你的角色：{name}的事，要记得用第一人称的口吻写出来。\n{format_instructions}\n")
                ],
                input_variables=["game_rule","name","question","previous_reply","story","timeline","missing_info_str","mission"],
                partial_variables={"format_instructions": format_instructions}
            )
        _input = prompt.format_prompt(game_rule=self.game_rule,name=self.name,mission=self.mission,question=question,previous_reply=previous_reply,story=self.character_story,timeline= self.character_timeline,missing_info_str=missing_info_str)
        n = 0
        improved_reply=""
        while n<=setting['max_output_retries']:
            n +=1
            output = self.llm(_input.to_messages())
            json_result = handling_output_parsing(output=output,output_parser=output_parser)
            if json_result == False:
                continue
            else:
                improved_reply  = json_result["改进后的回答"]
                break

        return improved_reply

    def _self_examination(self, question, improved_reply):
        response_schemas = [
            ResponseSchema(name="检查后的回答", description="对目前的回答进行反思检查"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        if setting['Cooperation_script']:
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        "你正在玩一场剧本杀游戏，以下是你的角色名：{name}，以下是你的人物角色剧本：{story}；还有你的案发日时间线：{timeline}；有人向你提问：{question}，"
                        "你之前回答过这个问题，以下是你之前的回答：{improved_reply}；"
                        "并着重参考你的个人任务：{mission}"
                        "进一步反思检查并修改你的回答，原则如下："
                        "根据角色剧本和案发时间线，确保陈述准确无误；"
                        "最后，凡是涉及到你的角色：{name}的事，要记得用第一人称的口吻写出来。\n{format_instructions}\n")
                ],
                input_variables=["name", "story", "timeline", "question", "improved_reply", "mission"],
                partial_variables={"format_instructions": format_instructions}
            )
        else:
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template(
                        "你正在玩一场剧本杀游戏，以下是你的角色名：{name}，以下是你的人物角色剧本：{story}；还有你的案发日时间线：{timeline}；有人向你提问：{question}，"
                        "你之前回答过这个问题，以下是你之前的回答：{improved_reply}；"
                        "并着重参考你的个人任务：{mission}"
                        "进一步反思检查并修改你的回答，原则如下："
                        "首先判断每个陈述是否对个人任务有利，若有利则根据角色剧本、案发时间线，确保陈述准确无误；"
                        "但是若某条陈述对个人任务不利，则通过隐瞒或说谎的方式把它变得有利。"
                        "如果你决定隐瞒，可以选择隐瞒整条时间线或时间线内发生的某些情节；"
                        "如果你决定说谎，那么只能对事件的目的或发生情节说谎，千万不能对事件的时间、地点和人物撒谎。"
                        "请注意，所有其他角色都可以看到你的回答，因此回答的最后不要解释你隐瞒或欺骗了什么内容。"
                        "最后，凡是涉及到你的角色：{name}的事，要记得用第一人称的口吻写出来。\n{format_instructions}\n")
                ],
                input_variables=["name", "story", "timeline", "question", "improved_reply", "mission"],
                partial_variables={"format_instructions": format_instructions}
            )

        _input = prompt.format_prompt(name=self.name, story=self.character_story,
                                      timeline=self.character_timeline, question=question, 
                                      improved_reply=improved_reply,
                                      mission = self.mission)
        n = 0
        verified_reply=""
        while n <= setting['max_output_retries']:
            n += 1
            output = self.llm(_input.to_messages())
            json_result = handling_output_parsing(output=output, output_parser=output_parser)
            if json_result == False:
                continue
            else:
                verified_reply = json_result["检查后的回答"]
                break

        return verified_reply

    def generate_characteristic_dialogue(self, original_diag, style):
        response_schemas = [
            ResponseSchema(name="个性化生成后的回答", description="根据角色说话风格重新生成对话"),
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(
                    "你正在玩一场剧本杀游戏，以下是你的角色名：{name}，"
                    "你之前回答过这个问题，以下是你之前的回答：{original_diag}；"
                    "请根据给定的问题，并参考你的个人说话风格：{characteristic}"
                    "重新生成你的回答，原则如下："
                    "确保每句话的主要内容和次序不改变，"
                    "只是根据个人说话风格进行简单修改。"
                    "另外，凡是涉及到你的角色：{name}的事，要记得用第一人称的口吻写出来。\n{format_instructions}\n")
            ],
            input_variables=["name","original_diag", "characteristic"],
            partial_variables={"format_instructions": format_instructions}
        )
        _input = prompt.format_prompt(name=self.name, original_diag=original_diag, characteristic=style)
        n = 0
        personalized_reply = ""
        while n <= setting['max_output_retries']:
            n += 1
            output = self.llm(_input.to_messages())
            json_result = handling_output_parsing(output=output, output_parser=output_parser)
            if json_result == False:
                continue
            else:
                personalized_reply = json_result["个性化生成后的回答"]
                break

        return personalized_reply

    def generate_dialogue_response(self, observation: str, inquirer: str, voting:bool = False, victim="") -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            '{agent_name}会说什么？回答问题请用以下格式： #回答#：要说的话 来回答问题。{agent_name}回答%s的问题说 #回答#：\n'%inquirer
        )
        n = 0
        n+=1
        full_result = self._generate_reaction(observation,inquirer, call_to_action_template, require_all_clues=voting)
        result = full_result
        if "#回答#：" in result:
            response_text = result.split("#回答#：")[1:][-1].strip()
            response_text = remove_quotes_and_colons(response_text)
            if voting == False and setting['Information_Supplement']:
                asking_timeline = self._is_question_asked_for_timeline(observation)
                if asking_timeline:
                    old_response_text = response_text
                    response_text = self._information_supplement(observation,response_text,victim)

            if response_text =='':
                print('输出为空'+response_text+'\n')

            return True, f"{response_text}"
        elif "#回答#" in result:
            response_text = result.split("#回答#")[1:][-1].strip()
            response_text = remove_quotes_and_colons(response_text)
            if voting == False and setting['Information_Supplement']:
                asking_timeline = self._is_question_asked_for_timeline(observation)
                if asking_timeline:
                    old_response_text = response_text
                    response_text = self._information_supplement(observation,response_text,victim)

            if response_text =='':
                print('输出为空'+response_text+'\n')
            return True, f"{response_text}"
        else:

            response_text = remove_quotes_and_colons(result)
            if voting == False and setting['Information_Supplement']:
                asking_timeline = self._is_question_asked_for_timeline(observation)
                if asking_timeline:
                    old_response_text = response_text
                    response_text = self._information_supplement(observation,response_text,victim)

            if response_text =='':
                print('输出为空'+response_text+'\n')
            return False, response_text

        return False, response_text

    def generate_voting_results(self, observation, voting_len:int):
        call_to_action_template = (
            '{agent_name}的对于该选择题的回答是什么？。\n'
        )
        result=""
        n=0
        while n<=setting['max_output_retries'] and result.isalpha()==False and len(result)!=voting_len:
            n+=1
            full_result = self._generate_reaction(observation, USER_NAME, call_to_action_template, require_all_clues=True)
            result = full_result.strip()

        return result


    def generate_dialogue_question(self, observation: str, respondent: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            '看见%s的回答，{agent_name}会想要问%s什么问题？请以第一人称的方式来向%s进行提问。用#提问#：要问的问题 的格式来提问。{agent_name}向%s提问 #提问#：'%(respondent.name,respondent.name,respondent.name,respondent.name)
        )
        full_result = self._generate_reaction(observation,respondent.name, call_to_action_template)
        result = full_result.strip().split('\n')[0].replace('#提问#：','').replace('#提问#','')
        result = remove_quotes_and_colons(result)
        return result

    def _take_action_from_choice(self,action):
        def select_ask(self):
            other_players = [p for p in self.players_in_game if p!=self.name]
            players_you_can_ask = '，'.join(other_players)
            description ="在【%s】%s人之间，选出你最想提问的人"%(players_you_can_ask,len(other_players))
            response_schemas = [
                ResponseSchema(name="你想提问的人的名字", description=description),
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            story = self.character_story+'\n'+self.character_timeline
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template("根据你的人物剧本：{story}。着重参考游戏中你的个人任务：{mission}。请选出你最想提问的人。\n{format_instructions}\n")
                ],
				input_variables=["story","mission"],
                partial_variables={"format_instructions": format_instructions}
            )
            _input = prompt.format_prompt(story=story, mission=self.mission)

            n = 0
            while n<=setting['max_output_retries']:
                n+=1
                output = self.llm(_input.to_messages())
                json_result = handling_output_parsing(output=output,output_parser=output_parser)
                if json_result == False:
                    continue
                elif json_result.get("你想提问的人的名字",None) not in other_players:
                    continue
                else:
                    player_to_ask = json_result["你想提问的人的名字"]
                    break

            context_str = ''
            if self.memory_retriever!=None:
                relevant_memories = self.fetch_memories('和%s有关的信息'%player_to_ask) # Fetch things related to the entity-action pair
                context_str = self._format_memories_to_summarize(relevant_memories)

            response_schemas = [
                ResponseSchema(name="你想提问的问题", description="你想要问%s的问题"%player_to_ask),
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()
            prompt = ChatPromptTemplate(
                messages=[
                    HumanMessagePromptTemplate.from_template("根据你的人物故事情节：{story}。并依据你的个人任务:{mission}，以及之前游戏中你目击的和{player_to_ask}相关的信息：{context_str}。请说出你想要问{player_to_ask}的问题\n{format_instructions}\n")
                ],
                input_variables=["story", "mission","player_to_ask","context_str"],
                partial_variables={"format_instructions": format_instructions}
            )
            
            _input = prompt.format_prompt(story=story, mission=self.mission, player_to_ask=player_to_ask, context_str = context_str)

            n = 0
            while n<=setting['max_output_retries']:
                n+=1
                output = self.llm(_input.to_messages())
                json_result = handling_output_parsing(output=output,output_parser=output_parser)
                if json_result == False:
                    continue
                else:
                    question_to_ask = json_result["你想提问的问题"]
                    break

            return player_to_ask, question_to_ask

        switch = {
            'sa': select_ask,
        }

        question_to_ask = switch[action](self)
        return question_to_ask

    def _generate_fd_questions(self,clues_given: bool) -> Tuple[str, str]:

        chosen_action = 'sa'
        self.action_history.append(chosen_action)
        player_to_ask, question_to_ask = self._take_action_from_choice(chosen_action)
        return player_to_ask, remove_quotes_and_colons(question_to_ask)


def count_time_matches(s: str) -> int:

    pattern = r'([01]?[0-9]|2[0-3])[:：][0-5][0-9]'
    matches = re.findall(pattern, s)
    return len(matches)

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    # ret = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=setting['k'],score_threshold = 0.3)
    # ret.search_kwargs = {'k':setting['k']}
    ret = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                 search_kwargs={"score_threshold": setting['score_threshold'], 
                                                "k": setting['k']})
    return ret

def question_decomposition(question,agent):
    q1 = f"请把这句话：{question}里的所有问题或者指令提取出来。\n每个问题或指令之间用\n隔开。"
    prompt = PromptTemplate.from_template(
        "{q1}\n\n"
    )
    chain = LLMChain(llm=agent.llm, prompt=prompt, verbose=agent.verbose)
    question_list = chain.run(q1=q1).strip().split('\n')
    return question_list
def interview_agent(inquirer: str,agent: GenerativeAgent, message: str, voting = False,victim="") -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{inquirer}对{agent.name}说：\"{message}\""

    if inquirer == '主角玩家':
        n = 0
        result = ''
        while n<=setting['max_output_retries']:
            n+=1
            results = agent.generate_dialogue_response(new_message,inquirer,voting=voting,victim=victim)
            if "抱歉" in results:
                continue
            if results == None:
                print()
            else:
                result = results[1]
            if result!='':
                break
        if setting['Self_Examination']:
            examinated_result = agent._self_examination(new_message, result)
        else:
            examinated_result = result


        if setting["Two_Step_Characterized"]:
            style = agent.characteristic.split('并且')
            personalized_result_tmp = agent.generate_characteristic_dialogue(examinated_result, style[0])
            personalized_result = agent.generate_characteristic_dialogue(personalized_result_tmp, style[1])
        else:
            personalized_result = agent.generate_characteristic_dialogue(examinated_result, agent.characteristic)




        return personalized_result

    else:
        n = 0
        result = ''
        while n<=setting['max_output_retries']:
            n+=1
            results = agent.generate_dialogue_response(new_message,inquirer,victim)
            if "抱歉" in results:
                continue
            if results == None:
                print()
            else:
                result = results[1]
            if result!='':
                break
        if setting['Self_Examination']:
            examinated_result = agent._self_examination(new_message, result)
        else:
            examinated_result = result
        
        if setting["Two_Step_Characterized"]:
            style = agent.characteristic.split('并且')
            personalized_result_tmp = agent.generate_characteristic_dialogue(examinated_result, style[0])
            personalized_result = agent.generate_characteristic_dialogue(personalized_result_tmp, style[1])
        else:
            personalized_result = agent.generate_characteristic_dialogue(examinated_result, agent.characteristic)

        return personalized_result

def self_introduction(agents: List[GenerativeAgent], victim: String) -> None:
    """Runs a conversation between agents."""
    # time.sleep(10)
    random_agents = random.sample(agents, len(agents))
    # self-introduction
    for agent in random_agents:
        if setting['Cooperation_script']:
            question = "请你先介绍一下你的角色，然后说一下你的个人背景，即你为什么会来到这个地方。最后再用一段话详细介绍一下你在案发日的时间线。要具体到你在案发之日见过什么人和做过什么事（如果有具体时间点，也需要提供）。"
        else:
            question = "请你先介绍一下你的角色，然后说一下你所认识的案件的受害人：%s是一个怎么样的人，以及你和他的关系。最后再用一段话详细介绍一下你在案发日的时间线。要具体到你在案发之日几点几分见过什么人和做过什么事。"%victim

        reply = interview_agent(USER_NAME,agent, question,victim)
        print('\n')
        print(f"\n（问题）{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n（回答）{agent.name}对{USER_NAME}说：\"{reply}\"")
        global_output_list.append(f"\n（问题）{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n（回答）{agent.name}对{USER_NAME}说：\"{reply}\"")
        agent.add_memory(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")
        # agent.self_history.append(reply)
        for other in random_agents:
            if other == agent:
                continue
            other.add_memory(f"{USER_NAME}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{USER_NAME}说：\"{reply}\"")


        for other in random_agents:
            if other == agent:
                continue
            question = other.generate_dialogue_question(observation=f"{agent.name}对{USER_NAME}说：\"{reply}\"",respondent=agent)
            reply = interview_agent(other.name,agent, question,victim)
            print('\n')
            print(f"\n（问题）{other.name}对{agent.name}说：\"{question}\"\n\n\n（回答）{agent.name}对{other.name}说：\"{reply}\"")

            global_output_list.append(f"\n（问题）{other.name}对{agent.name}说：\"{question}\"\n\n\n（回答）{agent.name}对{other.name}说：\"{reply}\"")
            # agent.self_history.append(reply)

            other.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")
            agent.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")

            for other_other in random_agents:
                if agent == other_other or other == other_other:
                    continue

                other_other.add_memory(f"{other.name}对{agent.name}说：\"{question}\"\n\n\n{agent.name}对{other.name}说：\"{reply}\"")

def free_discussion(agents: List[GenerativeAgent], clues_given: bool) -> None:
    """Runs a conversation between agents."""
    random_agents = random.sample(agents, len(agents))

    for agent in random_agents:
        n = 0
        other_players = [p for p in agent.players_in_game if p !=agent.name ]
        player_to_ask = random.choice(other_players)
        question_to_ask = ''
        while n<=setting['max_output_retries']:
            n+=1
            player_to_ask, question_to_ask =  agent._generate_fd_questions(clues_given=clues_given)

            if player_to_ask in other_players and question_to_ask!='':
                break

        player_to_ask_agent = [r_a for r_a in random_agents if r_a.name == player_to_ask][0]

        reply2 = interview_agent(agent.name,player_to_ask_agent, question_to_ask,victim)
        print('\n')
        print(f"\n（问题）{agent.name}对{player_to_ask_agent.name}说：\"{question_to_ask}\"\n\n\n（回答）{player_to_ask_agent.name}对{agent.name}说：\"{reply2}\"")
        global_output_list.append(f"\n（问题）{agent.name}对{player_to_ask_agent.name}说：\"{question_to_ask}\"\n\n\n（回答）{player_to_ask_agent.name}对{agent.name}说：\"{reply2}\"")
        # agent.self_history.append(reply2)
        for other in random_agents:

            other.add_memory(f"{agent.name}对{player_to_ask_agent.name}说：\"{question_to_ask}\"\n\n\n{player_to_ask_agent.name}对{agent.name}说：\"{reply2}\"")

def get_next_run_number(folder_path,prefix,suffix):
    run_files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith(suffix)]
    run_numbers = [int(f[len(prefix):-len(suffix)]) for f in run_files]
    if not run_numbers:
        return 1
    return max(run_numbers) + 1	

def handling_output_parsing(output_parser,output):
    if 'json' not in output.content:
        try:
            character_info = output_parser.parse(output.content.replace('```\n','```json\n'))
            return character_info
        except:
            try:
                character_info = output_parser.parse(output.content.replace('```\n','```json\n').replace(',\n}','\n}'))
                return character_info
            except:
                print("输出格式有误，重新生成")
                return False

    else:
        try:
            character_info = output_parser.parse(output.content)
            return character_info
        except:
            try:
                character_info = output_parser.parse(output.content.replace(',\n}','\n}'))
                return character_info
            except:
                try:
                    character_info = output_parser.parse(output.content+'```')
                    return character_info
                except:
                    try:
                        character_info = output_parser.parse(output.content.replace('"\n\t"','"\n\t,"'))
                        return character_info
                    except:
                        try:
                            character_info =  output_parser.parse(output.content.replace('"\n    "','"\n\t,"'))
                            return character_info
                        except:
                            try:
                                character_info = output_parser.parse(output.content.replace('"\n\n\t"','"\n\t,"'))
                                return character_info
                            except:
                                try:
                                    character_info = output_parser.parse(output.content.replace('\n}','"\n}'))
                                    return character_info
                                except:

                                    try:
                                        character_info = output_parser.parse(output.content.replace('，\n\t',',\n\t'))
                                        return character_info
                                    except:
                                        try:
                                            character_info = output_parser.parse(output.content.split(' // ')[0]+'\n}\n```')
                                            return character_info
                                        except:
                                            try:
                                                character_info = output_parser.parse(output.content.replace('False','"False"').replace('True','"True"'))
                                                return character_info
                                            except:

                                                print("输出格式有误，重新生成")
                                                return False


def record_experiment_results(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    run_number = get_next_run_number(folder_path,prefix='run',suffix='.txt')
    file_name = f"run{run_number}.txt"
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'w') as f:
        for result in global_output_list:
            f.write(result+'\n')

def record_agent(folder_path=None,agents= None,phase=None):
    assert folder_path!=None

    assert agents!=None

    numofchat =  len(agents[0].chat_history)
    for agent in agents:
        assert numofchat == len(agent.chat_history)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # assert phase in ['pre_r1','post_r1','post_r2']

    run_number = get_next_run_number(folder_path,prefix=phase,suffix='.pkl')
    file_name = f"{phase}_{run_number}.pkl"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(agents, f)


def remove_quotes_and_colons(s: str) -> str:

    quotes_and_colons = ['"', '“', '”', ':', '：','#']
    
    while len(s) > 0 and s[0] in quotes_and_colons:
        s = s[1:]
    
    while len(s) > 0 and s[-1] in quotes_and_colons:
        s = s[:-1]
    
    return s

def str_to_bool(s: str) -> bool:
    if isinstance(s, bool):
        return s
    return s.lower() == 'true'

def remove_numeric_prefix(s: str) -> str:

    return re.sub(r'^\d+\.\s*', '', s)

def decimal_to_percentage(decimal_value: float) -> str:
    if 0 <= decimal_value <= 1:
        return "{:.0f}%".format(decimal_value * 100)
    else:
        raise ValueError("The provided value is not between 0 and 1.")

def list_to_string(names):
    result = ""
    for i, name in enumerate(names, 1):
        result += f"{i}. {name} "
    return result.strip()

def create_next_exp_folder(path):

    if not os.path.exists(path):
        os.makedirs(path)
    
    dir_entries = os.listdir(path)
    max_exp_num = 0
    
    for entry in dir_entries:
        if entry.startswith("exp") and entry[3:].isdigit():
            num = int(entry[3:])
            if num > max_exp_num:
                max_exp_num = num
    
    new_folder_name = "exp" + str(max_exp_num+1)
    new_folder_path = os.path.join(path, new_folder_name)
    os.makedirs(new_folder_path)
    
    return new_folder_path

if __name__ == "__main__":

    play = setting['Play_no.']
    with open('%s.json'%(play), encoding='utf-8') as f:
        data = json.load(f)
    victim = data['受害人']
    agents = []

    for i in range(len(data['角色'])):


        agents.append(GenerativeAgent(name=data['角色'][i]['角色名'],
                      age=data['角色'][i]['年龄'],
                      role=data['角色'][i]['角色'],
                      mission = data['角色'][i]['角色任务'].replace('你',data['角色'][i]['角色名']).replace('我',data['角色'][i]['角色名']),
                      character_story = '【%s的角色剧本】%s'%(data['角色'][i]['角色名'],data['角色'][i]['人物剧本']),
                      story_background = '【剧本背景】'+data['剧本背景'],
                      character_timeline = '【%s的案发日详细时间线】'%(data['角色'][i]['角色名'])+data['角色'][i]['案发日时间线'],
                      game_rule = '【游戏规则】：%s'%data['游戏规则'],
                      memory_retriever=create_new_memory_retriever() if setting['Retrieval'] else None,
                      players_in_game = [c['角色名']for c in data['角色']],
                      characteristic = data['角色'][i]['说话风格'],
                      llm=LLM,
                      daily_summaries = [
                       ]
                     )
        )



    path_to_save_agents = create_next_exp_folder("%s/saved_agents_inpc/%s/%s"%(setting['Project_path'],play,selected_gpt_model))
    print(path_to_save_agents)
    with open('%s/setting.json'%path_to_save_agents, 'w', encoding='utf-8') as json_file:
        json.dump(setting, json_file, ensure_ascii=False, indent=4)

    record_agent(folder_path=path_to_save_agents,agents= agents,phase='r0')



    print('****************游戏开始*******************\n')

    print('****************自我介绍*******************\n')

    self_introduction(agents,victim)
    record_agent(folder_path=path_to_save_agents,agents= agents,phase='s1')

    print('\n****************第一轮自由问答*******************\n')
    for i in range(0,setting['num of rounds for first free discussion']):
        free_discussion(agents,clues_given=False)
        record_agent(folder_path=path_to_save_agents,agents= agents,phase='f%s'%(i+1))
    
    agent2clues = {}
    for agent in agents:
    
        clues =  ['线索%s：“'%letter_list[idx] + c['clue_type'] + "：" + c['content'] + '”' for idx, c in enumerate(data['线索'][::-1])]
        agent2clues[agent.name] = clues
    
    
    for agent in agents:
    
        for line in agent2clues[agent.name]:
            agent.add_memory(line,isclue=True)
            agent.clue_list.append(line)
    
    print('\n****************第二轮自由问答*******************\n')
    for i in range(setting['num of rounds for first free discussion'],setting['num of rounds for first free discussion'] + setting['num of rounds for second free discussion']):
        free_discussion(agents,clues_given=True)
        record_agent(folder_path=path_to_save_agents,agents= agents,phase='f%s'%(i+1))


    # print('\n****************投票环节*******************\n')

    # agent_score = defaultdict(int)
    # vote_themes = data['投票']
    # for vote in vote_themes:
    #     vote_content = vote['投票主题']
    #     vote_voter = vote['提问对象'].split('；')
    #     vote_choice = vote['投票选项']
    #     vote_len = vote['输出选项个数']
    #     vote_add = vote['加分对象']
    #     vote_sub = vote['减分对象']
    #     vote_answer = vote['答案']
    #     vote_question = ("现在讨论环节结束，进入最终的投票环节。"
    #                 "请利用你的推理能力，结合相关信息，回答选择题：\n"
    #                 "%s\n"
    #                 "选项：%s\n"
    #                 "请注意：回答只需要输出相应答案对应字母，不需要输出推理过程和汉字答案。以下是一个回答的例子：A")%(vote_content, vote_choice)


    #     for agent in agents:
    #         if agent.name in vote_voter:
    #             n = 0
    #             vote_results = ''
    #             while n <= setting['max_output_retries']:
    #                 n += 1
    #                 vote_results = agent.generate_voting_results(vote_question, vote_len)
    #                 if len(vote_results) == vote_len:
    #                     break

    #             if vote_results == vote_answer:
    #                 agent_score[vote_add] += 1
    #                 if vote_sub != "":
    #                     agent_score[vote_sub] -= 1
    #             elif vote_results.isalpha() and len(vote_results)==vote_len:
    #                 agent_score[vote_add] -= 1
    #                 if vote_sub != "":
    #                     agent_score[vote_sub] += 1

    #             print(f"{USER_NAME}对{agent.name}说：\"{vote_question}\"")
    #             print(f"{agent.name}对{USER_NAME}说：\"{vote_results}\"")
    #             global_output_list.append(f"{USER_NAME}对{agent.name}说：\"{vote_question}\"")
    #             global_output_list.append(f"{agent.name}对{USER_NAME}说：\"{vote_results}\"")

    # for agent_name, score in agent_score.items():
    #     print(f"{agent_name}的得分是：{score}分\n")
    #     global_output_list.append(f"{agent_name}的得分是：{score}分\n")

    record_experiment_results(folder_path=path_to_save_agents)






