# 代理架构

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel,Field
from typing import List

# plan agent 的输出：List[str]
class Plan(BaseModel):
    '''
    制定的计划
    '''
    Topic:str =Field(description="课题名称和描述")
    outline_list:List[str]=Field(description="论文的详细大纲")
    Topic_Highlights:str=Field(description="其他思考")

    # 注释给大模型看的 #Field 是pydantic库中的一个装饰器，用来为这个字段添加额外的元数据。

_planner_system_templete=''' 

角色：你是一名优秀的研究生，需要按照老师的建议修改前面版本的论文大纲，要补充更多字数呈现写作思路，形成课题计划，绝对不能偷懒，字数计划上每章节要更多一些。

要求：
1、仔细阅读前面版本的论文大纲，了解需要补充的内容和思路。
2、根据每个章节的主题，补充相关的写作思路和内容，确保每章节字数更多。
3、修改后的论文大纲，确保总字数超过要求字数。
4、确保整体课题计划的完整性和逻辑性，包括综述、方法等部分


输出：
{output_format}
'''

_planner_human_templete='''
提交的课题：
{Topic}
课堂重点：
{Topic_Highlights}
提交的大纲与写作思路：
{outline_list}
老师的建议：
{advise}

'''
class Planner2:
    def __init__(self,llm):
        _prompt = ChatPromptTemplate.from_messages([
            ("system", _planner_system_templete),
            ("human",_planner_human_templete)
        ])
        _parser=JsonOutputParser(pydantic_object=Plan)
        _prompt=_prompt.partial(output_format=_parser.get_format_instructions())
        self._chain=_prompt|llm|_parser
    def __call__(self, state):
        return self._chain.invoke(state)
    
if __name__ == "__main__":

    from langchain_openai import ChatOpenAI

    _llm=ChatOpenAI(
        api_key="ollama",
        model="qwen2.5:7b",
        base_url="http://0.0.0.0:60016/v1",
        temperature=0.6
    )
    
    #创建planner角色
    _planner=Planner2(_llm)


    _rt=_planner({"query":"写一篇关于幼儿教育的课题文章，字数2000左右，要结合今年的教育热题",
        "Topic": "基于当前教育热点的幼儿教育策略探究",
        "outline_list":['引言：介绍幼儿教育的重要性及本文的研究目的（150-200字）', '背景分析：探讨当前社会对幼儿教育的关注点及其原因（300-400字）', '理论基础：阐述幼儿心理发展特点和学习方式，结合相关心理学理论进行论述（400-500字）', '现状分析：分析当前幼教中存在的问题及解决策略（400-600字）', '热点案例研究：选取当前教育领域的热点事件或人物进行深入剖析，如家长过度干预、幼儿教育商业化等现象（400-600字）', '对策建议：提出符合新时代需求的幼儿教育改进措施（300-500字）', '结论：总结全文观点，强调研究意义及对实践的指导价值（150-200字）'],
        'Topic_Highlights': '本文将结合当前社会和教育领域的热点问题，探讨幼儿教育的发展趋势与策略。通过对理论基础、现状分析以及热点案例的研究，旨在为广大家长和幼教工作者提供切实可行的建议，推动幼儿教育事业健康发展。',       
        'advise':'''
了解这个课题的主题和大纲后，我建议你进一步明确课题的具体方向。当前热点可能包括但不限于幼儿教育中的过度商业化、家庭教育与学校教育的关系、幼儿心理健康的重要性等。以下是我为你准备的一些参考信息：
1. 教育部发布《3-6岁儿童学习与发展指南》，强调以游戏为基本活动。
2. 世界卫生组织关于幼儿早期发展的重要性的报告。
3. 美国心理学家加德纳的多元智能理论及其对幼儿教育的影响。
4. 当前社会中家长过度干预的现象及影响分析。
请结合这些信息，进一步完善你的课题大纲。具体建议如下：
1. 引言部分可以更多地介绍当前社会对于幼儿教育的需求和期望。
2. 背景分析中添加关于家庭教育与学校教育相互作用的观点。
3. 理论基础部分可以加入多元智能理论、游戏化学习等现代教育理念的阐述。
4. 现状分析中讨论家长过度干预的具体表现及其对孩子成长的影响。
5. 热点案例研究可聚焦于当前社会普遍关注的热点事件或人物，如“双减”政策对幼儿教育的影响。
6. 对策建议部分可以结合以上提到的各种理论和现实情况提出具体可行的改进措施。'''})
    print(_rt)