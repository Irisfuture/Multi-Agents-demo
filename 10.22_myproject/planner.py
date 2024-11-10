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
    outline_list:List[str]=Field(description="论文大纲")
    Topic_Highlights:str=Field(description="其他思考")
    # 注释给大模型看的 #Field 是pydantic库中的一个装饰器，用来为这个字段添加额外的元数据。

_planner_system_templete=''' 
背景：我是一名语文老师，负责指导学生进行论文课题写作。为了确保学生能够顺利完成论文，我要求学生
1、按照输入中的写作要求选定合适的子课题，并提供一段简要描述，阐明选择该课题的理由及其研究价值。
2、然后撰写一个完整的大纲，以便有条不紊地展开写作过程。

角色：请你扮演一位优秀的学生提供示范。

任务：要求根据所选定的子课题，撰写一个完整的大纲(包含字数范围），为后续的写作过程提供清晰的方向和框架。

要求：
1. 需提供选定的子课题名称和简要描述。
2. 在大纲中明确论文的主题和目的，以及各个部分的内容、字数范围和顺序，为后续的写作提供指导。
3. 大纲应包含引言、论点和论据、结论等基本要素，以确保论文的逻辑和结构的连贯性。
4. 可以参考相关的学术资料和范例，以提高大纲的质量和准确性。  

输出：
{output_format}
'''
_planner_human_templete='''
查询目标：
{query}
'''
class Planner:
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
        base_url="http://192.168.10.13:60001/v1",
        temperature=0.6
    )
    
    #创建planner角色
    _planner=Planner(_llm)
    _rt=_planner({"query":"写一篇关于幼儿教育的课题文章，字数8000左右，要结合今年的教育热题"})

    #_rt=_planner({"messages":[("human",{"query":"写一篇关于幼儿教育的课题文章，字数5000左右，要结合今年的教育热题"})]})
    print(_rt)