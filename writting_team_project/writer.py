from langchain_core.prompts import ChatPromptTemplate
# 定义输出的结构
from pydantic import BaseModel,Field
from langchain_core.output_parsers import JsonOutputParser

# class Outline(BaseModel):
#     title:str=(description="一级目录")

# class Scheme(BaseModel):
#     abstract:str =Field(description="全文的摘要")
#     outline:list[Outline]= Field(description="项目大纲")

class Scheme(BaseModel):
    content:str = Field(description="生成的内容")

_writer_system_template = r"""
角色：您是一个优秀且有耐心且勤劳的文案写手。
您的职责在于，基于课题和大纲计划，精心撰写每个章节的内容。
您需要确保文字详实，与课题和写作要求保持一致，同时确保内容的连贯性和逻辑性。
为了达到客户的满意，字数上不要偷懒！

要求：
1.每个章节的字数要超过计划的字数。
2.写作的文章中一定不要体现出字数计划，
3.要求保持内容的连贯性和逻辑性。

输出：
{output_format}
"""

_writer_human_template = r"""
写作要求：{query}
课题标题：{Topic}
课题大纲和计划：{outline_list}
课题亮点：{Topic_Highlights}
"""

class Writer:
    def __init__(self, llm):
        self._llm = llm
        _prompt = ChatPromptTemplate.from_messages([
            ("system", _writer_system_template),
            ("human", _writer_human_template)
        ])
        _parser = JsonOutputParser(pydantic_object=Scheme)
        _prompt=_prompt.partial(output_format = _parser.get_format_instructions())
        self._chain = _prompt | self._llm | _parser

    def __call__(self, state):
        return self._chain.invoke(state)
    

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        api_key="ollama",
        model="qwen2.5:7b",
        base_url="http://0.0.0.0:60016/v1",
        temperature=0.6
    )

    write = Writer(_llm)
    rt = write({"query":"写一篇关于幼儿教育的课题文章，字数2000左右，要结合今年的教育热题",
                'Topic': '基于当前教育热点的幼儿教育策略探究', 
                'outline_list': ['引言：介绍幼儿教育的重要性及本文的研究目的（250字）', '背景分析：探讨当前社会对幼儿教育的关注点及其原因（400字）', '理论基础：阐述幼儿心理发展特点和学习方式，结合相关心理学理论进行论述（500字）', '现状分析：分析当前幼教中存在的问题及解决策略（600字）', '热点案例研究：选取当前教育领域的热点事件或人物进行深入剖析，如家长过度干预、幼儿教育商业化等现象（600字）', '对策建议：提出符合新时代需求的幼儿教育改进措施（450字）', '结论：总结全文观点，强调研究意义及对实践的指导价值（250字）'], 
                'Topic_Highlights': '本课题旨在结合当前社会和教育领域的热点问题，深入探讨幼儿教育的发展趋势与策略。通过对理论基础、现状分析以及热点案例的研究，为广大家长和幼教工作者提供切实可行的建议，推动幼儿教育事业健康发展。'})
    print(rt["content"])
