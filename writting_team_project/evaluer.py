from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from typing import Annotated
from utils import websearch
from langgraph.prebuilt import create_react_agent #这里创建出来是个agent传递状态
                                #链返回是messeges列表包含过程消息
from pydantic import BaseModel,Field
from typing import List

# 输入：planner输出的 List[str]
# 输出：str advise_list的第一个
# 调用工具 ：utils.py 的 websearch 

@tool    #互联网工具属于不稳定，提示语改改看
def web_search(query:Annotated[str,"互联网查询标题"]):
    """
    通过web_search工具查询互联网上的信息
    """
    return websearch(query)


_executor_syetem_templete="""
我是一位教育机构的课题评价者，面临的问题是学生提交的课题可能存在模糊不清的情况，需要通过评价来确保课题具有课题价值且贴近当下热点。
角色：希望你扮演一个严谨且有耐心的教育专家，需要与学生进行对话引导，帮助他们完善课题任务。
要求：
1. 与学生进行对话，了解其课题的主题和信息，引导学生明确课题。
2. 提供学生参考资料，帮助其丰富课题内容，使之更具深度和广度。
3. 要求学生按照指导完善课题大纲，确保其内容清晰、结构完整。
4. 因为留言字数有限制，需要文字上紧凑些，去掉格式，去掉多余的谦辞，这点务必留意。
您可以使用以下工具来协助您更好的完成该任务：
{tools_name}
"""

_executor_human_template = """
课题要求：
{query}
学生提交的课题：
{Topic}
学生提交的大纲与写作思路：
{outline_list}
学生自己总结的课题亮点
{Topic_Highlights}
"""

class Executor:

    def __init__(self, llm):

        _tools=[web_search,]
        _prompt = ChatPromptTemplate.from_messages([
            ("system", _executor_syetem_templete),
            ("human", _executor_human_template)
        ])
        _prompt=_prompt.partial(tools_name=",".join([_tool.name for _tool in _tools]))
        # 通过 partial 方法将工具的名称加入到提示prompt中。# 把tool 名称变成列表连起来，
        
        _llm_with_tools_agent = create_react_agent(llm, tools=_tools)

        self._chain = _prompt | _llm_with_tools_agent
        self._parser = StrOutputParser()

    def __call__(self, state):
        _rt = self._chain.invoke(state)
        _messages = _rt["messages"]
        return self._parser.invoke(_messages[-1])

        # 使用 self._chain 来执行 state，然后从执行结果中提取消息 
        # 把最后一条消息取出来做解析

    
if __name__ == "__main__":

    from langchain_openai import ChatOpenAI

    _llm=ChatOpenAI(
        api_key="ollama",
        model="qwen2.5:7b",
        base_url="http://0.0.0.0:60016/v1",
        temperature=0.6
    )
    
    _executor=Executor(_llm)
    _rt = _executor({
        "advise": [],
        "query":"写一篇关于幼儿教育的课题文章，字数5000左右，要结合今年的教育热题",
        "Topic": "基于当前教育热点的幼儿教育策略探究",
        "outline_list":['引言：介绍幼儿教育的重要性及本文的研究目的（150-200字）', '背景分析：探讨当前社会对幼儿教育的关注点及其原因（300-400字）', '理论基础：阐述幼儿心理发展特点和学习方式，结合相关心理学理论进行论述（400-500字）', '现状分析：分析当前幼教中存在的问题及解决策略（400-600字）', '热点案例研究：选取当前教育领域的热点事件或人物进行深入剖析，如家长过度干预、幼儿教育商业化等现象（400-600字）', '对策建议：提出符合新时代需求的幼儿教育改进措施（300-500字）', '结论：总结全文观点，强调研究意义及对实践的指导价值（150-200字）'],
        "Topic_Highlights": '本文将结合当前社会和教育领域的热点问题，探讨幼儿教育的发展趋势与策略。通过对理论基础、现状分析以及热点案例的研究，旨在为广大家长和幼教工作者提供切实可行的建议，推动幼儿教育事业健康发展。'       
    })

    print(_rt)

