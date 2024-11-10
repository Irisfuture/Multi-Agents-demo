from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
# 定义输出的结构
from pydantic import BaseModel,Field
from langchain_core.output_parsers import StrOutputParser  #规范点就用JsonOutputParser

class Outline(BaseModel):   # 定义输出的结构，替代
    title:str=Field(description="一级目录")
    sub_titles:list[str] = Field(description="二级目录列表")

class Scheme(BaseModel):  #定义数据模型，定义了 JSON 输出的结构和类型
    abstract:str =Field(description="全文的摘要")
    outline:list[Outline]= Field(description="项目大纲")

from langchain_core.tools import tool
from typing import Annotated
from utils import websearch
from langgraph.prebuilt import create_react_agent #这里创建出来是个agent传递状态 #有带工具的要加这个,就不用chain

@tool    #互联网工具属于不稳定，提示语改改看
def web_search(query:Annotated[str,"互联网查询标题"]):
    """
    通过web_search工具查询互联网上的信息
    """
    return websearch(query)

_researcher_system_template= """
你是一个乐于助人的人工智能助手，与其他助手合作。
使用提供的工具来回答问题。
如果你不能完全回答，另一个助手用不同的工具。
这将有助于你取得进展。尽你所能取得进展。
您可以访问以下工具：
{tools_name}
"""

class Researcher:
    def __init__(self, llm):
        _tools=[]
        _prompt = ChatPromptTemplate.from_messages([
            ("system", _researcher_system_template),
            MessagesPlaceholder(variable_name="messages")  #
        ])

        _prompt=_prompt.partial(tools_name=",".join([_tool.name for _tool in _tools]))
        self._llm = llm
        _llm_with_tool_agent=create_react_agent(self._llm,tools=_tools)
        self._chain = _prompt|_llm_with_tool_agent

    def __call__(self, state):
        _rt=self._chain.invoke(state)
        _messages=_rt["messages"]
        return _messages[-1]

if __name__ == '__main__':
    from langchain_openai import ChatOpenAI
    _llm = ChatOpenAI(
        base_url="http://0.0.0.0:60016/v1",
        model="qwen2.5:7b",
        api_key="ollama"
    )
    _researcher = Researcher(_llm)
    _rt = _researcher({"messages":[("human","获取英国过去5年的国内生产总值。一旦你把它编码好，并执行画图，就完成。")]})
    print(_rt)
