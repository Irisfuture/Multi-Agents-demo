from langchain_core.prompts import ChatPromptTemplate 
from langchain_openai import ChatOpenAI  
_llm=ChatOpenAI(
        api_key="ollama",
        model="qwen2.5:7b",
        base_url="http://0.0.0.0:60016/v1",
        temperature=0.6
    )
from planner import Planner
from evaluer import Executor
from planner_grand2 import Planner2
from writer import Writer

_planner = Planner(_llm)
_evaluer = Executor(_llm)
_planner_grand2 = Planner2(_llm)
_writer= Writer(_llm)

from typing import TypedDict
from typing import Annotated
from typing import List
from operator import add

class PlanState(TypedDict):
    query:str
    Topic:str
    outline_list:List[str]     #规约是覆盖，不用写
    advise:Annotated[List[str],add]
    Topic_Highlights:str
    content:str
def _planner_node(state):
    _rt = _planner(state)
    return _rt      # 返回的state 中就有了list
def _evaluer_node(state):
    _advise=state.get("advise",[]) # 初始第一个是空
    _outline_list= state["outline_list"]
    _Topic_Highlights=state["Topic_Highlights"]
    _Topic=state["Topic"]
    _query=state["query"]
    evaluation_result=_evaluer({
        "query":_query,
        "outline_list":_outline_list,
        "Topic":_Topic,
        "Topic_Highlights":_Topic_Highlights
            })
    return{"advise":[evaluation_result]}
def _planner_grand2_node(state):
    _advise=state["advise"]
    _outline_list= state["outline_list"]
    _Topic_Highlights=state["Topic_Highlights"]
    _Topic=state["Topic"]
    planner_result= _planner_grand2({
        "outline_list":_outline_list,
        "Topic":_Topic,
        "Topic_Highlights":_Topic_Highlights,
        "advise":_advise
    })
    return planner_result  
def _writer_node(state):
    _query=state["query"]
    _outline_list= state["outline_list"]
    _Topic_Highlights=state["Topic_Highlights"]
    _Topic=state["Topic"]
    _rt = _writer({
        "query":_query,
        "outline_list":_outline_list,
        "Topic":_Topic,
        "Topic_Highlights":_Topic_Highlights
    })
    return {"content":_rt}

from langgraph.graph import StateGraph,MessagesState,START,END

_builder=StateGraph(PlanState)

_builder.add_node("_planner_node", _planner_node)
_builder.add_node("_evaluer_node", _evaluer_node)
_builder.add_node("_planner_grand2_node", _planner_grand2_node)
_builder.add_node("_writer_node", _writer_node)


_builder.add_edge(START,'_planner_node')
_builder.add_edge('_planner_node',"_evaluer_node")
_builder.add_edge('_evaluer_node',"_planner_grand2_node")
_builder.add_edge("_planner_grand2_node",'_writer_node')
_builder.add_edge('_writer_node',END)

_graph =_builder.compile()

if __name__ == "__main__":
    input1=input("请输入写作需求:")
    _rt=_graph.invoke({'query':input1})
    print(_rt)