from zhipuai import ZhipuAI

class WebSearch:

    def __init__(self):
        self._client = ZhipuAI(api_key="51109e5e1fd67c96a7a51eb74e5ae8ca.SCpBvbMmVT6i3axx")

        self._tools = [{
            "type": "web_search",
            "web_search": {
                "enable": True #默认为关闭状态（False） 禁用：False，启用：True。
            }
        }]
    
    def __call__(self,query):
        messages = [{
            "role": "user",
            "content": query
        }]

        response = self._client.chat.completions.create(
            model="glm-4-plus",
            messages=messages,
            tools=self._tools
        )
        return response.choices[0].message.content

websearch = WebSearch()


from langchain_experimental.utilities import PythonREPL
repl = PythonREPL()

# PythonREPL 类来创建一个 Python 交互式 REPL（Read-Eval-Print Loop）环境
# REPL 环境允许用户输入命令，然后立即执行并显示结果，这非常适合于快速测试和交互式探索。