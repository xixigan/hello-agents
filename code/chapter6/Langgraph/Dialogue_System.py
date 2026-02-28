"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户需求
2. 使用Tavily API真实搜索信息  
3. 生成基于搜索结果的回答
"""

import asyncio
import os
import time
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from tavily import TavilyClient

# 加载.env文件
load_dotenv()

# 定义状态结构
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str        # 用户查询
    search_query: str      # 优化后的搜索查询
    search_results: str    # Tavily搜索结果
    final_answer: str      # 最终答案
    step: str             # 当前步骤

# 初始化模型
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    temperature=0.7
)

def retry_llm_invoke(prompt, max_retries=3, delay=1):
    """带重试机制的LLM调用函数，处理限流等API错误"""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            error_msg = str(e)
            print(f"❌ LLM调用失败 (尝试 {attempt+1}/{max_retries}): {error_msg}")
            
            # 检查是否为429限流错误
            if "429" in error_msg or "rate limit" in error_msg.lower() or "Rate limit" in error_msg:
                wait_time = delay * (2 ** attempt)  # 指数退避
                print(f"⏳ 限流错误，将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                # 其他错误不再重试
                raise
    
    # 所有重试都失败
    raise Exception(f"LLM调用在 {max_retries} 次尝试后仍失败")

def understand_query_node(state: SearchState) -> SearchState:
    """步骤1：理解用户查询并生成搜索关键词"""
    
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    try:
        response = retry_llm_invoke([SystemMessage(content=understand_prompt)])
        
        # 提取搜索关键词
        response_text = response.content
        search_query = user_message  # 默认使用原始查询
        
        if "搜索词：" in response_text:
            search_query = response_text.split("搜索词：")[1].strip()
        elif "搜索关键词：" in response_text:
            search_query = response_text.split("搜索关键词：")[1].strip()
        
        return {
            "user_query": response.content,
            "search_query": search_query,
            "step": "understood",
            "messages": [AIMessage(content=f"我理解您的需求：{response.content}")]
        }
    except Exception as e:
        error_msg = f"理解用户查询时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        
        # 如果LLM调用失败，使用原始查询作为搜索词
        return {
            "user_query": user_message,
            "search_query": user_message,
            "step": "understood",
            "messages": [AIMessage(content=f"我将直接搜索您的问题")]
        }

def tavily_search_node(state: SearchState) -> SearchState:
    """步骤2：使用Tavily API进行真实搜索"""
    
    search_query = state["search_query"]
    
    try:
        print(f"🔍 正在搜索: {search_query}")
        
        # 调用Tavily搜索API
        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            max_results=5
        )
        
        # 处理搜索结果
        search_results = ""
        
        # 优先使用Tavily的综合答案
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"
        
        # 添加具体的搜索结果
        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"
        
        if not search_results:
            search_results = "抱歉，没有找到相关信息。"
        
        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [AIMessage(content=f"✅ 搜索完成！找到了相关信息，正在为您整理答案...")]
        }
        
    except Exception as e:
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "search_results": f"搜索失败：{error_msg}",
            "step": "search_failed",
            "messages": [AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")]
        }

def generate_answer_node(state: SearchState) -> SearchState:
    """步骤3：基于搜索结果生成最终答案"""
    
    try:
        # 检查是否有搜索结果
        if state["step"] == "search_failed":
            # 如果搜索失败，基于LLM知识回答
            fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""
            
            response = retry_llm_invoke([SystemMessage(content=fallback_prompt)])
            
            return {
                "final_answer": response.content,
                "step": "completed",
                "messages": [AIMessage(content=response.content)]
            }
        
        # 基于搜索结果生成答案
        answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""

        response = retry_llm_invoke([SystemMessage(content=answer_prompt)])
        
        return {
            "final_answer": response.content,
            "step": "completed",
            "messages": [AIMessage(content=response.content)]
        }
    except Exception as e:
        error_msg = f"生成答案时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        
        # 返回友好的错误提示
        error_response = f"抱歉，我在生成答案时遇到了一些问题：\n{str(e)}\n\n建议您稍后重试，或者尝试使用更具体的问题描述。"
        
        return {
            "final_answer": error_response,
            "step": "completed",
            "messages": [AIMessage(content=error_response)]
        }

# 构建搜索工作流
def create_search_assistant(tavily_client_instance):
    # 将tavily_client设为全局变量，以便搜索节点使用
    global tavily_client
    tavily_client = tavily_client_instance
    
    workflow = StateGraph(SearchState)
    
    # 添加三个节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    
    # 设置线性流程
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    
    # 编译图
    memory = InMemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

async def main():
    """主函数：运行智能搜索助手"""
    
    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return
    
    # 初始化Tavily客户端
    global tavily_client
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    app = create_search_assistant(tavily_client)
    
    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")
    
    session_count = 0
    
    while True:
        user_input = input("🤔 您想了解什么: ").strip()
        
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        
        if not user_input:
            continue
        
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        # 初始状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start"
        }
        
        try:
            print("\n" + "="*60)
            
            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
            error_msg = str(e)
            print(f"❌ 发生错误: {error_msg}")
            
            # 根据错误类型提供更具体的建议
            if "429" in error_msg or "rate limit" in error_msg.lower():
                print("💡 提示：当前服务存在限流，建议您稍后重试或使用更具体的问题描述。\n")
            elif "API_KEY" in error_msg or "api_key" in error_msg.lower():
                print("💡 提示：请检查您的API密钥配置是否正确。\n")
            else:
                print("请重新输入您的问题。\n")

if __name__ == "__main__":
    asyncio.run(main())