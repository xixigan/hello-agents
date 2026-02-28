from dotenv import load_dotenv
# 加载 .env 文件中的环境变量
load_dotenv()

import os
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any

def search(query: str) -> str:
    """
    一个基于百度搜索的网页搜索工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [百度] 网页搜索: {query}")
    try:
        # 使用百度桌面版搜索，参数更简单
        url = "https://www.baidu.com/s"
        params = {
            "wd": query,  # 桌面版使用wd参数
            "ie": "utf-8",
            "rn": "10"  # 返回10条结果
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Host": "www.baidu.com",
            "Referer": "https://www.baidu.com/",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        response = requests.get(url, params=params, headers=headers)
        # 检查响应头中的编码信息
        content_type = response.headers.get('Content-Type', '')
        if 'charset=' in content_type:
            encoding = content_type.split('charset=')[-1].strip()
            response.encoding = encoding
        else:
            # 尝试常见编码
            try:
                response.encoding = 'utf-8'
                # 测试编码是否正确
                _ = response.text
            except UnicodeDecodeError:
                response.encoding = 'gbk'
        
        # 调试：保存返回的HTML内容到文件
        with open("baidu_response.html", "w", encoding="utf-8") as f:
            f.write(response.text)
            print("已保存百度搜索响应到 baidu_response.html 文件")
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 智能解析：优先寻找最直接的答案
        # 1. 寻找百度的直接答案区域（如百度知道、百度百科摘要等）
        answer_area = None
        
        # 检查百度百科摘要
        baike_summary = soup.find("div", class_="c-border c-border-gray2 c-bg-gray")
        if baike_summary:
            title = baike_summary.find("h3", class_="t c-gap-bottom-small")
            content = baike_summary.find("div", class_="c-abstract")
            if title and content:
                return f"{title.text.strip()}\n{content.text.strip()}"
        
        # 检查百度知道答案
        zhidao_answer = soup.find("div", class_="bd answer")
        if zhidao_answer:
            title = soup.find("h3", class_="t c-gap-bottom-small")
            if title:
                return f"{title.text.strip()}\n{zhidao_answer.text.strip()}"
        
        # 检查百度经验答案
        jingyan_answer = soup.find("div", class_="exp-content")
        if jingyan_answer:
            title = soup.find("h3", class_="t c-gap-bottom-small")
            if title:
                return f"{title.text.strip()}\n{jingyan_answer.text.strip()}"
        
        # 2. 寻找有机搜索结果
        organic_results = soup.find_all("div", class_="result")
        if organic_results:
            snippets = []
            for i, result in enumerate(organic_results[:3]):
                title_tag = result.find("h3", class_="t")
                content_tag = result.find("div", class_="c-abstract")
                
                if title_tag:
                    title = title_tag.text.strip()
                    snippet = content_tag.text.strip() if content_tag else ""
                    snippets.append(f"[{i+1}] {title}\n{snippet}")
            
            if snippets:
                return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"
    
from typing import Dict, Any

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")
        
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为 '{tool_name}' 的工具。")
