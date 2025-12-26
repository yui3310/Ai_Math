# langchain_bot.py (LangGraph 現代版)
import datetime
import math
import io
import pyautogui
import ollama

# --- LangChain 核心 ---
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# --- 🔥 關鍵升級：使用 LangGraph 的預建 Agent ---
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. 工具定義 (Tools) - 這部分跟之前一樣
# ==========================================

@tool
def get_current_time():
    """獲取當前系統時間。"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str):
    """
    數學計算機。支援加減乘除、次方(**)、開根號(sqrt)。
    expression: 數學算式字串，例如 "33 * 45"
    """
    allowed_names = {"sqrt": math.sqrt, "pow": math.pow, "pi": math.pi}
    try:
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                return f"錯誤：不允許使用函數 {name}"
        return str(eval(code, {"__builtins__": {}}, allowed_names))
    except Exception as e:
        return f"計算錯誤: {e}"

@tool
def look_at_screen(instruction: str = "描述畫面"):
    """
    視覺能力：觀看使用者的電腦螢幕。
    """
    print(f"📸 [視覺] 正在觀察螢幕... ({instruction})")
    try:
        screenshot = pyautogui.screenshot()
        screenshot.thumbnail((1024, 1024))
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        response = ollama.chat(
            model='moondream', 
            messages=[{
                'role': 'user',
                'content': f"Describe this image. Focus on: {instruction}",
                'images': [img_bytes]
            }]
        )
        return f"【視覺描述】: {response['message']['content']}"
    except Exception as e:
        return f"視覺分析失敗: {e}"

# 搜尋工具
search_tool = DuckDuckGoSearchRun(name="search_web", description="搜尋網路即時資訊、新聞、天氣。")

tools = [get_current_time, calculate, search_tool, look_at_screen]

# ==========================================
# 2. 初始化模型與 LangGraph Agent
# ==========================================

# 您的模型
MODEL_NAME = "hf.co/MaziyarPanahi/DeepSeek-R1-0528-Qwen3-8B-GGUF:Q4_K_M" 

llm = ChatOllama(
    model=MODEL_NAME, 
    temperature=0.1,
    keep_alive="5m"
)

# 🔥 這裡使用 LangGraph 的 create_react_agent
# 它直接取代了舊版的 AgentExecutor，更穩定
agent_executor = create_react_agent(llm, tools)

# ==========================================
# 3. 對外接口
# ==========================================

def chat_with_langchain(user_input: str, system_context: str = ""):
    """
    使用 LangGraph 處理對話
    """
    # 組合 System Prompt (人設 + RAG 記憶)
    final_system_prompt = (
        "你是一個叫「愛音」的 AI Vtuber，性格可愛帶點毒舌，說話口語化且有主見。\n"
        "遇到不知道的事情(時間、新聞、數學、畫面)請務必使用工具查詢。\n"
        "查詢後，請用你的語氣回答用戶。\n"
        f"{system_context}"
    )

    try:
        # LangGraph 的輸入格式是 messages 列表
        messages = [
            SystemMessage(content=final_system_prompt),
            HumanMessage(content=user_input)
        ]
        
        # 執行 invoke
        # LangGraph 會自動處理工具調用迴圈
        result = agent_executor.invoke({"messages": messages})
        
        # 取得最後一條訊息 (也就是 AI 的最終回答)
        final_response = result["messages"][-1].content
        return final_response

    except Exception as e:
        return f"愛音核心錯誤 (LangGraph): {str(e)}"