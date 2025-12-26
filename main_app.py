import requests
import json
import atexit
# 確保從正確的地方導入工具執行器
from mcp_handler import execute_tool, TOOLS_SCHEMA 

# ==========================================
# 🔧 設定區 (雙腦架構)
# ==========================================
# 1. 左腦 (工具判斷)
TOOL_MODEL = "qwen2.5:7b" 

# 2. 右腦 (對話生成)
# 建議：如果 Qwen 7B 還是會重複，您可以嘗試換回 "llama3.1:8b" 試試看，Llama 在邏輯控管上通常稍好一些
#CHAT_MODEL = "dolphin3:8b" 
CHAT_MODEL = "qwen2.5:7b" 

# ==========================================
# 🔌 硬體模組導入 (STT, TTS, Memory)
# ==========================================

"""
try:
    from STT import speech_to_text
    from TTS import text_to_speech
    from memory_chroma import add_memory, search_memory
    from speaker_identity import identify_speaker
except ImportError:
    print("⚠️ [警告] 找不到 STT/TTS/Memory 模組，將使用測試模式。")
    def speech_to_text(): 
        import time; time.sleep(2); return "測試語音輸入", "test.wav"
    def text_to_speech(text): return "output.wav"
    def add_memory(text, role): pass
    def search_memory(query, n_results=2): return []
    def identify_speaker(audio_path): return True, 0.99
"""
from STT import speech_to_text
from TTS import text_to_speech
from memory_chroma import add_memory, search_memory
from speaker_identity import identify_speaker

# ==========================================
# 🛠️ 輔助類別與函數
# ==========================================

def unload_model():
    """程式結束時通知 Ollama 釋放顯卡資源 (釋放兩個模型)"""
    print("\n🧹 [系統] 正在釋放模型資源...")
    api_url = "http://127.0.0.1:11434/api/generate"
    try:
        requests.post(api_url, json={"model": TOOL_MODEL, "keep_alive": 0}, timeout=1)
        requests.post(api_url, json={"model": CHAT_MODEL, "keep_alive": 0}, timeout=1)
    except:
        pass

# ==========================================
# 🧠 核心對話函數 (雙腦架構 - 抗重複優化版)
# ==========================================

def chat_with_dual_brain(system_prompt, user_text):
    url = "http://127.0.0.1:11434/api/chat"
    tool_results_text = ""

    # --- 第一階段：左腦 (工具判斷) ---
    print(f"⚡ [左腦 {TOOL_MODEL}] 正在監聽並判斷意圖...")
    
    # 🚨【關鍵修正】針對圖片描述 (Visual Description) 下達強制指令
    tool_system_prompt = (
        "You are a strict tool selector. Analyze the user input.\n"
        "Rules:\n"
        "1. If the input contains a **Visual Description** of a math problem (e.g., integrals, equations, physics), YOU MUST CALL 'ask_wolfram_alpha'.\n"
        "2. Translate the math problem into a clear English query for the tool (e.g., 'integrate 1/(1+e^sqrt(x)) from 0 to infinity').\n"
        "3. If the input asks for time, wiki, or search, call the respective tools.\n"
        "4. If no tool is needed, output nothing."
    )

    qwen_messages = [
        {"role": "system", "content": tool_system_prompt},
        {"role": "user", "content": user_text}
    ]

    qwen_payload = {
        "model": TOOL_MODEL,
        "messages": qwen_messages,
        "tools": TOOLS_SCHEMA,
        "stream": False,
        "options": {"temperature": 0.0} # 絕對理性
    }

    try:
        # 左腦逾時設定
        response = requests.post(url, json=qwen_payload, timeout=30)
        
        if response.status_code == 200:
            resp_json = response.json()
            message = resp_json.get("message", {})
            
            if message.get("tool_calls"):
                print(f"🔧 [左腦] 決定使用工具！數量: {len(message['tool_calls'])}")
                
                for tool in message["tool_calls"]:
                    func_name = tool["function"]["name"]
                    func_args = tool["function"]["arguments"]
                    
                    print(f"   └── 執行: {func_name} | 參數: {func_args}")
                    
                    try:
                        result = execute_tool(func_name, func_args)
                        # 截斷過長的工具結果，保留關鍵資訊
                        result_str = str(result)
                        if len(result_str) > 5000:
                            result_str = result_str[:5000] + "\n...(略)..."
                        tool_results_text += f"\n【工具 {func_name} 回傳結果】:\n{result_str}\n"
                    except Exception as tool_err:
                        print(f"❌ 工具執行錯誤: {tool_err}")
            else:
                # 左腦沒反應，通常是因為它覺得這只是一段描述
                # 如果 user_text 包含 "圖片內容分析"，我們可以強制提示使用者
                if "圖片內容分析" in user_text:
                    print("⚠️ 左腦未觸發工具，但偵測到圖片。")
        else:
            print(f"❌ 左腦 API 錯誤: {response.status_code}")

    except Exception as e:
        print(f"⚠️ 左腦錯誤: {e}")

    # --- 第二階段：右腦 (對話生成) ---
    print(f"🗣️ [右腦 {CHAT_MODEL}] 正在組織語言...")

    final_user_content = user_text
    if tool_results_text:
        final_user_content += f"\n\n(系統提示：以下是工具查詢到的真實資訊，請參考這些資訊回答用戶)\n{tool_results_text}"

    # 強制右腦使用 LaTeX 格式
    system_prompt += "\n重要：如果涉及數學公式，請務必使用 LaTeX 格式 (例如 $x^2$) 輸出，以便網頁渲染。"

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": final_user_content}
    ]

    chat_payload = {
        "model": CHAT_MODEL,
        "messages": chat_messages,
        "stream": True, 
        "options": {
            "temperature": 0.5,       
            "repeat_penalty": 1.25,   
            "num_predict": 4096,      
            "stop": ["<|endoftext|>", "user:", "model:", "</s>"] 
        } 
    }

    try:
        return requests.post(url, json=chat_payload, stream=True, timeout=90)
    except Exception as e:
        print(f"❌ 右腦連線錯誤: {e}")
        return None