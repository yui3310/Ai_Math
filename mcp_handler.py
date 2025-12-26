# tool_registry.py
import inspect
import json
import functools
from pdf2image import convert_from_bytes # 新增：PDF 轉圖片庫
import base64
import fitz 

# 儲存工具定義 (給 Ollama 看)
TOOLS_SCHEMA = []
# 儲存實際函數 (給 Python 執行)
TOOLS_MAPPING = {}

def get_type_name(t):
    """將 Python type 轉為 JSON schema type"""
    if t == str: return "string"
    if t == int: return "integer"
    if t == float: return "number"
    if t == bool: return "boolean"
    return "string" # 預設

def register_tool(func):
    """
    這是一個裝飾器 (@register_tool)。
    只要掛在函數上，就會自動讀取函數的名稱、參數和註解，
    生成 Ollama 需要的 JSON Schema。
    """
    # 1. 取得函數資訊
    func_name = func.__name__
    doc = func.__doc__.strip() if func.__doc__ else "無描述"
    sig = inspect.signature(func)
    
    # 2. 構建參數 Schema
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # 忽略 self, cls 等參數 (如果有)
        if param_name in ['self', 'cls']: continue
        
        # 取得參數型別 (預設為 str)
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
        
        # 嘗試從 docstring 或是簡單設定描述 (這裡簡化處理，不強制解析 docstring 中的參數說明)
        # 如果您想要更完美的描述，建議參數名稱取直觀一點
        
        properties[param_name] = {
            "type": get_type_name(param_type),
            "description": f"Parameter: {param_name}" 
        }
        
        # 如果沒有預設值，就是必填
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # 3. 組合完整的 Tool Definition
    tool_def = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": doc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
    
    # 4. 註冊
    TOOLS_SCHEMA.append(tool_def)
    TOOLS_MAPPING[func_name] = func
    
    print(f"🔧 [系統] 已註冊工具: {func_name}")
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def execute_tool(tool_name, arguments):
    """通用執行入口"""
    func = TOOLS_MAPPING.get(tool_name)
    if not func:
        return f"錯誤: 找不到工具 '{tool_name}'"
    
    try:
        # 處理參數格式 (有時是 JSON 字串，有時是 dict)
        if isinstance(arguments, str):
            args = json.loads(arguments)
        else:
            args = arguments or {}
            
        print(f"⚙️ [執行工具] {tool_name} | 參數: {args}")
        return func(**args)
    except Exception as e:
        return f"執行工具發生錯誤: {e}"
    

    # mcp_handler.py
import datetime
import requests


# === 您的工具定義區 (盡情發揮！) ===

@register_tool
def get_current_time():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"現在時間: {now}"

def _analyze_image_with_ollama(image_base64, instruction=""):
    """
    內部共用函數：將 Base64 圖片發送給 Ollama 視覺模型
    """
    # 針對 Moondream 優化 Prompt
    final_prompt = "Describe this image." 
    if instruction:
        final_prompt = f"Describe this image. Focus on: {instruction}"

    payload = {
        "model": VISION_MODEL,
        "prompt": final_prompt,
        "images": [image_base64],
        "stream": False,
        "options": {"num_predict": 4096} # 限制輸出長度
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            description = result.get("response", "").strip()
            return description
        else:
            return f"Error: Vision model returned status {response.status_code}"
    except Exception as e:
        return f"Error: Connection failed {e}"

def process_uploaded_image(image_base64, user_text):
    """
    給網頁上傳專用的函數
    """
    print(f"🖼️ [系統] 收到網頁上傳圖片，正在分析...")
    description = _analyze_image_with_ollama(image_base64, user_text)
    
    return (
        f"【使用者上傳了一張圖片】\n"
        f"視覺模型描述(英文): {description}\n"
        f"----------------------------------\n"
        f"使用者問題: {user_text}\n"
        f"(請根據圖片描述回答使用者的問題)"
    )

from duckduckgo_search import DDGS


# --- 設定區 ---
# 建議使用 moondream (快且準) 或 qwen2.5vl
VISION_MODEL = "qwen2.5vl:3b" 
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

def _capture_window_to_base64():
    """內部函數：截取當前活動視窗並轉為 Base64"""
    try:
        screenshot = None
        
        # 嘗試鎖定當前視窗
        if gw:
            active_window = gw.getActiveWindow()
            if active_window:
                # 加一點邊距修正，避免切到邊框陰影
                screenshot = pyautogui.screenshot(region=(
                    active_window.left, 
                    active_window.top, 
                    active_window.width, 
                    active_window.height
                ))
                print(f"📸 [視覺] 已鎖定視窗: {active_window.title}")
        
        # 如果無法鎖定視窗或沒有安裝 gw，則全螢幕截圖
        if screenshot is None:
            print("⚠️ 無法鎖定視窗，進行全螢幕截圖。")
            screenshot = pyautogui.screenshot()

        # 圖片縮放 (Moondream 不需要太大張，512x512 效果最佳且快)
        screenshot.thumbnail((512, 512))
        
        buffered = io.BytesIO()
        screenshot.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
        
    except Exception as e:
        print(f"❌ 截圖失敗: {e}")
        return None

#@register_tool
def look_at_screen(instruction: str = "描述畫面"):
    """
    (System Action) 視覺能力：觀看使用者的電腦螢幕。
    當用戶說「你看」、「這張圖」、「畫面」時，【必須】使用此工具。
    instruction: (選填) 重點，例如 "這張圖" 或 "翻譯文字"。
    """
    print(f"👀 [視覺] 正在觀察: {instruction} ...")
    
    image_base64 = _capture_window_to_base64()
    if not image_base64:
        return "錯誤：無法截取畫面。"

    # 針對 Moondream 優化 Prompt
    # Moondream 對英文指令反應較好
    final_prompt = f"Describe this image briefly. Focus on: {instruction}"
    if VISION_MODEL == "moondream":
        final_prompt = "Describe this image." # Moondream 喜歡簡單指令

    payload = {
        "model": VISION_MODEL,
        "prompt": final_prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "num_predict": 100, # 限制輸出長度，避免廢話
            "repeat_penalty": 1.2
        }
    }

    try:
        # 直接呼叫 Ollama API (獨立於主對話模型)
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            description = result.get("response", "").strip()
            
            print(f"👀 [視覺結果]: {description[:100]}...")
            
            # 回傳給主模型 (Qwen/DeepSeek) 讓它翻譯並吐槽
            return (
                f"【視覺模組回傳的畫面描述 (英文)】\n{description}\n"
                f"(請根據以上描述，假裝是你親眼看到的，用中文回答用戶問題: '{instruction}')"
            )
        else:
            return f"視覺模型錯誤: {response.status_code}"
            
    except Exception as e:
        return f"視覺連線失敗: {e} (請確認 ollama pull {VISION_MODEL} 已執行)"
    
import wikipedia
try:
    wikipedia.set_lang("zh")
except:
    print("設定維基百科語言失敗，預設使用英文")

@register_tool
def search_wikipedia(query: str):
    """
    (System Action) 查詢維基百科 (Wikipedia)。
    適用情境：
    1. 用戶詢問「定義」類問題 (例如: 什麼是量子力學? 什麼是三體問題?)。
    2. 查詢歷史事件、人物介紹、科學名詞。
    3. 當 search_web (搜尋引擎) 資訊太雜亂時，使用此工具可獲得精準定義。
    """
    print(f"📖 [Wiki] 正在查閱: {query} ...")
    
    try:
        # 1. 搜尋條目 (Search)
        search_results = wikipedia.search(query)
        
        if not search_results:
            return "維基百科找不到相關條目。"
        
        # 2. 獲取最接近的頁面摘要 (Summary)
        # sentences=3 表示只抓前 3 句，避免內容太長爆字數
        # auto_suggest=False 避免它自作聰明跳轉到錯誤頁面
        try:
            summary = wikipedia.summary(search_results[0], sentences=3, auto_suggest=False)
            page_url = wikipedia.page(search_results[0], auto_suggest=False).url
            
            return (
                f"【維基百科摘要 - {search_results[0]}】\n"
                f"{summary}\n"
                f"(來源: {page_url})"
            )
            
        except wikipedia.exceptions.DisambiguationError as e:
            # 如果這個詞有歧義 (例如 'Joker' 可以是電影、撲克牌、蝙蝠俠反派)
            options = e.options[:5] # 只列出前 5 個選項
            return f"這個詞有多種含義，請告訴我您是指哪一個：\n" + ", ".join(options)
            
        except wikipedia.exceptions.PageError:
            return "找不到該具體頁面的內容。"

    except Exception as e:
        return f"維基百科查詢失敗: {e}"
    


    import sys
import io
import contextlib

# 引入數學庫供 exec 使用
import math
import sympy
import numpy as np
import xml.etree.ElementTree as ET 
WOLFRAM_APP_ID = 'TJE5A4WK2V'
@register_tool
def ask_wolfram_alpha(query: str):
    """
    (System Action) 使用 WolframAlpha 計算引擎解決數學、科學、物理、化學應用題。
    
    Args:
        query: 要查詢的問題。
        
        🚨【重要指令 / IMPORTANT INSTRUCTION】🚨
        WolframAlpha 只看懂英文！WolframAlpha ONLY understands ENGLISH!
        如果用戶的問題是中文，你必須先將其「翻譯成英文關鍵字」後再傳入此參數。
        不要傳入整句中文，請提取物理/數學關鍵字。

        【範例 / Examples】:
        - 用戶: "積分 x平方 sin x" 
          -> 你的參數 query="integrate x^2 sin(x)"
        - 用戶: "拋體運動 初速度 20m/s 角度 30度" 
          -> 你的參數 query="projectile motion v0=20m/s angle=30 deg"
        - 用戶: "水的密度"
          -> 你的參數 query="density of water"
        - 用戶: "把 x^2 + 5x + 6 因式分解"
          -> 你的參數 query="factor x^2 + 5x + 6"
    """
    # 這裡的代碼不需要大改，因為翻譯工作已經由 LLM 在呼叫前完成了
    # 我們只需要保留原本的邏輯即可
    
    print(f"🐺 [Wolfram] 正在計算 (Arg): {query}")
    
    if "YOUR_WOLFRAM_APP_ID" in WOLFRAM_APP_ID:
        return "錯誤: 請先在 mcp_handler.py 設定 WOLFRAM_APP_ID"

    # 使用 Full Results API (v2/query)
    api_url = "http://api.wolframalpha.com/v2/query" 
    
    params = {
        "appid": WOLFRAM_APP_ID,
        "input": query, 
        "units": "metric",
        "format": "plaintext",
        "output": "xml",
        "podstate": "Step-by-step solution" 
    }

    try:
        response = requests.get(api_url, params=params, timeout=None) # 延長一點時間給複雜運算
        
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            
            if root.attrib.get('success') != 'true':
                didyoumeans = root.findall('.//didyoumean')
                suggestions = [d.text for d in didyoumeans if d.text]
                msg = "WolframAlpha 無法理解此問題 (可能翻譯不夠精準)。"
                if suggestions:
                    msg += f" 建議嘗試搜尋: {', '.join(suggestions)}"
                return msg
            
            result_parts = []
            
            for pod in root.findall('.//pod'):
                title = pod.attrib.get('title', 'Result')
                
                subpod_texts = []
                for subpod in pod.findall('.//subpod'):
                    plaintext = subpod.find('plaintext')
                    if plaintext is not None and plaintext.text:
                        text = plaintext.text.strip()
                        if text:
                            subpod_texts.append(text)
                
                if subpod_texts:
                    content = "\n".join(subpod_texts)
                    result_parts.append(f"--- {title} ---\n{content}\n")
                        
            if not result_parts:
                return "WolframAlpha 執行成功，但未返回文字結果 (可能是純圖片)。"

            combined_result = "【WolframAlpha 分析結果】\n\n" + "\n".join(result_parts)
            return combined_result 

        else:
            return f"WolframAlpha API Error: {response.status_code}"
            
    except Exception as e:
        return f"WolframAlpha Connection Failed: {e}"
    
def _analyze_image_with_ollama(image_base64, instruction=""):
    """內部共用函數：將 Base64 圖片發送給 Ollama 視覺模型"""
    final_prompt = (
        "Please explicitly read and transcribe all text, numbers, and mathematical formulas in this image. "
        "Do not summarize; provide the full content verbatim."
    )
    if instruction:
        final_prompt += f" Focus on: {instruction}"

    payload = {
        "model": VISION_MODEL,
        "prompt": final_prompt,
        "images": [image_base64],
        "stream": False,
        "options": {"num_predict": 512} 
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=None)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Error: Vision model status {response.status_code}"
    except Exception as e:
        return f"Error: Connection failed {e}"

def process_uploaded_image(image_base64, user_text):
    """給網頁上傳圖片專用"""
    print(f"🖼️ [系統] 正在分析圖片...")
    description = _analyze_image_with_ollama(image_base64, user_text)
    print(description)
    return f"【圖片內容分析】\n{description}\n---\n使用者問題: {user_text}"

def process_pdf_pipeline(pdf_bytes, page_num, user_text):
    """
    新增：處理 PDF 檔案 (使用 PyMuPDF/fitz 引擎)
    1. 將 PDF 的指定頁面 (page_num) 轉為圖片
    2. 呼叫視覺模型分析該圖片
    """
    print(f"📄 [系統] 正在處理 PDF 第 {page_num} 頁...")
    
    if not fitz:
        return "錯誤：伺服器缺少 pymupdf 套件。請執行 `pip install pymupdf`。"

    try:
        # 使用 fitz 開啟 PDF 串流
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # 檢查頁碼 (注意：page_num 是從 1 開始，但 fitz 是從 0 開始)
        total_pages = len(doc)
        if page_num < 1 or page_num > total_pages:
            return f"錯誤：PDF 只有 {total_pages} 頁，您要求的第 {page_num} 頁超出範圍。"

        # 載入頁面 (0-indexed)
        page = doc.load_page(page_num - 1)
        
        # 轉為圖片 (Pixmap)
        # dpi=150 通常對於文字辨識已經足夠，若太模糊可調高到 300
        pix = page.get_pixmap(dpi=300)
        
        # 轉為 bytes (JPEG 格式)
        img_bytes = pix.tobytes("jpeg")
        
        # 轉為 Base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        # 呼叫視覺分析
        description = _analyze_image_with_ollama(img_base64, user_text)
        
        return (
            f"【PDF 第 {page_num} 頁內容分析】\n"
            f"{description}\n"
            f"----------------------------------\n"
            f"使用者問題: {user_text}\n"
            f"(請根據以上 PDF 頁面內容進行數學解題)"
        )
        
    except Exception as e:
        print(f"PDF 處理失敗: {e}")
        return f"PDF 讀取失敗: {e}"