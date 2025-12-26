# vision_module.py (根本解決版：Moondream + 視窗鎖定)

import io
import base64
import requests
import pyautogui
import pygetwindow as gw
from PIL import Image

# 🧠 根本解法 1: 改用 Moondream (更老實、更快)
# 請先執行: ollama pull moondream
VISION_MODEL = "qwen2.5vl:3b" 
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

def capture_active_window_to_base64():
    """
    👀 根本解法 2: 只看「當前視窗」，排除工作列和雜訊
    """
    try:
        # 1. 獲取當前最上層的視窗
        active_window = gw.getActiveWindow()
        
        if active_window is None:
            print("⚠️ 無法偵測當前視窗，改為全螢幕截圖。")
            screenshot = pyautogui.screenshot()
        else:
            # 2. 根據視窗位置截圖 (去除周圍雜訊)
            # 這裡加一點邊距修正 (通常 Windows 視窗邊框會有陰影，稍微內縮一點更準)
            screenshot = pyautogui.screenshot(region=(
                active_window.left, 
                active_window.top, 
                active_window.width, 
                active_window.height
            ))
            print(f"📸 [視覺] 已鎖定視窗: {active_window.title}")

        # 3. 圖片縮放 (Moondream 不需要太大張，縮小能大幅加速)
        # 限制最大邊長為 512 (Moondream 的最佳解析度)
        screenshot.thumbnail((512, 512))
        
        # 4. 轉 Base64
        buffered = io.BytesIO()
        screenshot.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str
    except Exception as e:
        print(f"❌ 截圖失敗: {e}")
        return None

def analyze_screen(prompt_text="描述這張圖"):
    print("👀 [視覺] 正在觀察...")
    
    image_base64 = capture_active_window_to_base64()
    if not image_base64:
        return "無法截取畫面。"

    # Moondream 的 Prompt 越簡單越好
    # 如果是用 moondream，建議用英文問，它反應最快，然後我們再叫愛音翻譯成中文吐槽
    if VISION_MODEL == "moondream":
        final_prompt = "Describe this image briefly." 
    else:
        final_prompt = prompt_text

    payload = {
        "model": VISION_MODEL,
        "prompt": final_prompt,
        "images": [image_base64],
        "stream": False,

        "options": {
            #"num_predict": 80,   # 👈 殺手鐧：最多只能講 80 個 token (約 50-60 個中文字)
            #"temperature": 0.2,  # 降低溫度，讓它專注，不要發散
            "repeat_penalty": 1.3, # 👈 重複懲罰調高 (預設 1.1)，只要重複就扣分
            #"top_k": 10
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
            description = result.get("response", "")
            if len(description) > 200:
                description = description[:200] + "..."
            
            print(f"👀 [視覺看到]: {description}")
            return description
        else:
            return f"視覺模型錯誤: {response.status_code}"
    except Exception as e:
        return f"視覺連線失敗: {e}"
