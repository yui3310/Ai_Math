# stt_module.py (使用 Whisper 離線辨識)

import speech_recognition as sr
import whisper
import os

# --- 配置參數 ---
LISTENING_TIMEOUT = 86000
PAUSE_THRESHOLD = 1.0
TEMP_AUDIO_FILE = "temp_audio.wav"
LANGUAGE = 'zh'

# --- 1. 全域載入 Whisper 模型 ---
try:
    WHISPER_MODEL_NAME = "small" # 可以改成 'small' 追求更高準確性
    print(f"🧠 [Whisper] 正在載入 '{WHISPER_MODEL_NAME}' 模型... (首次運行耗時較久)")
    model = whisper.load_model(WHISPER_MODEL_NAME) 
except Exception as e:
    print(f"❌ [Whisper] 載入模型失敗: {e}")
    model = None


def speech_to_text():
    """從麥克風錄音並將其轉換為文字，使用 Whisper 離線辨識。"""
    
    if model is None:
        return None

    r = sr.Recognizer()
    r.energy_threshold = 1000  
    r.dynamic_energy_threshold = False # 建議設為 False 以固定該數值
    r.pause_threshold = PAUSE_THRESHOLD
    
    with sr.Microphone() as source:
        print(f"[STT] 請說話... (等待 {LISTENING_TIMEOUT} 秒後超時)")
        
        # 解決 AssertionError：避免在設定 pause_threshold 後調用 adjust_for_ambient_noise 帶 duration 參數的衝突
        try:
             r.adjust_for_ambient_noise(source) 
        except AssertionError:
             print("[STT] adjust_for_ambient_noise 衝突，跳過校準。")
             pass 
        
        try:
            audio = r.listen(source, timeout=LISTENING_TIMEOUT)
        except sr.WaitTimeoutError:
            print(f"[STT] 超時 ({LISTENING_TIMEOUT} 秒)，沒有偵測到語音。")
            return None

    try:
        print("[Whisper] 正在進行離線辨識...")
        
        # 1. 存為臨時 WAV 檔案
        with open(TEMP_AUDIO_FILE, "wb") as f:
            f.write(audio.get_wav_data())

        # 2. 使用 Whisper 轉錄
        result = model.transcribe(
            TEMP_AUDIO_FILE, 
            fp16=False, 
            language=LANGUAGE,
            # initial_prompt 幫助 Whisper 更好地開始辨識
            initial_prompt="你好，請問"
        )
        text = result["text"].strip()
        
        # 3. 清理臨時檔案
        if os.path.exists(TEMP_AUDIO_FILE):
            os.remove(TEMP_AUDIO_FILE)

        print(f"[STT] 您說了: {text}")
        return text, TEMP_AUDIO_FILE
        
    except Exception as e:
        print(f"❌ [Whisper] 辨識過程中發生錯誤: {e}")
        if os.path.exists(TEMP_AUDIO_FILE):
             os.remove(TEMP_AUDIO_FILE)
        return None