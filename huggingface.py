# huggingface_r1.py — 使用 HuggingFace DeepSeek-R1-8B + 流式輸出
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

# 🌟 你可換其他模型（7B/8B/14B/32B）
#HF_MODEL_NAME = "deepseek-ai/DeepSeek-R1"
HF_MODEL_NAME = r"D:\ai_vtuber\DeepSeek-R1-8B"

# 全域模型快取：只載入一次
_tokenizer = None
_model = None


def load_hf_model():
    """
    載入 DeepSeek R1 模型（只載入一次）
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    print("🧠 [HF] 正在載入 DeepSeek-R1 模型（初次載入會花時間）...")

    _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        dtype=torch.bfloat16,            
        quantization_config=bnb_config, 
        device_map="auto",               
        trust_remote_code=True,
    )

    print("✅ [HF] 模型載入完成！")
    return _tokenizer, _model


def get_ollama_response_stream(prompt: str, model_name: str = None):
    """
    將 HuggingFace Streaming 改成正確的 Thread + TextIteratorStreamer 寫法。
    """

    tokenizer, model = load_hf_model()

    # 🌟 修正 4：使用 TextIteratorStreamer，它專門為 Python Generator 設計
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True,             # 跳過 Prompt 本身
        skip_special_tokens=True      # 跳過 <|end of sentence|> 等特殊標記
    )

    # 處理輸入 Prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("--- [DEBUG] 成功建立輸入張量並準備生成 ---")

    # 💥 修正 5：在單獨的執行緒中運行 model.generate() 
    # 讓主程式可以同時接收 streamer 的輸出
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=768,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        streamer=streamer, # 將 streamer 傳入
    )

    # 啟動生成執行緒
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 🌟 修正 6：主程式 yield streamer.on_text_stream 
    # 這是 TextIteratorStreamer 專門設計的迭代器
    for new_text in streamer:
        yield new_text

    # 等待生成結束
    thread.join()
