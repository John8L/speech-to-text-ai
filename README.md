# 檔案下載
英文語錄23秒.mp3
https://reurl.cc/AMD1qK

# 語音轉文字應用

這是一個使用 OpenAI Whisper 模型的語音轉文字工具，支援檔案上傳和錄音輸入。

## 功能特點
- 支援 WAV/MP3 格式檔案上傳 (最大500MB)
- 支援瀏覽器錄音輸入
- 使用 Whisper-small 模型進行語音識別
- 簡單易用的網頁界面

## 安裝與執行

1. clone repository：
   ```bash
   git clone https://github.com/John8L/speech-to-text-ai.git
   cd speech-to-text-app
   ```

2. 安裝dependency：
   ```bash
   pip install -r requirements.txt
   ```

3. 執行應用：
   ```bash
   streamlit run app.py
   ```

## 系統需求
- Python 3.8+
- FFmpeg (用於音訊格式轉換)
- webapp
