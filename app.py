#v1.0.1
import os
# 禁用 Streamlit 文件監視以避免 torch 兼容性問題
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import tempfile
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import subprocess
import sys

# 設置上傳限制為500MB (適用於Streamlit Cloud等環境)
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "500"

# 檢查並安裝必要套件
def check_and_install_packages():
    required_packages = {
        'streamlit': 'streamlit',
        'pydub': 'pydub',
        'soundfile': 'soundfile',
        'numpy': 'numpy',
        'transformers': 'transformers',
        'torch': 'torch',
        'streamlit-webrtc': 'streamlit_webrtc',
        'ffmpeg-python': 'ffmpeg'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.warning(f"正在安裝缺少的套件: {', '.join(missing_packages)}...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.success("套件安裝完成! 請重新載入頁面。")
        st.experimental_rerun()

check_and_install_packages()

# 頁面設置
st.set_page_config(
    page_title="語音轉文字工具v1.0.1",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 標題和說明
st.title("🎤 語音轉文字工具")
st.markdown("""
此工具使用OpenAI的Whisper模型將語音轉換為文字，完全免費且可在線上使用。
""")

# 初始化模型
@st.cache_resource
def load_model():
    from transformers import pipeline
    try:
        return pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device="cpu"
        )
    except Exception as e:
        st.error(f"模型加載失敗: {str(e)}")
        return None

# 音訊處理函數
def process_audio(file_path, model):
    try:
        audio_data, sample_rate = sf.read(file_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_wav.name, audio_data, sample_rate, format='WAV')
        
        result = model(temp_wav.name)
        os.unlink(temp_wav.name)
        return result["text"]
    except Exception as e:
        st.error(f"音訊處理錯誤: {str(e)}")
        return None

# 主應用界面
def main():
    model = load_model()
    
    tab1, tab2 = st.tabs(["📁 檔案上傳 (最大30秒)", "🎤 錄音輸入"])
    
    with tab1:
        st.header("📁 檔案上傳 (最大30秒)")
        uploaded_file = st.file_uploader(
            "選擇音訊檔案 (WAV, MP3)",
            type=["wav", "mp3"],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # 顯示檔案大小資訊
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # 轉換為MB
            st.info(f"檔案大小: {file_size:.2f}MB")
            
            if file_size > 500:
                st.error("檔案大小超過500MB限制")
            else:
                st.audio(uploaded_file)
                
                if st.button("轉換檔案", key="file_convert"):
                    with st.spinner("轉換中，請稍候..."):
                        try:
                            # 使用 getvalue() 獲取檔案內容而不是 chunks()
                            file_content = uploaded_file.getvalue()
                            
                            # 保存到臨時文件
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                                tmp_file.write(file_content)
                                tmp_path = tmp_file.name
                            
                            # 轉換為WAV格式
                            if uploaded_file.name.endswith('.mp3'):
                                try:
                                    audio = AudioSegment.from_mp3(tmp_path)
                                    wav_path = tmp_path + ".wav"
                                    audio.export(wav_path, format="wav")
                                    os.unlink(tmp_path)
                                except Exception as e:
                                    st.error(f"MP3轉換失敗: {str(e)}")
                                    os.unlink(tmp_path)
                                    return
                            else:
                                wav_path = tmp_path
                            
                            # 執行轉換
                            text = process_audio(wav_path, model)
                            os.unlink(wav_path)
                            
                            if text:
                                st.success("轉換成功!")
                                st.text_area("轉換結果", text, height=200)
                        except Exception as e:
                            st.error(f"轉換失敗: {str(e)}")
                            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                os.unlink(tmp_path)
    
    with tab2:
        st.header("錄音輸入")
        st.warning("注意: 錄音功能需要瀏覽器麥克風權限")
        
        webrtc_ctx = webrtc_streamer(
            key="record_audio",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"audio": True, "video": False}
        )
        
        if webrtc_ctx.audio_receiver:
            if st.button("轉換錄音", key="record_convert"):
                with st.spinner("正在處理..."):
                    try:
                        audio_frames = []
                        for _ in range(100):
                            frame = webrtc_ctx.audio_receiver.get_frame(timeout=1)
                            if frame: audio_frames.append(frame)
                        
                        if not audio_frames:
                            st.error("未收到音訊數據")
                            return
                            
                        audio_np = np.concatenate([f.to_ndarray() for f in audio_frames])
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            sf.write(tmp_file.name, audio_np, 16000)
                            text = process_audio(tmp_file.name, model)
                            os.unlink(tmp_file.name)
                        
                        if text:
                            st.success("轉換成功!")
                            st.text_area("轉換結果", text, height=200)
                    except Exception as e:
                        st.error(f"轉換失敗: {str(e)}")


    # 側邊欄資訊
    st.sidebar.markdown("""
    ### 使用說明
    1. 選擇輸入方式: 上傳檔案或錄音
    2. 點擊轉換按鈕
    3. 查看並複製轉換結果

    ### 技術資訊
    - 使用OpenAI Whisper小型模型
    - 支援中文、英文等多種語言
    - 完全免費開源方案
    """)


if __name__ == "__main__":
    main()