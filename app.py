import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import tempfile
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from transformers import pipeline

# 頁面設置
st.set_page_config(
    page_title="語音轉文字工具",
    page_icon="🎤",
    layout="wide"
)

# 標題和說明
st.title("🎤 語音轉文字工具")
st.markdown("""
將音訊檔案或即時錄音轉換為文字 (支援 WAV/MP3 格式)
""")

# 初始化模型
@st.cache_resource
def load_model():
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
    
    tab1, tab2 = st.tabs(["📁 檔案上傳", "🎤 錄音輸入"])
    
    with tab1:
        st.header("檔案上傳 (最大500MB)")
        uploaded_file = st.file_uploader(
            "選擇音訊檔案",
            type=["wav", "mp3"],
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            if file_size > 500:
                st.error(f"檔案大小 {file_size:.2f}MB 超過500MB限制")
            else:
                st.audio(uploaded_file)
                
                if st.button("轉換檔案"):
                    with st.spinner("轉換中..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                                tmp_file.write(uploaded_file.getbuffer())
                                tmp_path = tmp_file.name
                            
                            if uploaded_file.name.endswith('.mp3'):
                                audio = AudioSegment.from_mp3(tmp_path)
                                wav_path = tmp_path + ".wav"
                                audio.export(wav_path, format="wav")
                                os.unlink(tmp_path)
                            else:
                                wav_path = tmp_path
                            
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
        st.warning("此功能需要瀏覽器麥克風權限，請確保已授予權限")
        
        audio_bytes = st.audio_recorder("點擊開始錄音", pause_threshold=2.0)
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("轉換錄音"):
                with st.spinner("轉換中..."):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        text = process_audio(tmp_path, model)
                        os.unlink(tmp_path)
                        
                        if text:
                            st.success("轉換成功!")
                            st.text_area("轉換結果", text, height=200)
                    except Exception as e:
                        st.error(f"轉換失敗: {str(e)}")

if __name__ == "__main__":
    main()