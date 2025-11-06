import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import pandas as pd
import datetime
import os
import re

# --- 설정 및 초기화 ---

# 챗봇 제목 및 페이지 설정
st.set_page_config(
    page_title="방탈출 앱 고객 불편 응대 AI 챗봇",
    layout="wide"
)
st.title("🤖 방탈출 앱 고객 불편 응대 AI 챗봇")
st.caption("Gemini API (모델: gemini-2.0-flash) 활용")

# Streamlit secrets에서 API 키 로드 또는 임시 입력 UI 제공
def get_api_key():
    """API 키를 가져옵니다. st.secrets에서 먼저 시도하고, 없으면 사용자 입력을 받습니다."""
    try:
        # 1. st.secrets에서 키 로드 시도
        api_key = st.secrets["GEMINI_API_KEY"]
        st.sidebar.success("API Key 로드 완료 (st.secrets)")
        return api_key
    except (KeyError, AttributeError):
        # 2. st.secrets에 키가 없으면 사용자 입력 UI 표시
        st.sidebar.warning("`st.secrets['GEMINI_API_KEY']`를 찾을 수 없습니다. API 키를 입력해 주세요.")
        user_key = st.sidebar.text_input(
            "Gemini API Key 입력",
            type="password",
            placeholder="AI Studio 또는 Google Cloud에서 발급받은 키를 입력하세요."
        )
        if user_key:
            return user_key
        return None

GEMINI_API_KEY = get_api_key()

# API 키가 없으면 앱 실행 중단
if not GEMINI_API_KEY:
    st.info("Gemini API 키를 제공해야 챗봇을 사용할 수 있습니다.")
    st.stop()

# 모델 설정 (선택 가능 목록 및 기본값)
AVAILABLE_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-pro", "gemini-2.5-pro"]
DEFAULT_MODEL = "gemini-2.0-flash"

# Sidebar: 모델 선택, 로그 기록 옵션, 대화 초기화 버튼
with st.sidebar:
    st.subheader("⚙️ 설정")
    selected_model = st.selectbox(
        "사용할 Gemini 모델 선택 (gemini-2.0-flash 기본)",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0,
        key="model_select"
    )
    
    # CSV 자동 기록 옵션
    if 'log_to_csv' not in st.session_state:
        st.session_state.log_to_csv = False
        
    st.session_state.log_to_csv = st.checkbox("대화 내용 CSV 자동 기록", value=st.session_state.log_to_csv)
    
    st.markdown("---")
    
    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화", help="현재 대화 내용을 모두 지웁니다."):
        st.session_state.messages = []
        st.session_state.chat = initialize_chat_session(GEMINI_API_KEY, selected_model)
        st.session_state.history_reset_count = 0
        st.success("대화가 초기화되었습니다.")

# 시스템 프롬프트 정의
SYSTEM_PROMPT = """
당신은 '방탈출' 어플리케이션의 친절하고 유능한 고객 응대 챗봇입니다.
사용자는 어플 사용 중 (예: 결제 오류, 버그, 예약 문제, 게임 진행 불편 등) 겪은 불편/불만을 언급합니다.
다음 지침을 **엄격하게** 따르세요:

1.  **공감 및 정중한 응답:** 사용자의 불편사항에 대해 **정중하고 공감 어린 말투**로 응답하며, 불편을 끼쳐드린 점에 대해 깊이 사과합니다.
2.  **문제 구체화 및 수집:** 사용자가 겪은 발생 문제를 **구체적으로 정리**하여 (예: '무엇이', '언제', '어플 내 어느 테마 또는 과정에서', '어떻게') 정보를 수집하고, 이를 어플 운영 및 기술 담당자에게 전달하여 **신속히 해결하겠다**는 취지로 안내합니다.
3.  **이메일 요청:** 답변의 마지막에는 담당자가 검토 후 신속하게 답변을 드릴 수 있도록 **연락 가능한 이메일 주소**를 요청해야 합니다.
4.  **이메일 제공 거부 시:** 만일 사용자가 연락 제공을 원치 않으면:
    "고객님의 개인 정보 보호 의사를 존중합니다. 다만, 담당자의 상세 검토 내용을 별도로 전달드릴 방법이 없어, 이 점 양해 부탁드립니다."라고 **정중히 안내**합니다.
"""

# Gemini 채팅 세션 초기화 함수
def initialize_chat_session(api_key, model_name):
    """Gemini API 클라이언트를 설정하고 새 채팅 세션을 시작합니다."""
    try:
        genai.configure(api_key=api_key)
        
        # 안전 설정 (필요 시)
        safety_settings = [
            # 적절한 안전 설정을 추가할 수 있습니다. 예를 들어:
            # {
            #     "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
            #     "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            # }
        ]
        
        # Chat Session 생성
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            # safety_settings=safety_settings  # 안전 설정 적용
        )
        # st.session_state.messages는 Streamlit 메시지 UI 표시에 사용
        st.session_state.messages = [] 
        # model.start_chat()은 대화 히스토리 및 API 호출에 사용
        return model.start_chat(history=[])

    except Exception as e:
        st.error(f"Gemini API 설정 또는 세션 시작 중 오류가 발생했습니다: {e}")
        st.stop()

# 대화 히스토리 및 채팅 세션 관리
if "chat" not in st.session_state:
    st.session_state.chat = initialize_chat_session(GEMINI_API_KEY, selected_model)
    st.session_state.messages = [] # Streamlit UI용 메시지 리스트
    st.session_state.history_reset_count = 0 # 429 재시도 카운트용
    
# --- 대화 히스토리 CSV 로깅 함수 ---
LOG_FILE_PATH = "chat_log.csv"

def log_to_csv(role, content):
    """대화 내용을 CSV 파일에 기록합니다."""
    if not st.session_state.log_to_csv:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 모델명과 세션 정보 추가 (초기화 횟수로 세션 구분)
    session_info = f"{st.session_state.model_select}_{st.session_state.history_reset_count}"
    
    new_entry = pd.DataFrame([{
        "Timestamp": timestamp,
        "Session": session_info,
        "Role": role,
        "Content": content.replace('\n', ' ') # 줄바꿈 제거하여 CSV에 깔끔하게 기록
    }])
    
    # 파일이 존재하는지 확인하고, 헤더 추가 여부 결정
    if os.path.exists(LOG_FILE_PATH):
        new_entry.to_csv(LOG_FILE_PATH, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(LOG_FILE_PATH, mode='w', header=True, index=False)

# --- 429 재시도 및 대화 유지 로직 ---

def get_response_with_retry(prompt, model_name, max_retries=3):
    """
    Gemini API를 호출하고 429 에러 발생 시 대화 히스토리를 정리 후 재시도합니다.
    최근 6턴(User 3, Assistant 3)을 유지하려고 시도합니다.
    """
    current_chat_history = st.session_state.chat.history
    
    for attempt in range(max_retries):
        try:
            # API 호출
            response = st.session_state.chat.send_message(prompt, stream=True)
            return response
        
        except genai.errors.ResourceExhaustedError as e:
            # 429 에러 발생 시 처리
            st.warning(f"⚠️ API 호출 횟수 제한(429) 발생. ({attempt + 1}/{max_retries} 재시도 중...)")
            
            if attempt < max_retries - 1:
                # 최근 6턴(User 3, Assistant 3)만 남기고 히스토리 정리 후 재시도
                
                # Streamlit UI 메시지에서도 최근 6개만 남김
                st.session_state.messages = st.session_state.messages[-6:]
                
                # Gemini Chat.history에서 최근 6개의 Part만 남김
                # st.session_state.chat.history는 Content 객체의 리스트이며, 각 Content는 Parts 리스트를 가짐.
                # Content는 role과 parts로 구성됨.
                if len(current_chat_history) > 6:
                    st.session_state.chat.history = current_chat_history[-6:]
                    
                # Chat 세션을 아예 재시작 (히스토리 정리 효과)
                st.session_state.chat = initialize_chat_session(GEMINI_API_KEY, model_name)
                # 새로운 초기화 횟수 기록 (세션 구분용)
                st.session_state.history_reset_count += 1
                
                # 사용자에게 알림
                st.info("대화 히스토리가 길어져 최근 3번의 질문/답변만 남기고 세션을 재시작했습니다. 다시 질문해 주세요.")
                time.sleep(2) # 잠시 대기 후 재시도
                continue # 다음 시도로 넘어감 (이 시점에서 유저의 프롬프트는 아직 처리되지 않았으므로 다시 send_message를 시도해야 함)
            
            else:
                # 최대 재시도 횟수 초과
                st.error("API 호출 횟수 제한이 계속 발생하여 더 이상 진행할 수 없습니다. 잠시 후 다시 시도해 주세요.")
                st.session_state.messages.append({"role": "assistant", "content": "죄송합니다. 서비스 요청 과부하로 인해 응답할 수 없습니다. 잠시 후 다시 시도해 주십시오."})
                return None
        
        except Exception as e:
            st.error(f"예상치 못한 오류가 발생했습니다: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주십시오."})
            return None
    
    # max_retries를 모두 소진하고도 응답을 받지 못했을 경우
    return None

# --- UI 및 주요 로직 ---

# 현재 대화 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("불편하신 내용을 알려주시면 신속하게 도와드리겠습니다."):
    
    # 1. 사용자 메시지 기록 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    log_to_csv("user", prompt) # CSV 로깅
    
    # 2. AI 응답 생성 및 처리
    with st.chat_message("assistant"):
        
        # 429 재시도 로직을 포함한 응답 생성
        response_stream = get_response_with_retry(prompt, selected_model)
        
        if response_stream:
            # 스트리밍 응답을 위한 컨테이너
            placeholder = st.empty()
            full_response = ""
            
            # 스트리밍으로 응답 받기
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌") # 커서 효과
            
            placeholder.markdown(full_response) # 최종 응답 표시
            
            # 3. AI 응답 히스토리 기록 및 CSV 로깅
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            log_to_csv("assistant", full_response) # CSV 로깅

# --- 사이드바 추가 기능 ---

# 모델/세션 표시
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ 현재 상태")
st.sidebar.markdown(f"**모델:** `{selected_model}`")
st.sidebar.markdown(f"**세션 구분:** `RST: {st.session_state.history_reset_count}`")

# 로그 다운로드 버튼
if os.path.exists(LOG_FILE_PATH):
    with open(LOG_FILE_PATH, "rb") as file:
        st.sidebar.download_button(
            label="⬇️ 대화 로그 (CSV) 다운로드",
            data=file,
            file_name=LOG_FILE_PATH,
            mime="text/csv"
        )
else:
    st.sidebar.info("저장된 대화 로그가 없습니다.")