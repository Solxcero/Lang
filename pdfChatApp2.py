import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
import os

# ✅ 최신 import 권장
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import ChatMessage, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.vectorstores import FAISS  # (FAISS는 커뮤니티 모듈 유지)
from pdfminer.high_level import extract_text

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

# ✅ 다국어 성능 좋은 임베딩 지정
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

class MarkdownStreamHandler(BaseCallbackHandler):
    def __init__(self, output_container, initial_content=""):
        self.output_container = output_container
        self.generated_content = initial_content
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.generated_content += token
        self.output_container.markdown(self.generated_content)

def extract_text_from_pdf(file):
    try:
        # ✅ UploadedFile을 안전하게 BytesIO로 변환
        data = file.read()
        file.seek(0)  # 혹시 이후 참조 대비
        return extract_text(BytesIO(data))
    except Exception as error:
        st.error(f"PDF 텍스트 추출 중 오류 발생: {error}")
        return ""

def handle_uploaded_file(file):
    if not file:
        return None, None

    document_text = extract_text_from_pdf(file) if file.type == 'application/pdf' else ""
    if not document_text:
        st.error("텍스트 추출 불가")
        return None, None

    # ✅ 조금 더 견고한 Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    document_chunks = text_splitter.create_documents([document_text])
    st.info(f"{len(document_chunks)} 개의 문서 단락 생성.")

    vectorstore = FAISS.from_documents(document_chunks, embedding_model)
    return vectorstore, document_text

def get_rag_response(user_query, vectorstore, callback_handler):
    if not vectorstore:
        st.error("No Vectorstore. Upload docs first.")
        return ""

    # ✅ 다국어 임베딩이면 그대로 검색해도 좋습니다.
    #   (정밀도를 더 높이고 싶으면 아래 주석처럼 영어 번역 후 검색 로직 추가)
    # eng_query = translate_to_english(user_query)  # 선택(아래 참고)
    retrieved_docs = vectorstore.similarity_search(user_query, k=3)
    retrieved_text = "\n".join(
        f"문서 {i+1} : {doc.page_content}" for i, doc in enumerate(retrieved_docs)
    )

    # ✅ 최신 signature: model=, 그리고 한국어 고정 지시 추가
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True, callbacks=[callback_handler])

    rag_prompt = [
        SystemMessage(content=(
            "당신은 문서 기반 어시스턴트입니다. "
            "오직 제공된 문서 내용을 근거로 답변하세요. "
            "정보가 없으면 '모르겠습니다'라고 답하세요. "
            "항상 한국어로 간결하고 정확하게 답변하세요."
        )),
        HumanMessage(content=f"질문: {user_query}\n\n참고 문서 발췌:\n{retrieved_text}")
    ]

    try:
        response = chat_model(rag_prompt)
        return response.content
    except Exception as error:
        st.error(f"응답 생성 중 오류 발생: {error}")
        return ""

st.set_page_config(page_title="My Doc's QnA")
st.title("📂 문서 기반 QnA 💬")

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        ChatMessage(role="assistant", content="안녕하세요, 문서를 기반으로 질문해 주세요. (답변은 한국어로 제공됩니다)")
    ]

uploaded_file = st.file_uploader("문서를 업로드하세요(PDF):", type=["pdf"])
if uploaded_file and uploaded_file != st.session_state.get("uploaded_file"):
    vectorstore, document_text = handle_uploaded_file(uploaded_file)
    if vectorstore:
        st.session_state["vectorstore"] = vectorstore
        st.session_state["uploaded_file"] = uploaded_file

chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        st.chat_message(message.role).write(message.content)

if user_query := st.chat_input("업로드된 문서를 기반으로 질문하세요 (한국어로 질문하세요)"):
    st.session_state.chat_history.append(ChatMessage(role="user", content=user_query))
    with chat_container:
        st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        stream_output = MarkdownStreamHandler(st.empty())
        assistant_response = get_rag_response(user_query, st.session_state.get("vectorstore"), stream_output)
        if assistant_response:
            st.session_state.chat_history.append(ChatMessage(role="assistant", content=assistant_response))
