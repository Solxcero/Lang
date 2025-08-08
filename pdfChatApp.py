import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import ChatMessage, HumanMessage, SystemMessage
from langchain.callbacks.base  import BaseCallbackHandler
from pdfminer.high_level import extract_text

import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.environ["OPENAI_API_KEY"] 

embedding_model = OpenAIEmbeddings()

class MarkdownStreamHandler(BaseCallbackHandler):

    '''Streamlit 마크다운 컨테이너에 생성된 토큰을 실시간으로 스트리밍하는 사용자 정의 핸들러'''

    def __init__(self, output_container, initial_content=""):
        self.output_container = output_container
        self.generated_content = initial_content

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.generated_content += token
        self.output_container.markdown(self.generated_content)

def extract_text_from_pdf(file):
    try:
        return extract_text(file)
    except Exception as error:
        st.error(f"PDF 텍스트 추출 중 오류 발생: {error}")
        return ""
    

def handle_uploaded_file(file):
    '''업로드 된 PDF 파일을 처리하고 벡터 스토어 준비'''
    if not file:
        return None, None
    
    document_text = extract_text_from_pdf(file) if file.type == 'application/pdf' else ""
    if not document_text:
        st.error("텍스트 추출 불가")
        return None, None
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )

    document_chunks = text_splitter.create_documents([document_text])
    st.info(f"{len(document_chunks)} 개의 문서 단락 생성.")

    # 유사성 검색을 위한 벡터 스토어 생성
    vectorstore = FAISS.from_documents(document_chunks, embedding_model)
    return vectorstore, document_text

def get_rag_response(user_query, vectorstore, callback_handler):
    if not vectorstore:
        st.error("No Vectorstore. Upload docs first.")
        return ""
    
    retrieved_docs = vectorstore.similarity_search(user_query, k=3)
    retrieved_text = "\n".join(f"문서 {i+1} : {doc.page_content}" for i, doc in enumerate(retrieved_docs))

    chat_model = ChatOpenAI(model_name = "gpt-4", temperature=0, streaming=True, callbacks=[callback_handler])

    rag_prompt = [
        SystemMessage(content = "제공된 문서를 기반으로 사용자의 질문에 답변하세요. 정보가 없으면 '모르겠습니다' 라고 답변하세요"),
        HumanMessage(content=f"질문 : {user_query}\n\n{retrieved_text}")

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
        ChatMessage(role="assistant", content="안녕하세요, 문서를 기반으로 질문해 주세요.")
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

if user_query := st.chat_input("업로드된 문서를 기반으로 질문하세요"):
    st.session_state.chat_history.append(ChatMessage(role="user", content=user_query))

    with chat_container:
        st.chat_message("user").write(user_query)

    
    with st.chat_message("assistant"):
        stream_output = MarkdownStreamHandler(st.empty())
        assistant_response = get_rag_response(user_query, st.session_state.get("vectorstore"), stream_output)

        if assistant_response:
            st.session_state.chat_history.append(ChatMessage(role="assistant", content=assistant_response))
