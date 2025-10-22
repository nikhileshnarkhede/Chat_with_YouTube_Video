import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs
import re
from dotenv import load_dotenv

load_dotenv()

# ---- Functions ----
def get_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    if "youtu.be" in url:
        return url.split("/")[-1].split("?")[0]
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if "v" in query_params:
        return query_params["v"][0]
    match = re.search(r'/embed/([^/?]+)', parsed_url.path)
    if match:
        return match.group(1)
    return parsed_url.path.split("/")[-1]

def get_transcript(video_id: str) -> str:
    """Fetch transcript as a single string."""
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id)
        transcript = " ".join(chunk["text"] for chunk in transcript_list.to_raw_data())
        return transcript
    except TranscriptsDisabled:
        return None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# ---- Streamlit UI ----
st.title("YouTube Transcript Q&A")

# Step 1: Input YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL:")

if youtube_url:
    video_id = get_video_id(youtube_url)
    transcript = get_transcript(video_id)

    if transcript is None:
        st.warning("No captions available for this video.")
    else:
        st.success("Transcript loaded successfully!")

        # Step 2: Chunk & embed transcript
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        embeddings = HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')
        vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "lambda_mult": 0.5}
        )

        # Step 3: Setup LLM & prompt
        LLM = HuggingFaceEndpoint(repo_id='deepseek-ai/DeepSeek-R1', task='text-generation')
        llm = ChatHuggingFace(llm=LLM)

        prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
            """,
            input_variables=['context', 'question']
        )

        # Parallel chain
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        # Step 4: Input question
        question = st.text_input("Ask a question about this video:")
        if question:
            with st.spinner("Getting answer..."):
                answer = main_chain.invoke(question)
                st.markdown(f"**Answer:** {answer}")
