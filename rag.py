# Loading essential libraries
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import streamlit as st
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import re

#  function for extracting video id from yt url
def extract_youtube_video_id(url):
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
load_dotenv()


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
llm= HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-0528", type="text-generation",temperature=0.2)
model=ChatHuggingFace(llm=llm)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


st.header('YT Videos Info Generator')

# Getting the transcript of the uplaoded video
video_link=str(st.text_input('Enter your yt video HTTP link'))
video_id=extract_youtube_video_id(video_link)
question=st.text_input('Enter your question about the video')
query_video=""
if st.button('Answer'):
    query_video=video_id

    try:
        # If you don’t care which language, this returns the “best” one
        transcript_list = YouTubeTranscriptApi.get_transcript(query_video, languages=["en"])

        # Flatten it to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print(transcript)

    except TranscriptsDisabled:
        print("No captions available for this video.")


    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )


    retrieved_docs    = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # context_text

    final_prompt = prompt.invoke({"context": context_text, "question": question})

    answer = model.invoke(final_prompt)

    st.write(answer.content)

    # executed the whole code using chains also 

    # parallel_chain = RunnableParallel({
    # 'context': retriever | RunnableLambda(format_docs),
    # 'question': RunnablePassthrough()
    # })

    # parser = StrOutputParser()

    # main_chain = parallel_chain | prompt | model | parser

    # main_chain.invoke('summarize the video for me')
