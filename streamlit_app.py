import streamlit as st
import numpy as np
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import os
os.environ["OPENAI_API_KEY"] = "sk-proj-"

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def rag(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=['en']).to_raw_data()

        ## flatten transcript to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        # print(transcript)
        
        ## Splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        ## embedding and storing in vector stores
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={"k": 4})
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        prompt = PromptTemplate(
        template = """
        You are a helpful YouTube assistant.
        Answer ONLY from the provided transcript context,
        If the context is insufficient, just say you do not have the information for now.

        {context}
        Question: {question}
        
        """,
        input_variables=['context', 'question']
        )
        
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        
        main_chain = parallel_chain | prompt | llm | parser
        
        return main_chain, None

    except TranscriptsDisabled:
        return None, "Transcripts are disabled or unavailable for this video."
    
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

        
        
## streamlit app

st.title("YouTube Transcript Q&A Bot")
input_video_id = st.text_input("Enter the Youtube Video ID")
question = st.text_input("Ask a question about the Video")

if st.button("Get Answer"):
    if input_video_id and question:
        chain, error = rag(input_video_id)
        if error:
            st.error(error)
        else:
            answer = chain.invoke(question)
            st.success("Here's the answer:")
            st.write(answer)
    else:
        st.warning("Please provide both the Video ID and a question.")