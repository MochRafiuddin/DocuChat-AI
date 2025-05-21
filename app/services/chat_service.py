from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from app.config import OPENAI_API_KEY
# from langchain_core.runnables import RunnableConfig

class ChatService:
    def __init__(self):
        self.chat_history = []
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
    
    def create_chain(self, retriever):
        if not retriever:
            return None
            
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            get_chat_history=lambda h: h
        )
    
    def get_response(self, chain, question):
        # Jika tidak ada chain (tidak ada dokumen)
        if not chain:
            messages = [{"role": "user", "content": question}]
            response = self.llm.invoke(messages).content
            
            # Simpan pesan ke riwayat
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=response))
            
            return response
        
        # Jika ada chain, gunakan untuk analisis dokumen
        response = chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Simpan pesan ke riwayat
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response['answer']))
        
        return response['answer']