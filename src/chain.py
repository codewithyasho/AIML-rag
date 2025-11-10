"""
RAG chain setup
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from config.settings import GROQ_MODEL, GROQ_TEMPERATURE, RETRIEVAL_K


def get_llm():
    """Initialize LLM"""
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE
    )
    return llm


def get_prompt():
    """Get RAG prompt template"""
    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context. 
        Think step by step before answering. If you don't know the answer, just say that you don't know.

        <context>
        {context}
        </context>

        Question: {input}
    """)
    return prompt


def create_rag_chain(vectorstore):
    """Create complete RAG chain"""
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_K}
    )

    # Get LLM and prompt
    llm = get_llm()
    prompt = get_prompt()

    # Create chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("🔗 RAG chain created successfully")
    return rag_chain
