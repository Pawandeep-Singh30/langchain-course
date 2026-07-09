
import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

print("Initializing components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = OllamaLLM(model="llama3.2:latest", temperature=0.0)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME_TXT"],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """ Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

def format_docs(docs):
    """ Format retrieved documents into a single string. """
    return "\n\n".join(doc.page_content for doc in docs)

# ===========================================================
# IMPLEMENTATION 1: without LCEL
# ===========================================================

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LLM.
    Manually retrieves documents, formats them and generates a response.
    
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error prone
    """
    #Step 1: retrieve relevant documents
    docs = retriever.invoke(query)

    #Step 2: Format documents into context string
    context = format_docs(docs)

    #Step 3: Format the prompt with context and question
    prompt = prompt_template.format(context = context, question = query)

    #Step 4: Invoke llm with the formatted messages
    response = llm.invoke(prompt)

    #Step 5: Return the content
    return response

# ===========================================================
# IMPLEMENTATION 2: with LCEL - BETTER APPROACH
# ===========================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain with LCEL.
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-lcel approach:
    - Declarative and composable: Easy to chain operations with pipe operator(|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() are available
    - Batch processing: chain.batch() for multiple Inputs
    - Type safety: Beter integration with langchain's type system
    - less code: more concise and readable
    - reusable: chain can be saved. shared and composed with other chains
    - better debugging: langchain provides better observability tools
    """

    retrieval_chain = (
        RunnablePassthrough.assign( 
            context = itemgetter("question") |retriever | format_docs
        )
        | prompt_template 
        | llm 
        | StrOutputParser()
    )
    return retrieval_chain





if __name__ == "__main__":
    print("Retrieving...")

    #Query
    query = "What is Pinecone in machine learning?"

    # ===========================================================
    # Option 0: Raw invocation without RAG
    # ===========================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM invocation (No RAG)")
    print("=" * 70 )
    result_raw = llm.invoke(query)
    print("\nAnswer:")
    print(result_raw)


    # ===========================================================
    # Option 1: Use implementation without LCEL
    # ===========================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: without LCEL")
    print("=" * 70 )
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)


    # ===========================================================
    # Option 2: Use implementation with LCEL
    # ===========================================================

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: with LCEL")
    print("=" * 70 )
    print("Why LCEL is better?")
    print("- more concise and declarative")
    print("- built-in streaming: chain.stream()") 
    print("- built-in async: chain.ainvoke()")
    print("- east to compose ith other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)

