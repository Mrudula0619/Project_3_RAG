import os
import pandas as pd
from typing import TypedDict, List
import great_expectations as ge

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph


# ---------------------------------------------------------
# 1. Ingestion & Data Quality Validation
# ---------------------------------------------------------
def setup_vectorstore(file_path: str):
    print("\n--- [1/3] Loading & Validating Documents ---")
    loader = TextLoader(file_path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = splitter.split_documents(docs)

    # Validate document chunks with Great Expectations
    df = pd.DataFrame([{"text": s.page_content} for s in splits])
    ge_df = ge.from_pandas(df)
    
    val_nulls = ge_df.expect_column_values_to_not_be_null("text")
    val_length = ge_df.expect_column_value_lengths_to_be_between("text", min_value=10, max_value=1000)

    if not (val_nulls.success and val_length.success):
        raise ValueError("Data validation failed during ingestion!")
    print("✓ Great Expectations Validation Passed.")

    # Embed locally using HuggingFace & Index into ChromaDB
    print("--- [2/3] Indexing Documents into ChromaDB (Local HuggingFace) ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="rag_collection"
    )
    print("✓ Vector store successfully populated.")
    return vectorstore.as_retriever()


# ---------------------------------------------------------
# 2. LangGraph Workflow Construction
# ---------------------------------------------------------
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    retry_count: int
    is_grounded: bool


def build_app(retriever):
    llm = ChatOllama(model="llama3.2", temperature=0)

    def retrieve_node(state: GraphState):
        print("\n  --> [Node] Retrieving context from ChromaDB...")
        docs = retriever.invoke(state["question"])
        return {"documents": docs, "question": state["question"]}

    def grade_documents_node(state: GraphState):
        print("  --> [Node] Grading retrieved document relevance...")
        question = state["question"]
        relevant_docs = []
        for doc in state["documents"]:
            prompt = f"System: Respond ONLY with the single word 'YES' or 'NO'. Do not add any preamble.\nIs this document relevant to the query '{question}'?\nDoc: {doc.page_content}"
            res = llm.invoke(prompt).content.strip().upper()
            if "YES" in res:
                relevant_docs.append(doc)
        return {"documents": relevant_docs, "question": question}

    def generate_node(state: GraphState):
        print("  --> [Node] Generating answer from local Ollama model...")
        context = "\n\n".join([d.page_content for d in state["documents"]])
        prompt = f"System: Provide a direct answer using ONLY the context provided.\nContext:\n{context}\n\nQuestion: {state['question']}"
        res = llm.invoke(prompt).content.strip()
        return {"generation": res}

    def hallucination_check_node(state: GraphState):
        print("  --> [Guardrail Node] Checking for hallucinations...")
        generation = state["generation"]
        documents = state["documents"]
        
        context = "\n\n".join([d.page_content for d in documents])
        prompt = f"""System: Assess if the Answer is directly derived from Facts. Respond ONLY with 'YES' or 'NO'.

Facts:
{context}

Answer:
{generation}"""
        res = llm.invoke(prompt).content.strip().upper()
        
        if "NO" not in res and "YES" in res:
            print("  ✓ Generation passed hallucination check.")
            return {"is_grounded": True}
            
        print("  ⚠️ Hallucination detected!")
        return {"is_grounded": False}

    def rewrite_query_node(state: GraphState):
        print("  --> [Self-Correction] Rewriting query for better retrieval...")
        prompt = f"System: Rewrite the input query to be a concise 1-sentence vector search query. Output ONLY the raw search query, nothing else.\nQuery: {state['question']}"
        res = llm.invoke(prompt).content.strip().replace('"', '')
        retries = state.get("retry_count", 0) + 1
        return {"question": res, "retry_count": retries}

    # Conditional Routing Logic
    def decide_after_grading(state: GraphState):
        if not state["documents"]:
            if state.get("retry_count", 0) >= 2:
                return "generate"
            return "rewrite"
        return "generate"

    def decide_after_hallucination_check(state: GraphState):
        if state.get("is_grounded", False):
            return "end"
        if state.get("retry_count", 0) >= 2:
            return "end"
        return "rewrite"

    # Assemble Graph
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_docs", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("rewrite", rewrite_query_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_docs")
    workflow.add_conditional_edges(
        "grade_docs", 
        decide_after_grading, 
        {"rewrite": "rewrite", "generate": "generate"}
    )
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_after_hallucination_check,
        {"end": END, "rewrite": "rewrite"}
    )
    workflow.add_edge("rewrite", "retrieve")

    return workflow.compile()


# ---------------------------------------------------------
# 3. Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- Starting Execution (Local Mode) ---")
    retriever = setup_vectorstore("sample_docs.txt")
    app = build_app(retriever)

    # Print ASCII Graph Representation
    try:
        print("\n--- [LangGraph DAG Structure] ---")
        print(app.get_graph().draw_ascii())
    except Exception as e:
        print(f"Skipped ASCII draw: {e}")

    # Run Query
    print("\n--- [3/3] Running Query Through Self-Correcting Graph ---")
    query = "What tools are used for validation and monitoring?"
    
    inputs = {"question": query, "retry_count": 0, "is_grounded": False}
    output = app.invoke(inputs)

    print("\n==================================================")
    print(f" FINAL QUESTION : {output['question']}")
    print(f" FINAL ANSWER   : {output['generation']}")
    print("==================================================\n")
