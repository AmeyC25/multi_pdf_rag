import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader
import requests
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Set page config
st.set_page_config(page_title="Advanced PDF QA", page_icon="🧠", layout="wide")

# Custom types
class DocumentMetadata(BaseModel):
    source: str
    page: int
    section: Optional[str] = None
    doc_type: Optional[str] = None

class EnhancedDocument(Document):
    metadata: DocumentMetadata

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'doc_metadata' not in st.session_state:
    st.session_state.doc_metadata = {}
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'doc_types' not in st.session_state:
    st.session_state.doc_types = {}

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class IntelligentSplitter:
    def __init__(self):
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
        )

    def detect_document_type(self, text: str) -> str:
        """Classify document type based on content patterns"""
        if re.search(r'\babstract\b|\bintroduction\b|\breferences\b', text, re.I):
            return "research_paper"
        elif re.search(r'\btable of contents\b|\bchapter\b|\bsection\b', text, re.I):
            return "book"
        elif re.search(r'\bprocedure\b|\bpolicy\b|\bguideline\b', text, re.I):
            return "policy"
        elif re.search(r'\bassembly\b|\binstallation\b|\btroubleshooting\b', text, re.I):
            return "manual"
        return "generic"

    def extract_sections(self, text: str, doc_type: str) -> List[Dict[str, Any]]:
        """Extract logical sections based on document type"""
        if doc_type == "research_paper":
            return self._split_research_paper(text)
        elif doc_type == "manual":
            return self._split_manual(text)
        return [{"content": text, "section": "whole"}]

    def _split_research_paper(self, text: str) -> List[Dict[str, Any]]:
        sections = []
        current_section = ""
        last_header = "preamble"
        
        for line in text.split('\n'):
            if re.match(r'^\d+\.\s+.+', line) or re.match(r'^[A-Z][A-Z\s]+$', line):
                if current_section:
                    sections.append({"content": current_section, "section": last_header})
                last_header = line.strip()
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        if current_section:
            sections.append({"content": current_section, "section": last_header})
        
        return sections

    def _split_manual(self, text: str) -> List[Dict[str, Any]]:
        sections = []
        current_section = ""
        last_header = "start"
        
        for line in text.split('\n'):
            if re.match(r'^(Warning|Caution|Note|Step \d+|Chapter \d+)', line, re.I):
                if current_section:
                    sections.append({"content": current_section, "section": last_header})
                last_header = line.strip()
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        
        if current_section:
            sections.append({"content": current_section, "section": last_header})
        
        return sections

    def split_text(self, text: str, doc_id: str) -> List[Document]:
        doc_type = self.detect_document_type(text)
        st.session_state.doc_types[doc_id] = doc_type
        
        sections = self.extract_sections(text, doc_type)
        documents = []
        
        for i, section in enumerate(sections):
            chunks = self.default_splitter.split_text(section["content"])
            
            for j, chunk in enumerate(chunks):
                metadata = {
                    "source": doc_id,
                    "page": i,
                    "section": section["section"],
                    "doc_type": doc_type
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=metadata  # Now passing a regular dict
                ))
                
                chunk_id = f"{doc_id}_chunk_{i}_{j}"
                st.session_state.doc_metadata[chunk_id] = metadata
        
        return documents

class GroqClient:
    def __init__(self, model_name: str = "mixtral-8x7b-32768"):
        self.api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 1024) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Enhanced prompt engineering
        system_prompt = """You are an expert at answering questions based on provided context.
        - Always stay faithful to the context
        - If asked to compare between documents, analyze all relevant sources
        - For manuals, provide step-by-step instructions when appropriate
        - For research papers, maintain academic rigor
        - For policies, be precise and include relevant sections"""
        
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.3,  # Lower for more factual responses
            "max_tokens": max_tokens,
            "top_p": 0.9
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API error: {response.text}")

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def process_uploaded_files(uploaded_files):
    splitter = IntelligentSplitter()
    documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        text = extract_text_from_pdf(tmp_file_path)
        doc_id = uploaded_file.name
        doc_documents = splitter.split_text(text, doc_id)
        documents.extend(doc_documents)
        
        st.session_state.processed_docs.append(doc_id)
        os.unlink(tmp_file_path)
    
    if documents:
        st.session_state.vectorstore = FAISS.from_documents(documents, embeddings)
        st.success(f"Processed {len(uploaded_files)} PDF(s) with {len(documents)} intelligent chunks")

def get_relevant_documents(
    query: str, 
    k: int = 4, 
    filter_source: Optional[str] = None,
    filter_doc_type: Optional[str] = None,
    filter_section: Optional[str] = None
) -> List[Document]:
    if not st.session_state.vectorstore:
        return []
    
    docs = st.session_state.vectorstore.similarity_search(query, k=k*3)  # Get extra for filtering
    
    results = []
    for doc in docs:
        metadata = doc.metadata  # This is a dictionary
        
        # Apply filters - access dictionary keys, not attributes
        if filter_source and metadata.get('source') != filter_source:
            continue
        if filter_doc_type and metadata.get('doc_type') != filter_doc_type:
            continue
        if filter_section and metadata.get('section') != filter_section:
            continue
            
        results.append(doc)
        if len(results) >= k:  # Stop when we have enough filtered results
            break
    
    return results

# UI Components
st.title("🧠 Advanced PDF QA with Intelligent Chunking")
st.markdown("Upload PDFs and get answers with enhanced context understanding")

# Sidebar for advanced controls
with st.sidebar:
    st.header("Advanced Controls")
    
    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.1-8b-instant", "gemma2-9b-it"],
        index=0
    )
    
    st.subheader("Chunking Settings")
    chunk_size = st.slider("Base chunk size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200)
    
    st.subheader("Filtering Options")
    show_advanced_filters = st.toggle("Enable advanced filters", False)
    
    st.markdown("---")
    if st.button("Clear All Documents"):
        st.session_state.vectorstore = None
        st.session_state.doc_metadata = {}
        st.session_state.processed_docs = []
        st.session_state.doc_types = {}
        st.rerun()

# Main content
upload_col, query_col = st.columns([1, 2])

with upload_col:
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files and st.button("Process PDFs"):
        with st.spinner("Analyzing and chunking documents..."):
            process_uploaded_files(uploaded_files)
    
    if st.session_state.processed_docs:
        st.subheader("Processed Documents")
        for doc in st.session_state.processed_docs:
            doc_type = st.session_state.doc_types.get(doc, "unknown")
            st.markdown(f"- {doc} ({doc_type.replace('_', ' ').title()})")

with query_col:
    st.subheader("Ask Questions")
    
    if st.session_state.processed_docs:
        question = st.text_area(
            "Your question", 
            placeholder="Ask something about the documents...",
            height=100
        )
        
        if show_advanced_filters:
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_source = st.selectbox(
                    "Filter by document",
                    ["All documents"] + st.session_state.processed_docs
                )
            with col2:
                doc_types = list(set(st.session_state.doc_types.values()))
                filter_doc_type = st.selectbox(
                    "Filter by document type",
                    ["All types"] + doc_types
                )
            with col3:
                # Get unique sections from metadata
                all_sections = set()
                for meta in st.session_state.doc_metadata.values():
                    if meta.section:
                        all_sections.add(meta.section)
                filter_section = st.selectbox(
                    "Filter by section",
                    ["All sections"] + sorted(list(all_sections))
                )
        else:
            filter_source = st.selectbox(
                "Filter by document",
                ["All documents"] + st.session_state.processed_docs
            )
            filter_doc_type = "All types"
            filter_section = "All sections"
        
        if question:
            with st.spinner("Analyzing documents and generating response..."):
                try:
                    # Apply filters
                    source_filter = None if filter_source == "All documents" else filter_source
                    type_filter = None if filter_doc_type == "All types" else filter_doc_type
                    section_filter = None if filter_section == "All sections" else filter_section
                    
                    # Get relevant documents with filtering
                    relevant_docs = get_relevant_documents(
                        question,
                        k=6,  # Get more context for better answers
                        filter_source=source_filter,
                        filter_doc_type=type_filter,
                        filter_section=section_filter
                    )
                    
                    # Prepare enhanced context
                    
                    context_parts = []
                    for doc in relevant_docs:
                        meta = doc.metadata  # This is a dictionary
                        context_parts.append(
                            f"=== Document: {meta['source']} ===\n"
                            f"Type: {meta.get('doc_type', 'unknown')}\n"
                            f"Section: {meta.get('section', 'unknown')}\n"
                            f"Content:\n{doc.page_content}\n"
    )
                    context = "\n\n".join(context_parts)
                    
                    # Initialize Groq client
                    groq_client = GroqClient(model_name=model_choice)
                    
                    # Generate response
                    answer = groq_client.generate_response(question, context)
                    
                    # Display results
                    st.subheader("Answer")
                    st.markdown(answer)
                    
                    with st.expander("View context sources"):
                        for doc in relevant_docs:
                            meta = doc.metadata
                            st.markdown(f"**{meta.source}** (Type: {meta.doc_type}, Section: {meta.section})")
                            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    else:
        st.info("Upload and process PDF documents to enable question answering")

# Create secrets file if not exists
if not os.path.exists(".streamlit/secrets.toml"):
    os.makedirs(".streamlit", exist_ok=True)
    with open(".streamlit/secrets.toml", "w") as f:
        f.write('GROQ_API_KEY = "your_api_key_here"\n')
    st.sidebar.info("Please add your Groq API key to .streamlit/secrets.toml")

    #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