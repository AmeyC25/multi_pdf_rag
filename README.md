# multi_pdf_rag
# PDF Question Answering System with Groq API

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-00FF00?style=for-the-badge&logo=groq&logoColor=black)

A powerful system for extracting answers from PDF documents using Groq's ultra-fast LLM inference.

## Features

- **Intelligent PDF Processing**: Automatic document type detection (research papers, manuals, policies)
- **Advanced Chunking**: Semantic section splitting with context preservation
- **Multi-level Filtering**: Filter answers by document, type, or section
- **Fast Responses**: Powered by Groq's LPU inference engine




## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pdf-qa-groq.git
   cd pdf-qa-groq

2. **Set up virtual environment**:
    ```  python -m venv venv
        source venv/bin/activate  # Linux/Mac
        venv\Scripts\activate 

3. **Install dependencies**:
```pip install -r requirements.txt


## Configuration
Get your Groq API key from Groq Console

Set up environment:

bash
echo "GROQ_API_KEY=your_api_key_here" > .streamlit/secrets.toml

And launch using streamlit run !