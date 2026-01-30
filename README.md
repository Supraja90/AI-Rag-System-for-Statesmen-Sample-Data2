## Overview
This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** application deployed on a **High-Performance Computing (HPC)** cluster using **Slurm**. The system ingests large document collections (PDFs and images), performs OCR when required, generates embeddings, and retrieves relevant content using a **hybrid dense–sparse retrieval pipeline with reranking**.

An interactive **Streamlit web interface** is launched on a compute node and securely accessed via **SSH port forwarding**, enabling users to query documents using an **LLM served through an external Ollama API**.


## Steps to Execute
pip install -r requirements.txt

python3 ingest.py

Connect to API 

python3 rag_demo.py

Once it is working properly. Stop it and execute streamlit.

