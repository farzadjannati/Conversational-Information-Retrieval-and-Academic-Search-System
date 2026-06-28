<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:0284c7,100:38bdf8&height=220&section=header&text=Conversational%20Information%20Retrieval&fontSize=34&fontColor=ffffff&fontAlignY=50&animation=fadeIn" />
</div>

---

# LLM-Powered Conversational Search and Recommendation System

This project introduces a state-of-the-art Conversational Information Retrieval (CIR) framework that leverages Large Language Models (LLMs) to interact with users, clarify ambiguous information needs, rewrite queries based on conversation history, and retrieve highly relevant academic papers using dense vector representations and an Agentic AI workflow.

<div align="left">
  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
  [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
  [![LangChain](https://img.shields.io/badge/LangChain-Integration-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://www.langchain.com/)
  [![OpenAI](https://img.shields.io/badge/OpenAI-LLM_Agents-10A37F?style=flat&logo=openai&logoColor=white)](https://openai.com/)
  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Sentence_Transformers-F5A623?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/)
  [![LanceDB](https://img.shields.io/badge/LanceDB-Vector_Database-D84B2A?style=flat)](https://lancedb.github.io/lancedb/)
  [![PyMuPDF](https://img.shields.io/badge/PyMuPDF-Document_Parsing-6D28D9?style=flat)](#)
  [![AI Evaluation](https://img.shields.io/badge/Eval-LLM_as_a_Judge-0284C7?style=flat)](#)
  [![Domain](https://img.shields.io/badge/Domain-Information_Retrieval-059669?style=flat)](#)
  [![License](https://img.shields.io/badge/License-MIT-4B5563?style=flat)](https://opensource.org/licenses/MIT)
</div>

## Abstract
Traditional keyword-based and baseline semantic search systems often fail to satisfy complex user intents due to the vocabulary gap and polysemy. This project implements an advanced Conversational Search System that replaces standard search with an interactive AI assistant. By generating clarifying questions, simulating user responses, and employing context-aware query rewriting, the system dramatically improves retrieval precision (P@1 jumped from 50% to 95%). Furthermore, it incorporates an automated "LLM-as-a-Judge" evaluation pipeline utilizing Fleiss' Kappa to ensure the quality, naturalness, and usefulness of the generated dialogues.

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Methodology and Modules](#methodology-and-modules)
   - 4.1. [Document Vectorization](#document-vectorization)
   - 4.2. [User Simulation Strategy](#user-simulation-strategy)
   - 4.3. [Dialogue-Aware Query Rewriting](#dialogue-aware-query-rewriting)
5. [Evaluation and Metrics](#evaluation-and-metrics)
   - 5.1. [Retrieval Performance Comparison](#retrieval-performance-comparison)
   - 5.2. [LLM as a Judge Framework](#llm-as-a-judge-framework)
6. [Tools and Technologies](#tools-and-technologies)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Environment Variables](#environment-variables)
10. [License](#license)
11. [Author](#author)
12. [Support](#support)

# Overview
This system is designed to simulate a real-world interaction between a researcher with a specific (but initially vague) information need and an intelligent search assistant. Rather than retrieving documents based on a single, short initial query, the LLM-powered assistant actively clarifies the user's intent over multiple turns. The synthesized context is then transformed into a highly dense canonical query for semantic retrieval against a vector database of scientific papers.

---

# Key Features
* **Semantic Vector Search:** Powered by LanceDB and `BAAI/bge-small-en-v1.5` embeddings.
* **Intelligent Document Parsing:** Extracted dense semantic signals from scientific PDFs using PyMuPDF4LLM.
* **LLM-Based User Simulation:** Autonomous response generation based on hidden user personas and information needs.
* **Conversational Agent:** Multi-turn dialogue orchestration to clarify ambiguous search queries.
* **Automated Query Rewriting:** Transforming raw conversation history into robust, context-rich retrieval queries.
* **LLM-as-a-Judge Evaluation:** Quality assessment of the system's generated questions using Human vs. AI inter-rater reliability (Fleiss' Kappa).

---

# System Architecture
The pipeline follows a modular architecture separating the simulation layer, the conversational orchestration, and the retrieval engine. 

```mermaid
flowchart TB

subgraph User Simulation Layer
    U[Hidden Information Need]
    US[LLM User Simulator]
    U --> US
end

subgraph Agent Layer
    A[Conversational Search Assistant]
    QR[Query Rewriter Agent]
end

subgraph Retrieval Layer
    EM[SentenceTransformer Embeddings]
    VD[LanceDB Vector Store]
end

subgraph Knowledge Base
    PDF[Scientific Papers PDFs]
    LP[PyMuPDF Parser]
end

subgraph Evaluation
    Eval[P@1 Metric Calculation]
    Judge[LLM as a Judge]
end

US <-->|Multi-turn Dialogue| A
A -- Conversation History --> QR
QR -- Rewritten Query --> EM

PDF --> LP
LP -- Parsed Text --> VD
EM --> VD

VD -- Top Retrieved Paper --> Eval
A -- Clarifying Questions --> Judge
