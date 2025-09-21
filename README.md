# Market Intelligence & Pricing Agent – Hackathon Approach Document

**Prepared by cvk03**

---

## Problem Statement
Insurance companies often face significant challenges in maintaining their competitive edge due to the constantly changing landscape of market rates, claims trends, and regulatory requirements. Traditional methods of manually benchmarking competitor rates, analysing claims data, and monitoring regulatory filings are time-consuming, error-prone, and can result in missed business opportunities or incorrect pricing strategies.

---

## Solution Approach
The Market Intelligence & Pricing Agent is an AI-powered virtual analyst designed to automate and enhance insurance market intelligence and pricing workflows. The solution leverages advanced language models, vector search, and workflow orchestration to deliver actionable insights and recommendations.

### Key Components & Tools
| Tool                | Purpose                                              |
|---------------------|------------------------------------------------------|
| Semantic Kernel     | Agent orchestration and workflow management          |
| Google Gemini AI    | Summarization and analysis using large language models|
| FAISS               | Fast vector search for efficient document retrieval  |
| Streamlit           | Simple user interface for interaction                |
| Authentication      | Ensures basic security for the user interface        |

### Solution Workflow
1. **Data Collection:** Gathers insurance rates, claims, and regulatory filings from sample CSV/JSON files or public datasets.
2. **Data Ingestion & Processing:** Loads, cleans, and chunks data for analysis.
3. **Vector Indexing:** Converts processed documents into vector embeddings and stores them in FAISS for rapid, similarity-based retrieval.
4. **Query Handling:** On user query, retrieves relevant data chunks from the FAISS index.
5. **Orchestration:** Semantic Kernel manages the workflow, passing data to the language model for analysis.
6. **AI Analysis:** Google Gemini AI interprets and summarizes the data, providing concise, actionable insights.
7. **User Delivery:** Results are presented to the user via a Streamlit web interface.

---

## Benefits
- Enables rapid and reliable benchmarking of rates.
- Provides predictive analysis of claim trends.
- Automates the monitoring of regulatory and competitor activity.
- Delivers timely and actionable insights for pricing decisions.
- Reduces the manual effort involved in these processes to a significant extent.

---

## Features
- **Data Retrieval:** Collects up-to-date information on insurance rates, claims, and regulatory filings.
- **AI-driven Analysis & Summarization:** Uses LLMs to interpret and summarize complex data.
- **Actionable Recommendations:** Provides clear, practical guidance for pricing strategies.
- **Automated Monitoring:** Tracks market changes and updates with minimal manual intervention.
- **Interactive UI:** User-friendly Streamlit app for querying and visualizing insights.
- **Secure Access:** Basic authentication for safe usage.

---

## Example End-to-End Flow
1. User submits a query (e.g., "Benchmark auto insurance rates in Texas").
2. The agent retrieves relevant data from the FAISS index.
3. The language model summarizes results (average rates, top providers, trends).
4. Insights are delivered to the user via the web interface.

---

## Testing & Success Metrics
- **Retrieval Accuracy:** Percentage of relevant documents returned per query.
- **Summarization Quality:** Alignment of AI-generated summaries with expected outputs.
- **Response Time:** Time taken from query submission to result delivery.

---

## Project Structure
- `app.py`: Streamlit web application for user interaction.
- `src/agent/`: Core agent logic, skills, and Gemini service integration.
- `src/utils/`: Data loading, preparation, and vector store utilities.
- `data/`: Sample data files (rates, claims, regulatory filings).
- `vector_store/`: FAISS index and document storage.

---

## Getting Started
1. Clone the repository and install requirements from `requirements.txt`.
2. Add your Google API key to a `.env` file in the project root.
3. Run `python setup_agent.py` to generate sample data and build the vector store.
4. Launch the app with `streamlit run app.py`.

---

## Contact
For questions or support, contact cvk03
