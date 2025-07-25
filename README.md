# Intelligent Medical RAG System

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)

The Intelligent Medical RAG (Retrieval-Augmented Generation) System is a sophisticated application designed to securely access, interpret, and analyze patient data from disparate MySQL databases. It dynamically discovers database schemas, integrates multi-modal user inputs (text, images, and documents), and leverages a powerful generative AI model to deliver structured, insightful medical analysis.

## Key Features

  * **Dynamic Database Schema Discovery**: Automatically connects to specified MySQL databases, introspects their schemas, and uses an LLM to identify key patient-related tables and columns without manual configuration.
  * [cite\_start]**Multi-modal Input**: Accepts a combination of inputs for comprehensive analysis, including user-provided text, medical images (via local file path or URL), and medical documents in PDF format[cite: 2].
  * **Unified Patient Record Retrieval**: Fetches and aggregates patient records from multiple, structurally different databases. It resolves patient identities across these sources using a patient ID or full name and consolidates all related information.
  * [cite\_start]**AI-Powered Medical Analysis**: Utilizes Google's Gemini model to analyze the aggregated context (database records, user text, images, documents) and generate a structured JSON report containing a potential diagnosis, medication advice, specialist recommendation, and recovery estimate[cite: 2].
  * [cite\_start]**Scalable and Maintainable Architecture**: The system is designed with a modular architecture that separates configuration, data retrieval, and application logic, making it easy to maintain and scale to new data sources[cite: 2].

## System Architecture

The system is composed of several key modules that work in concert:

1.  **`Config.py`**: A centralized configuration module for managing all settings, including database credentials and API keys, loaded securely from a `.env` file.
2.  **`generate_config.py`**: A powerful utility that performs schema discovery. It connects to databases, extracts their schemas, uses an LLM to map patient data fields, and generates the `mp_data.jsonl` file.
3.  [cite\_start]**`mp_data.jsonl`**: The Master Patient Index (MPI) file[cite: 1]. [cite\_start]This auto-generated JSONL file acts as a "map" for the RAG system, storing the schema information and column mappings for each data source[cite: 1].
4.  **`RAG2.py`**: The core data retrieval and aggregation engine (the "R" in RAG). It reads the MPI map, manages database connection pooling, and fetches all relevant patient data from the configured sources. It can also process uploaded PDF documents.
5.  [cite\_start]**`google2.0.py`**: The main application entry point and orchestration layer (the "G" in RAG)[cite: 2]. [cite\_start]It handles all user interaction, coordinates with the RAG engine to fetch data, constructs the final prompt for the AI, and presents the structured analysis to the user[cite: 2].

## Workflow

The operational flow of the application is as follows:

1.  **Initialization**: On the first run, the `RAG` class in `RAG2.py` invokes the `setup_mpi_config` function from `generate_config.py`.
2.  **Schema Discovery**: The system connects to all databases defined in `Config.py`, analyzes their schemas, and generates the `mp_data.jsonl` file containing the necessary mappings.
3.  [cite\_start]**User Input**: The user runs `google2.0.py` and is prompted to provide a patient ID/name, a description of their medical concern, and optional image and document files[cite: 2].
4.  **Data Retrieval**: The `RAG2.py` engine uses the MPI map to resolve the patient's identity across the databases. It then fetches all historical and related records for that patient concurrently from all sources.
5.  [cite\_start]**Context Aggregation**: The retrieved database records are combined with the user's text, image data, and document text to form a single, comprehensive context[cite: 2].
6.  [cite\_start]**AI Analysis**: This aggregated context is sent to the Gemini model with a system prompt instructing it to perform a medical analysis and return a structured JSON response[cite: 2].
7.  [cite\_start]**Result Presentation**: The `google2.0.py` script parses the JSON response from the AI and displays a formatted report to the user[cite: 2].

## Setup and Installation

### Prerequisites

  * Python 3.x
  * An active MySQL server

### 1\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2\. Install Dependencies

Create a `requirements.txt` file with the following content:

```
mysql-connector-python
python-dotenv
google-generativeai
langchain
PyPDFLoader
pillow
requests
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 3\. Configure Environment

Create a `.env` file in the root of the project directory and add your credentials:

```env
DB_USER="your_db_user"
DB_PASSWORD="your_db_password"
DB_PORT="3306"
GEMINI_API_KEY="your_google_gemini_api_key"
```

### 4\. Configure Data Sources

Open `Config.py` and update the `DATABASE_CONNECTIONS_TO_SCAN` list with the connection details for all the MySQL databases you wish to include in the system.

```python
# In Config.py
class AppConfig:
    DATABASE_CONNECTIONS_TO_SCAN = [
        {
            "source_name": "your_source_name",
            "host": "localhost",
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "database": "your_database_name",
            "port": int(os.getenv("DB_PORT", 3306)),
        },
        # Add other database connections here
    ]
    # ...
```

## Usage

Run the main application script from your terminal:

```bash
python google2.0.py
```

The application will start and guide you through a series of interactive prompts to input patient information and describe the medical concern. The system will then process the information and display a complete analysis report.

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
