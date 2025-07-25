from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    DATABASE_CONNECTIONS_TO_SCAN = [
    {
        "source_name": "demo_testing_db",
        "host": "localhost",
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD"),
        "database": "testing",
        "port": int(os.getenv("DB_PORT", "3306")), #Enter the port number as needed
    }
]


    MPI_FILE_PATH: str = os.getenv("MPI_FILE_PATH")
    LOG_FILE: str = os.getenv("LOG_FILE", "system.log")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")