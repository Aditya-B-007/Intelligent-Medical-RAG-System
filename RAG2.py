from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mysql.connector.pooling import MySQLConnectionPool
from mysql.connector import Error
from typing import Optional, List, Dict, Any
import json
import concurrent.futures
import logging
import time
from Config import AppConfig
from generate_config import setup_mpi_config

logger = logging.getLogger("RAG")

class RAG:
    def __init__(self):
        logger.info("Initializing RAG pipeline...")
        db_connections = AppConfig.DATABASE_CONNECTIONS_TO_SCAN
        mpi_db_file_path = AppConfig.MPI_FILE_PATH
        full_config = setup_mpi_config(db_connections, mpi_db_file_path)

        if not full_config:
            self.mpi_data = []
            self.db_config_map = {}
            self.db_pools = {}
            logger.warning("RAG init: No config available, using empty state.")
            return

        self.mpi_data = self._load_mpi_data(full_config.get("mpi_file_path"))
        self.db_connections_config = full_config.get("data_sources", [])
        self.db_config_map = {conn['source_name']: conn for conn in self.db_connections_config}
        self.db_pools: Dict[str, MySQLConnectionPool] = {}

        for conn_info in self.db_connections_config:
            source_name = conn_info.get("source_name")
            config = conn_info.get("config")
            if not source_name or not config:
                logger.warning(f"Skipping pool creation for invalid config: {conn_info}")
                continue
            try:
                allowed_keys = {"host", "port", "user", "password", "database"}
                sanitized_config = {k: v for k, v in config.items() if k in allowed_keys}
                pool = MySQLConnectionPool(
                    pool_name=f"{source_name}_pool",
                    pool_size=5,
                    **sanitized_config
                )
                self.db_pools[source_name] = pool
                logger.info(f"Connection pool created for '{source_name}'.")
            except Error as e:
                logger.error(f"Failed to create pool for '{source_name}': {e}")

    def _load_mpi_data(self, file_path: Optional[str]) -> List[Dict[str, Any]]:
        if not file_path:
            logger.error("MPI file path not provided.")
            return []

        logger.info(f"Loading MPI from '{file_path}'...")
        mpi_data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        mpi_data.append(rec)
                    except json.JSONDecodeError:
                        logger.warning(f"Malformed MPI line: {line.strip()}")
        except FileNotFoundError:
            logger.error(f"MPI file not found: {file_path}")
        logger.info(f"{len(mpi_data)} MPI records loaded.")
        return mpi_data

    def document_loader(self, link: str):
        try:
            loader = PyPDFLoader(link)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(data)
            return texts
        except Exception as e:
            logger.error(f"PDF load failed: {e}")
            return []
    def _get_patient_identity_mappings(self, patient_id: Optional[str], full_name: Optional[str]) -> Optional[Dict[str, str]]:
        if not patient_id and not full_name:
            logger.warning("No patient ID or full name provided for MPI mapping.")
            return None

        mappings = {}
        for record in self.mpi_data:
            source = record.get("source_name")
            schema = record.get("schema_mapping", {})
            table = schema.get("table")
            columns = schema.get("columns", {})
            
            patient_id_col = columns.get("patient_id")
            patient_name_col = columns.get("patient_name")

            if not all([source, table, patient_id_col, patient_name_col]):
                continue

            pool = self.db_pools.get(source)
            if not pool:
                continue
            try:
                conn = pool.get_connection()
                cursor = conn.cursor(dictionary=True)

                # Build the query dynamically
                query_parts = []
                params = []
                
                if patient_id:
                    query_parts.append(f"`{patient_id_col}` = %s")
                    params.append(patient_id)

                if full_name:
                    query_parts.append(f"`{patient_name_col}` = %s")
                    params.append(full_name)

                if not query_parts:
                    continue

                where_clause = " OR ".join(query_parts)
                query = f"SELECT `{patient_id_col}` FROM `{table}` WHERE {where_clause} LIMIT 1"
                
                cursor.execute(query, tuple(params))
                result = cursor.fetchone()

                if result:
                    # Use the actual patient ID column name from the schema
                    mappings[source] = result[patient_id_col]

            except Exception as e:
                logger.warning(f"Failed MPI match in {source}: {e}")
            finally:
                if 'conn' in locals() and conn.is_connected():
                    cursor.close()
                    conn.close()

        return mappings if mappings else None

    def _query_single_database(self, hospital_name: str, patient_id: str) -> Optional[str]:
        pool = self.db_pools.get(hospital_name)
        conn_info = self.db_config_map.get(hospital_name)

        if not pool or not conn_info:
            return None

        schema = conn_info.get("schema_mapping")
        full_schema = conn_info.get("full_schema", {})
        if not schema or not full_schema:
            return None

        table = schema['table']
        col_map = schema['columns']
        patient_id_col = col_map.get('patient_id')

        if not patient_id_col:
            return None

        records = []

        try:
            conn = pool.get_connection()
            cursor = conn.cursor(dictionary=True)

            # Primary table query
            cursor.execute(f"SELECT * FROM `{table}` WHERE `{patient_id_col}` = %s", (patient_id,))
            row = cursor.fetchone()
            if row:
                body = "\n".join(
                    f"- {col.replace('_', ' ').title()}: {val}" for col, val in row.items()
                )
                records.append(f"--- Primary Record ({hospital_name}) ---\n{body}")

            # Query all other tables with matching patient_id column
            for table_name, columns in full_schema.items():
                if table_name == table:
                    continue

                column_names = [col['Field'] for col in columns]
                if patient_id_col not in column_names:
                    continue

                try:
                    cursor.execute(
                        f"SELECT * FROM `{table_name}` WHERE `{patient_id_col}` = %s",
                        (patient_id,)
                    )
                    results = cursor.fetchall()
                    if not results:
                        continue

                    section = f"--- {table_name.title()} Records ---\n"
                    for result in results:
                        for key, value in result.items():
                            section += f"- {key.replace('_', ' ').title()}: {value}\n"
                        section += "\n"
                    records.append(section)
                except Exception as e:
                    logger.warning(f"Failed to query table '{table_name}' in {hospital_name}: {e}")

            return "\n\n".join(records)

        except Error as e:
            logger.error(f"Query error on {hospital_name}: {e}")
            return None
        finally:
            try: cursor.close()
            except: pass
            try: conn.close()
            except: pass

    def fetch_patient_data(self, patient_id: Optional[str] = None, full_name: Optional[str] = None) -> Optional[str]:
        mappings = self._get_patient_identity_mappings(patient_id, full_name)
        if not mappings:
            logger.warning("Moving...")
            return None

        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(mappings))) as executor:
            future_to_db = {
                executor.submit(self._query_single_database, db_name, pid): db_name
                for db_name, pid in mappings.items()
            }
            for future in concurrent.futures.as_completed(future_to_db):
                try:
                    result = future.result()
                    if result:
                        records.append(result)
                except Exception as e:
                    logger.error(f"Thread failure for {future_to_db[future]}: {e}")

        if not records:
            return None

        return f"Aggregated Patient Records:\n{'=' * 40}\n\n" + "\n\n".join(records)
