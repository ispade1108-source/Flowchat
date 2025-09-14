# """
# Database Management Module
# Handles SQLite for structured data and ChromaDB for vector storage
# """

# import sqlite3
# import pandas as pd
# from pathlib import Path
# import chromadb

# from chromadb.config import Settings
# import logging
# from typing import List, Dict, Optional

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class DatabaseManager:
#     """Manages both SQL and vector databases for ARGO data."""
    
#     def __init__(self, db_path: str = "argo_data.db", chroma_path: str = "./chroma"):
#         self.db_path = db_path
#         self.chroma_path = chroma_path
#         self.conn = None
#         self.chroma_client = None
#         self.collection = None
#         self._init_databases()
    
#     def _init_databases(self):
#         """Initialize SQL and vector databases."""
#         # Initialize SQLite
#         self.conn = sqlite3.connect(self.db_path)
#         self._create_tables()
        
#         # Initialize ChromaDB
#         self.chroma_client = chromadb.PersistentClient(
#             path=self.chroma_path,
#             settings=Settings(anonymized_telemetry=False)
#         )
        
#         # Create or get collection
#         try:
#             self.collection = self.chroma_client.create_collection(
#                 name="argo_profiles",
#                 metadata={"hnsw:space": "cosine"}
#             )
#         except:
#             self.collection = self.chroma_client.get_collection("argo_profiles")
        
#         logger.info("Database initialized successfully")
    
#     def _create_tables(self):
#         """Create necessary SQL tables."""
#         cursor = self.conn.cursor()
        
#         # Create profiles table
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS profiles (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 platform_number TEXT,
#                 profile_number INTEGER,
#                 latitude REAL,
#                 longitude REAL,
#                 time TIMESTAMP,
#                 pressure REAL,
#                 depth REAL,
#                 temperature REAL,
#                 salinity REAL
#             )
#         """)
        
#         # Create metadata table
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS metadata (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 platform_number TEXT UNIQUE,
#                 num_profiles INTEGER,
#                 num_measurements INTEGER,
#                 lat_min REAL,
#                 lat_max REAL,
#                 lon_min REAL,
#                 lon_max REAL,
#                 time_start TIMESTAMP,
#                 time_end TIMESTAMP,
#                 depth_min REAL,
#                 depth_max REAL
#             )
#         """)
        
#         # Create indices for better query performance
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_platform ON profiles(platform_number)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON profiles(time)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON profiles(latitude, longitude)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_depth ON profiles(depth)")
        
#         self.conn.commit()
#         logger.info("SQL tables created successfully")
    
#     def insert_profile_data(self, df: pd.DataFrame, metadata: Dict):
#         """Insert profile data into SQL database."""
#         if df.empty:
#             logger.warning("Empty dataframe, skipping insertion")
#             return
        
#         # Insert profile data
#         df.to_sql('profiles', self.conn, if_exists='append', index=False)
        
#         # Insert or update metadata
#         cursor = self.conn.cursor()
#         cursor.execute("""
#             INSERT OR REPLACE INTO metadata 
#             (platform_number, num_profiles, num_measurements, 
#              lat_min, lat_max, lon_min, lon_max, 
#              time_start, time_end, depth_min, depth_max)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """, (
#             metadata['platform_number'],
#             metadata['num_profiles'],
#             metadata['num_measurements'],
#             metadata['lat_range'][0], metadata['lat_range'][1],
#             metadata['lon_range'][0], metadata['lon_range'][1],
#             metadata['time_range'][0], metadata['time_range'][1],
#             metadata['depth_range'][0], metadata['depth_range'][1]
#         ))
        
#         self.conn.commit()
#         logger.info(f"Inserted {len(df)} records for platform {metadata['platform_number']}")
    
#     def add_to_vector_db(self, doc_id: str, text: str, metadata: Dict):
#         """Add document to ChromaDB vector database."""
#         try:
#             self.collection.add(
#                 documents=[text],
#                 metadatas=[metadata],
#                 ids=[doc_id]
#             )
#             logger.info(f"Added document {doc_id} to vector database")
#         except Exception as e:
#             logger.error(f"Error adding to vector database: {e}")
    
#     def query_vector_db(self, query_text: str, n_results: int = 5) -> List[Dict]:
#         """Query the vector database for similar documents."""
#         try:
#             results = self.collection.query(
#                 query_texts=[query_text],
#                 n_results=n_results
#             )
            
#             return [
#                 {
#                     'id': results['ids'][0][i],
#                     'document': results['documents'][0][i],
#                     'metadata': results['metadatas'][0][i],
#                     'distance': results['distances'][0][i] if 'distances' in results else None
#                 }
#                 for i in range(len(results['ids'][0]))
#             ]
#         except Exception as e:
#             logger.error(f"Error querying vector database: {e}")
#             return []
    
#     def execute_sql(self, query: str) -> pd.DataFrame:
#         """Execute SQL query and return results as DataFrame."""
#         try:
#             df = pd.read_sql_query(query, self.conn)
#             logger.info(f"SQL query executed successfully, returned {len(df)} rows")
#             return df
#         except Exception as e:
#             logger.error(f"SQL query error: {e}")
#             return pd.DataFrame()
    
#     def get_platform_list(self) -> List[str]:
#         """Get list of all platform numbers in database."""
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT DISTINCT platform_number FROM profiles")
#         return [row[0] for row in cursor.fetchall()]
    
#     def get_data_summary(self) -> Dict:
#         """Get summary statistics of the database."""
#         cursor = self.conn.cursor()
        
#         # Count total records
#         cursor.execute("SELECT COUNT(*) FROM profiles")
#         total_records = cursor.fetchone()[0]
        
#         # Count platforms
#         cursor.execute("SELECT COUNT(DISTINCT platform_number) FROM profiles")
#         total_platforms = cursor.fetchone()[0]
        
#         # Get date range
#         cursor.execute("SELECT MIN(time), MAX(time) FROM profiles")
#         time_range = cursor.fetchone()
        
#         # Get geographic bounds
#         cursor.execute("""
#             SELECT MIN(latitude), MAX(latitude), 
#                    MIN(longitude), MAX(longitude) 
#             FROM profiles
#         """)
#         bounds = cursor.fetchone()
        
#         return {
#             'total_records': total_records,
#             'total_platforms': total_platforms,
#             'time_range': time_range,
#             'geographic_bounds': {
#                 'lat_min': bounds[0] if bounds else None,
#                 'lat_max': bounds[1] if bounds else None,
#                 'lon_min': bounds[2] if bounds else None,
#                 'lon_max': bounds[3] if bounds else None
#             }
#         }
    
#     def close(self):
#         """Close database connections."""
#         if self.conn:
#             self.conn.close()
#         logger.info("Database connections closed")




































































# """
# Database Management Module
# Handles SQLite for structured data and ChromaDB for vector storage
# """

# import os
# import sqlite3
# import pandas as pd
# from pathlib import Path
# import chromadb
# from chromadb.config import Settings
# import logging
# from typing import List, Dict, Optional

# # Disable Chroma telemetry globally
# os.environ["CHROMA_TELEMETRY_ENABLED"] = "0"

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class DatabaseManager:
#     """Manages both SQL and vector databases for ARGO data."""
    
#     def __init__(self, db_path: str = "argo_data.db", chroma_path: str = "./chroma"):
#         self.db_path = db_path
#         self.chroma_path = chroma_path
#         self.conn = None
#         self.chroma_client = None
#         self.collection = None
#         self._init_databases()
    
#     def _init_databases(self):
#         """Initialize SQL and vector databases."""
#         # Initialize SQLite (thread-safe)
#         self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
#         self._create_tables()
        
#         # Initialize ChromaDB (telemetry disabled via Settings)
#         self.chroma_client = chromadb.PersistentClient(
#             path=self.chroma_path,
#             settings=Settings(anonymized_telemetry=False)
#         )
        
#         # Create or get collection
#         try:
#             self.collection = self.chroma_client.create_collection(
#                 name="argo_profiles",
#                 metadata={"hnsw:space": "cosine"}
#             )
#         except Exception:
#             self.collection = self.chroma_client.get_collection("argo_profiles")
        
#         logger.info("Database initialized successfully")
    
#     def _create_tables(self):
#         """Create necessary SQL tables."""
#         cursor = self.conn.cursor()
        
#         # Create profiles table
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS profiles (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 platform_number TEXT,
#                 profile_number INTEGER,
#                 latitude REAL,
#                 longitude REAL,
#                 time TIMESTAMP,
#                 pressure REAL,
#                 depth REAL,
#                 temperature REAL,
#                 salinity REAL
#             )
#         """)
        
#         # Create metadata table
#         cursor.execute("""
#             CREATE TABLE IF NOT EXISTS metadata (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 platform_number TEXT UNIQUE,
#                 num_profiles INTEGER,
#                 num_measurements INTEGER,
#                 lat_min REAL,
#                 lat_max REAL,
#                 lon_min REAL,
#                 lon_max REAL,
#                 time_start TIMESTAMP,
#                 time_end TIMESTAMP,
#                 depth_min REAL,
#                 depth_max REAL
#             )
#         """)
        
#         # Create indices for better query performance
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_platform ON profiles(platform_number)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON profiles(time)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON profiles(latitude, longitude)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_depth ON profiles(depth)")
        
#         self.conn.commit()
#         logger.info("SQL tables created successfully")
    
#     def insert_profile_data(self, df: pd.DataFrame, metadata: Dict):
#         """Insert profile data into SQL database."""
#         if df.empty:
#             logger.warning("Empty dataframe, skipping insertion")
#             return
        
#         # Insert profile data
#         df.to_sql('profiles', self.conn, if_exists='append', index=False)
        
#         # Insert or update metadata
#         cursor = self.conn.cursor()
#         cursor.execute("""
#             INSERT OR REPLACE INTO metadata 
#             (platform_number, num_profiles, num_measurements, 
#              lat_min, lat_max, lon_min, lon_max, 
#              time_start, time_end, depth_min, depth_max)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """, (
#             metadata['platform_number'],
#             metadata['num_profiles'],
#             metadata['num_measurements'],
#             metadata['lat_range'][0], metadata['lat_range'][1],
#             metadata['lon_range'][0], metadata['lon_range'][1],
#             metadata['time_range'][0], metadata['time_range'][1],
#             metadata['depth_range'][0], metadata['depth_range'][1]
#         ))
        
#         self.conn.commit()
#         logger.info(f"Inserted {len(df)} records for platform {metadata['platform_number']}")
    
#     def add_to_vector_db(self, doc_id: str, text: str, metadata: Dict):
#         """Add document to ChromaDB vector database."""
#         try:
#             self.collection.add(
#                 documents=[text],
#                 metadatas=[metadata],
#                 ids=[doc_id]
#             )
#             logger.info(f"Added document {doc_id} to vector database")
#         except Exception as e:
#             logger.error(f"Error adding to vector database: {e}")
    
#     def query_vector_db(self, query_text: str, n_results: int = 5) -> List[Dict]:
#         """Query the vector database for similar documents."""
#         try:
#             results = self.collection.query(
#                 query_texts=[query_text],
#                 n_results=n_results
#             )
            
#             return [
#                 {
#                     'id': results['ids'][0][i],
#                     'document': results['documents'][0][i],
#                     'metadata': results['metadatas'][0][i],
#                     'distance': results['distances'][0][i] if 'distances' in results else None
#                 }
#                 for i in range(len(results['ids'][0]))
#             ]
#         except Exception as e:
#             logger.error(f"Error querying vector database: {e}")
#             return []
    
#     def execute_sql(self, query: str) -> pd.DataFrame:
#         """Execute SQL query and return results as DataFrame."""
#         try:
#             df = pd.read_sql_query(query, self.conn)
#             logger.info(f"SQL query executed successfully, returned {len(df)} rows")
#             return df
#         except Exception as e:
#             logger.error(f"SQL query error: {e}")
#             return pd.DataFrame()
    
#     def get_platform_list(self) -> List[str]:
#         """Get list of all platform numbers in database."""
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT DISTINCT platform_number FROM profiles")
#         return [row[0] for row in cursor.fetchall()]
    
#     def get_data_summary(self) -> Dict:
#         """Get summary statistics of the database."""
#         cursor = self.conn.cursor()
        
#         # Count total records
#         cursor.execute("SELECT COUNT(*) FROM profiles")
#         total_records = cursor.fetchone()[0]
        
#         # Count platforms
#         cursor.execute("SELECT COUNT(DISTINCT platform_number) FROM profiles")
#         total_platforms = cursor.fetchone()[0]
        
#         # Get date range
#         cursor.execute("SELECT MIN(time), MAX(time) FROM profiles")
#         time_range = cursor.fetchone()
        
#         # Get geographic bounds
#         cursor.execute("""
#             SELECT MIN(latitude), MAX(latitude), 
#                    MIN(longitude), MAX(longitude) 
#             FROM profiles
#         """)
#         bounds = cursor.fetchone()
        
#         return {
#             'total_records': total_records,
#             'total_platforms': total_platforms,
#             'time_range': time_range,
#             'geographic_bounds': {
#                 'lat_min': bounds[0] if bounds else None,
#                 'lat_max': bounds[1] if bounds else None,
#                 'lon_min': bounds[2] if bounds else None,
#                 'lon_max': bounds[3] if bounds else None
#             }
#         }
    
#     def close(self):
#         """Close database connections."""
#         if self.conn:
#             self.conn.close()
#         logger.info("Database connections closed")








































































"""
Database Management Module
Handles SQLite for structured data and ChromaDB for vector storage
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
import logging
from typing import List, Dict

# Disable Chroma telemetry globally
os.environ["CHROMA_TELEMETRY_ENABLED"] = "0"

# Ensure data folder exists
Path("data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages both SQL and vector databases for ARGO data."""
    
    def __init__(self, db_path: str = "data/argo_data.db", chroma_path: str = "data/chroma"):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.chroma_client = None
        self.collection = None
        self._init_databases()
    
    def _get_conn(self):
        """Always return a new SQLite connection (thread-safe)."""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def _init_databases(self):
        """Initialize SQL and vector databases."""
        with self._get_conn() as conn:
            self._create_tables(conn)
        
        # Initialize ChromaDB (telemetry disabled)
        self.chroma_client = PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name="argo_profiles",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self.collection = self.chroma_client.get_collection("argo_profiles")
        
        logger.info("Database initialized successfully")
    
    def _create_tables(self, conn):
        """Create necessary SQL tables."""
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform_number TEXT,
                profile_number INTEGER,
                latitude REAL,
                longitude REAL,
                time TIMESTAMP,
                pressure REAL,
                depth REAL,
                temperature REAL,
                salinity REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform_number TEXT UNIQUE,
                num_profiles INTEGER,
                num_measurements INTEGER,
                lat_min REAL,
                lat_max REAL,
                lon_min REAL,
                lon_max REAL,
                time_start TIMESTAMP,
                time_end TIMESTAMP,
                depth_min REAL,
                depth_max REAL
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_platform ON profiles(platform_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON profiles(time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON profiles(latitude, longitude)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_depth ON profiles(depth)")
        
        conn.commit()
        logger.info("SQL tables created successfully")
    
    def insert_profile_data(self, df: pd.DataFrame, metadata: Dict):
        """Insert profile data into SQL database."""
        if df.empty:
            logger.warning("Empty dataframe, skipping insertion")
            return
        
        with self._get_conn() as conn:
            df.to_sql('profiles', conn, if_exists='append', index=False)
            
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO metadata 
                (platform_number, num_profiles, num_measurements, 
                 lat_min, lat_max, lon_min, lon_max, 
                 time_start, time_end, depth_min, depth_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata['platform_number'],
                metadata['num_profiles'],
                metadata['num_measurements'],
                metadata['lat_range'][0], metadata['lat_range'][1],
                metadata['lon_range'][0], metadata['lon_range'][1],
                metadata['time_range'][0], metadata['time_range'][1],
                metadata['depth_range'][0], metadata['depth_range'][1]
            ))
            conn.commit()
        
        logger.info(f"Inserted {len(df)} records for platform {metadata['platform_number']}")
    
    def execute_sql(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            with self._get_conn() as conn:
                df = pd.read_sql_query(query, conn)
            logger.info(f"SQL query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"SQL query error: {e}")
            return pd.DataFrame()
    
    def get_platform_list(self) -> List[str]:
        """Get list of all platform numbers in database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT platform_number FROM profiles")
            return [row[0] for row in cursor.fetchall()]
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM profiles")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT platform_number) FROM profiles")
            total_platforms = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(time), MAX(time) FROM profiles")
            time_range = cursor.fetchone()
            
            cursor.execute("""
                SELECT MIN(latitude), MAX(latitude), 
                       MIN(longitude), MAX(longitude) 
                FROM profiles
            """)
            bounds = cursor.fetchone()
        
        return {
            'total_records': total_records,
            'total_platforms': total_platforms,
            'time_range': time_range,
            'geographic_bounds': {
                'lat_min': bounds[0] if bounds else None,
                'lat_max': bounds[1] if bounds else None,
                'lon_min': bounds[2] if bounds else None,
                'lon_max': bounds[3] if bounds else None
            }
        }
    
    def add_to_vector_db(self, doc_id: str, text: str, metadata: Dict):
        """Add document to ChromaDB vector database."""
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.info(f"Added document {doc_id} to vector database")
        except Exception as e:
            logger.error(f"Error adding to vector database: {e}")
    
    def query_vector_db(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the vector database for similar documents."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return [
                {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                for i in range(len(results['ids'][0]))
            ]
        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            return []
