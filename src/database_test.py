"""
Database Management Module - Optimized for Indian Ocean ARGO Data
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
from typing import List, Dict, Union
import json

# Disable Chroma telemetry globally
os.environ["CHROMA_TELEMETRY_ENABLED"] = "0"

# Ensure data folder exists
Path("data").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Enhanced database manager for Indian Ocean ARGO data with BGC parameters."""
    
    def __init__(self, db_path: str = "data/indian_ocean_argo.db", chroma_path: str = "data/chroma"):
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.chroma_client = None
        self.collection = None
        self._init_databases()
    
    def _get_conn(self):
        """Always return a new SQLite connection (thread-safe)."""
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def _init_databases(self):
        """Initialize SQL and vector databases with enhanced schema."""
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
                name="indian_ocean_argo",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            self.collection = self.chroma_client.get_collection("indian_ocean_argo")
        
        logger.info("Enhanced database initialized successfully")
    
    def _create_tables(self, conn):
        """Create comprehensive SQL tables for Indian Ocean data."""
        cursor = conn.cursor()
        
        # Enhanced profiles table with all biogeochemical parameters
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform_number TEXT,
                profile_number INTEGER,
                latitude REAL,
                longitude REAL,
                time TIMESTAMP,
                depth REAL,
                pressure REAL,
                temperature REAL,
                salinity REAL,
                region TEXT,
                dissolved_oxygen REAL,
                nitrate REAL,
                phosphate REAL,
                silicate REAL,
                ph REAL,
                chlorophyll REAL,
                density REAL,
                potential_density REAL,
                alkalinity REAL,
                pco2 REAL,
                fluorescence REAL,
                turbidity REAL,
                par REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Enhanced metadata table
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
                depth_max REAL,
                regions TEXT,  -- JSON array of regions
                data_source TEXT,
                variables TEXT,  -- JSON array of available variables
                quality_flags TEXT,  -- JSON object with quality metrics
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Regional statistics table for quick access
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regional_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT UNIQUE,
                measurement_count INTEGER,
                platform_count INTEGER,
                avg_temperature REAL,
                avg_salinity REAL,
                avg_oxygen REAL,
                avg_chlorophyll REAL,
                depth_range_min REAL,
                depth_range_max REAL,
                time_range_start TIMESTAMP,
                time_range_end TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create optimized indices
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_platform ON profiles(platform_number)",
            "CREATE INDEX IF NOT EXISTS idx_time ON profiles(time)",
            "CREATE INDEX IF NOT EXISTS idx_location ON profiles(latitude, longitude)",
            "CREATE INDEX IF NOT EXISTS idx_depth ON profiles(depth)",
            "CREATE INDEX IF NOT EXISTS idx_region ON profiles(region)",
            "CREATE INDEX IF NOT EXISTS idx_temp_sal ON profiles(temperature, salinity)",
            "CREATE INDEX IF NOT EXISTS idx_bgc ON profiles(dissolved_oxygen, chlorophyll, nitrate)",
            "CREATE INDEX IF NOT EXISTS idx_composite ON profiles(region, time, depth)"
        ]
        
        for index_sql in indices:
            cursor.execute(index_sql)
        
        conn.commit()
        logger.info("Enhanced SQL tables and indices created successfully")
    
    def insert_profile_data(self, df: pd.DataFrame, metadata: Dict):
        """Insert profile data with enhanced metadata handling."""
        if df.empty:
            logger.warning("Empty dataframe, skipping insertion")
            return
        
        with self._get_conn() as conn:
            # Insert profile data
            df.to_sql('profiles', conn, if_exists='append', index=False)
            
            # Prepare metadata for insertion
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO metadata 
                (platform_number, num_profiles, num_measurements, 
                 lat_min, lat_max, lon_min, lon_max, 
                 time_start, time_end, depth_min, depth_max,
                 regions, data_source, variables, quality_flags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata['platform_number'],
                metadata['num_profiles'],
                metadata['num_measurements'],
                metadata['lat_range'][0], metadata['lat_range'][1],
                metadata['lon_range'][0], metadata['lon_range'][1],
                metadata['time_range'][0], metadata['time_range'][1],
                metadata['depth_range'][0], metadata['depth_range'][1],
                json.dumps(metadata.get('regions', [])),
                metadata.get('data_source', 'ARGO'),
                json.dumps(metadata.get('variables', [])),
                json.dumps(self._calculate_quality_flags(df))
            ))
            
            # Update regional statistics
            self._update_regional_stats(conn, df)
            
            conn.commit()
        
        logger.info(f"Inserted {len(df)} records for platform {metadata['platform_number']}")
    
    def _calculate_quality_flags(self, df: pd.DataFrame) -> Dict:
        """Calculate data quality metrics."""
        quality = {
            'completeness': {},
            'outliers': {},
            'temporal_coverage': {},
            'spatial_coverage': {}
        }
        
        # Calculate completeness for key variables
        key_vars = ['temperature', 'salinity', 'pressure', 'dissolved_oxygen', 'chlorophyll']
        for var in key_vars:
            if var in df.columns:
                quality['completeness'][var] = (1 - df[var].isna().sum() / len(df))
        
        # Detect outliers using IQR method
        numeric_cols = df.select_dtypes(include=[float, int]).columns
        for col in numeric_cols:
            if col in df.columns and not df[col].isna().all():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                quality['outliers'][col] = outliers / len(df) if len(df) > 0 else 0
        
        # Temporal and spatial coverage
        if 'time' in df.columns:
            time_span = (df['time'].max() - df['time'].min()).days
            quality['temporal_coverage'] = {'days': time_span}
        
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_range = df['latitude'].max() - df['latitude'].min()
            lon_range = df['longitude'].max() - df['longitude'].min()
            quality['spatial_coverage'] = {
                'lat_range': lat_range,
                'lon_range': lon_range
            }
        
        return quality
    
    def _update_regional_stats(self, conn, df: pd.DataFrame):
        """Update regional statistics table."""
        if 'region' not in df.columns:
            return
        
        cursor = conn.cursor()
        
        for region in df['region'].unique():
            if pd.isna(region):
                continue
                
            region_df = df[df['region'] == region]
            
            # Calculate statistics
            stats = {
                'region': region,
                'measurement_count': len(region_df),
                'platform_count': region_df['platform_number'].nunique() if 'platform_number' in region_df else 1,
                'avg_temperature': region_df['temperature'].mean() if 'temperature' in region_df else None,
                'avg_salinity': region_df['salinity'].mean() if 'salinity' in region_df else None,
                'avg_oxygen': region_df['dissolved_oxygen'].mean() if 'dissolved_oxygen' in region_df else None,
                'avg_chlorophyll': region_df['chlorophyll'].mean() if 'chlorophyll' in region_df else None,
                'depth_range_min': region_df['depth'].min() if 'depth' in region_df else None,
                'depth_range_max': region_df['depth'].max() if 'depth' in region_df else None,
                'time_range_start': region_df['time'].min().isoformat() if 'time' in region_df and not region_df['time'].isna().all() else None,
                'time_range_end': region_df['time'].max().isoformat() if 'time' in region_df and not region_df['time'].isna().all() else None
            }
            
            # Insert or update regional stats
            cursor.execute("""
                INSERT OR REPLACE INTO regional_stats 
                (region, measurement_count, platform_count, avg_temperature, avg_salinity,
                 avg_oxygen, avg_chlorophyll, depth_range_min, depth_range_max,
                 time_range_start, time_range_end, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                stats['region'], stats['measurement_count'], stats['platform_count'],
                stats['avg_temperature'], stats['avg_salinity'], stats['avg_oxygen'],
                stats['avg_chlorophyll'], stats['depth_range_min'], stats['depth_range_max'],
                stats['time_range_start'], stats['time_range_end']
            ))
    
    def execute_sql(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute SQL query with optional parameters."""
        try:
            with self._get_conn() as conn:
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
                    df = pd.read_sql_query(query, conn)
            logger.info(f"SQL query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"SQL query error: {e}")
            return pd.DataFrame()
    
    def get_regional_summary(self) -> pd.DataFrame:
        """Get summary statistics by region."""
        query = """
            SELECT region, measurement_count, platform_count,
                   ROUND(avg_temperature, 2) as avg_temp_c,
                   ROUND(avg_salinity, 2) as avg_salinity_psu,
                   ROUND(avg_oxygen, 2) as avg_oxygen,
                   ROUND(avg_chlorophyll, 3) as avg_chlorophyll,
                   ROUND(depth_range_min, 1) as min_depth_m,
                   ROUND(depth_range_max, 1) as max_depth_m,
                   time_range_start, time_range_end
            FROM regional_stats
            ORDER BY measurement_count DESC
        """
        return self.execute_sql(query)
    
    def get_platform_list(self) -> List[str]:
        """Get list of all platform numbers in database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT platform_number FROM profiles ORDER BY platform_number")
            return [row[0] for row in cursor.fetchall()]
    
    def get_available_variables(self) -> Dict[str, List[str]]:
        """Get available variables categorized by type."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(profiles)")
            columns = [row[1] for row in cursor.fetchall()]
        
        # Categorize variables
        categories = {
            'physical': ['temperature', 'salinity', 'pressure', 'depth', 'density', 'potential_density'],
            'biogeochemical': ['dissolved_oxygen', 'nitrate', 'phosphate', 'silicate', 'ph', 'alkalinity', 'pco2'],
            'optical': ['chlorophyll', 'fluorescence', 'turbidity', 'par'],
            'spatial_temporal': ['latitude', 'longitude', 'time', 'region'],
            'metadata': ['platform_number', 'profile_number']
        }
        
        # Filter only available columns
        available = {}
        for category, vars_list in categories.items():
            available[category] = [var for var in vars_list if var in columns]
        
        return available
    
    def get_data_summary(self) -> Dict:
        """Get enhanced summary statistics of the database."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM profiles")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT platform_number) FROM profiles")
            total_platforms = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT region) FROM profiles WHERE region IS NOT NULL")
            total_regions = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(time), MAX(time) FROM profiles WHERE time IS NOT NULL")
            time_range = cursor.fetchone()
            
            cursor.execute("""
                SELECT MIN(latitude), MAX(latitude), 
                       MIN(longitude), MAX(longitude) 
                FROM profiles WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """)
            bounds = cursor.fetchone()
            
            cursor.execute("""
                SELECT MIN(depth), MAX(depth), AVG(depth)
                FROM profiles WHERE depth IS NOT NULL
            """)
            depth_stats = cursor.fetchone()
            
            # Get biogeochemical data coverage
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN dissolved_oxygen IS NOT NULL THEN 1 END) as oxygen_count,
                    COUNT(CASE WHEN chlorophyll IS NOT NULL THEN 1 END) as chlorophyll_count,
                    COUNT(CASE WHEN nitrate IS NOT NULL THEN 1 END) as nitrate_count,
                    COUNT(CASE WHEN ph IS NOT NULL THEN 1 END) as ph_count
                FROM profiles
            """)
            bgc_coverage = cursor.fetchone()
        
        return {
            'total_records': total_records,
            'total_platforms': total_platforms,
            'total_regions': total_regions,
            'time_range': time_range,
            'geographic_bounds': {
                'lat_min': bounds[0] if bounds and bounds[0] else None,
                'lat_max': bounds[1] if bounds and bounds[1] else None,
                'lon_min': bounds[2] if bounds and bounds[2] else None,
                'lon_max': bounds[3] if bounds and bounds[3] else None
            },
            'depth_stats': {
                'min_depth': depth_stats[0] if depth_stats and depth_stats[0] else None,
                'max_depth': depth_stats[1] if depth_stats and depth_stats[1] else None,
                'avg_depth': depth_stats[2] if depth_stats and depth_stats[2] else None
            },
            'bgc_coverage': {
                'oxygen_records': bgc_coverage[0] if bgc_coverage else 0,
                'chlorophyll_records': bgc_coverage[1] if bgc_coverage else 0,
                'nitrate_records': bgc_coverage[2] if bgc_coverage else 0,
                'ph_records': bgc_coverage[3] if bgc_coverage else 0
            }
        }
    
    def search_by_region_and_depth(self, region: str, min_depth: float = 0, max_depth: float = 5000) -> pd.DataFrame:
        """Optimized query for region and depth-based searches."""
        query = """
            SELECT platform_number, latitude, longitude, time, depth, temperature, salinity,
                   dissolved_oxygen, chlorophyll, nitrate, ph
            FROM profiles 
            WHERE region = ? AND depth BETWEEN ? AND ?
            ORDER BY time DESC, depth ASC
        """
        return self.execute_sql(query, (region, min_depth, max_depth))
    
    def get_time_series_data(self, region: str = None, variable: str = 'temperature', 
                           depth_range: tuple = (0, 100)) -> pd.DataFrame:
        """Get time series data for visualization."""
        base_query = """
            SELECT time, depth, {variable}, region, platform_number, latitude, longitude
            FROM profiles 
            WHERE {variable} IS NOT NULL AND depth BETWEEN ? AND ?
        """.format(variable=variable)
        
        params = [depth_range[0], depth_range[1]]
        
        if region:
            base_query += " AND region = ?"
            params.append(region)
        
        base_query += " ORDER BY time ASC, depth ASC"
        
        return self.execute_sql(base_query, tuple(params))
    
    def add_to_vector_db(self, doc_id: str, text: str, metadata: Dict):
        """Add document to ChromaDB vector database."""
        try:
            # Ensure metadata values are JSON serializable
            clean_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (list, dict)):
                    clean_metadata[k] = str(v)
                elif pd.isna(v):
                    clean_metadata[k] = ""
                else:
                    clean_metadata[k] = str(v)
            
            self.collection.add(
                documents=[text],
                metadatas=[clean_metadata],
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
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
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