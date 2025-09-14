
# # ARGO Data Ingestion Module - Optimized for CSV Input
# # Handles reading and processing of ARGO CSV files with Indian Ocean data
# # """

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Optional, Tuple
# from pathlib import Path
# import logging
# from datetime import datetime

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class ARGODataIngester:
#     """Processes ARGO CSV files and extracts relevant oceanographic data."""
    
#     def __init__(self):
#         # Updated column mappings based on your CSV structure
#         self.column_mappings = {
#             'Date': 'time',
#             'Depth (m)': 'depth',
#             'Temperature (°C)': 'temperature', 
#             'Salinity (PSL)': 'salinity',
#             'Pressure (dbar)': 'pressure',
#             'Ocean Region': 'region',
#             'DOXY (A,µmol/kg)': 'dissolved_oxygen',
#             'Nitrate (A,µmol/kg)': 'nitrate',
#             'Phosphate (A,µmol/kg)': 'phosphate',
#             'Silicate (A,µmol/kg)': 'silicate',
#             'pH': 'ph',
#             'Chlorophyll (mg/m³)': 'chlorophyll',
#             'Density (kg/m³)': 'density',
#             'Potential Density (kg/m³)': 'potential_density',
#             'Total Alkalinity (µmol/kg)': 'alkalinity',
#             'pCO2w (A,µatm)': 'pco2',
#             'Fluorescence (A,mg/m³)': 'fluorescence',
#             'Turbidity (NTU)': 'turbidity',
#             'PAR (A,µmol photons/m²/s)': 'par'
#         }
        
#         self.required_vars = ['time', 'depth', 'temperature', 'salinity', 'pressure']
#         self.data_cache = {}
        
#     def read_csv(self, filepath: str) -> pd.DataFrame:
#         """Read CSV file with proper parsing."""
#         try:
#             # Read CSV with robust parsing
#             df = pd.read_csv(
#                 filepath,
#                 parse_dates=['Date'],
#                 date_parser=lambda x: pd.to_datetime(x, errors='coerce'),
#                 na_values=['', 'NaN', 'nan', 'NULL', 'null', '-999', -999],
#                 skipinitialspace=True
#             )
            
#             # Clean column names (remove extra spaces)
#             df.columns = df.columns.str.strip()
            
#             logger.info(f"Successfully loaded CSV file: {filepath}")
#             logger.info(f"Shape: {df.shape}, Columns: {list(df.columns)}")
#             return df
#         except Exception as e:
#             logger.error(f"Error reading CSV file {filepath}: {e}")
#             raise
    
#     def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Standardize column names and add missing essential columns."""
#         df_clean = df.copy()
        
#         # Rename columns to standard names
#         rename_dict = {}
#         for old_col, new_col in self.column_mappings.items():
#             if old_col in df_clean.columns:
#                 rename_dict[old_col] = new_col
        
#         df_clean = df_clean.rename(columns=rename_dict)
        
#         # Add missing essential columns with estimates
#         if 'latitude' not in df_clean.columns:
#             # Estimate latitude based on region
#             region_lat_mapping = {
#                 'Arabian Sea': 15.0,
#                 'South Indian Ocean': -30.0,
#                 'Eastern Indian Ocean': -10.0,
#                 'Bay of Bengal': 15.0
#             }
#             df_clean['latitude'] = df_clean['region'].map(region_lat_mapping).fillna(0.0)
        
#         if 'longitude' not in df_clean.columns:
#             # Estimate longitude based on region
#             region_lon_mapping = {
#                 'Arabian Sea': 65.0,
#                 'South Indian Ocean': 80.0,
#                 'Eastern Indian Ocean': 95.0,
#                 'Bay of Bengal': 88.0
#             }
#             df_clean['longitude'] = df_clean['region'].map(region_lon_mapping).fillna(0.0)
        
#         # Generate synthetic platform numbers based on region and time grouping
#         if 'platform_number' not in df_clean.columns:
#             df_clean['platform_number'] = (
#                 df_clean['region'].astype(str) + '_' + 
#                 df_clean['time'].dt.strftime('%Y%m').astype(str)
#             ).fillna('UNKNOWN_PLATFORM')
        
#         # Generate profile numbers (group by platform and date)
#         if 'profile_number' not in df_clean.columns:
#             df_clean['profile_number'] = (
#                 df_clean.groupby(['platform_number', df_clean['time'].dt.date])
#                 .ngroup()
#             )
        
#         return df_clean
    
#     def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Clean and validate the data."""
#         df_clean = df.copy()
        
#         # Remove rows with all NaN values for essential variables
#         essential_cols = ['temperature', 'salinity', 'pressure', 'depth']
#         existing_essential = [col for col in essential_cols if col in df_clean.columns]
#         df_clean = df_clean.dropna(subset=existing_essential, how='all')
        
#         # Clean specific columns
#         numeric_columns = [
#             'depth', 'temperature', 'salinity', 'pressure', 'latitude', 'longitude',
#             'dissolved_oxygen', 'nitrate', 'phosphate', 'silicate', 'ph', 
#             'chlorophyll', 'density', 'potential_density', 'alkalinity', 
#             'pco2', 'fluorescence', 'turbidity', 'par'
#         ]
        
#         for col in numeric_columns:
#             if col in df_clean.columns:
#                 # Convert to numeric, coercing errors to NaN
#                 df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
#                 # Apply reasonable bounds for oceanographic data
#                 if col == 'temperature':
#                     df_clean[col] = df_clean[col].clip(-2, 40)
#                 elif col == 'salinity':
#                     df_clean[col] = df_clean[col].clip(20, 45)
#                 elif col == 'pressure':
#                     df_clean[col] = df_clean[col].clip(0, 11000)
#                 elif col == 'depth':
#                     df_clean[col] = df_clean[col].clip(0, 11000)
#                 elif col in ['latitude']:
#                     df_clean[col] = df_clean[col].clip(-90, 90)
#                 elif col in ['longitude']:
#                     df_clean[col] = df_clean[col].clip(-180, 180)
        
#         # Remove duplicate rows
#         df_clean = df_clean.drop_duplicates()
        
#         logger.info(f"Cleaned data shape: {df_clean.shape}")
#         return df_clean
    
#     def extract_profile_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Extract and structure profile data."""
#         df_processed = self.standardize_columns(df)
#         df_clean = self.clean_data(df_processed)
        
#         # Ensure we have required columns
#         required_columns = {
#             'platform_number': 'UNKNOWN',
#             'profile_number': 0,
#             'latitude': 0.0,
#             'longitude': 0.0,
#             'time': pd.Timestamp.now(),
#             'depth': 0.0,
#             'temperature': np.nan,
#             'salinity': np.nan,
#             'pressure': 0.0
#         }
        
#         for col, default_val in required_columns.items():
#             if col not in df_clean.columns:
#                 df_clean[col] = default_val
        
#         return df_clean
    
#     def process_csv_file(self, filepath: str) -> Tuple[pd.DataFrame, Dict]:
#         """Process a single CSV file."""
#         df_raw = self.read_csv(filepath)
#         df = self.extract_profile_data(df_raw)
        
#         # Generate metadata for vector database
#         metadata = self.generate_metadata(df, filepath)
        
#         return df, metadata
    
#     def generate_metadata(self, df: pd.DataFrame, filepath: str) -> Dict:
#         """Generate metadata dictionary."""
#         if df.empty:
#             return {
#                 'file_path': filepath,
#                 'platform_number': 'unknown',
#                 'num_profiles': 0,
#                 'num_measurements': 0,
#                 'lat_range': [0, 0],
#                 'lon_range': [0, 0],
#                 'time_range': ['', ''],
#                 'depth_range': [0, 0],
#                 'variables': []
#             }
        
#         # Calculate time range safely
#         time_col = df['time'].dropna()
#         if len(time_col) > 0:
#             time_start = time_col.min().isoformat()
#             time_end = time_col.max().isoformat()
#         else:
#             time_start = time_end = ''
        
#         # Get unique regions for richer metadata
#         regions = df['region'].unique().tolist() if 'region' in df.columns else []
        
#         metadata = {
#             'file_path': filepath,
#             'platform_number': str(df['platform_number'].iloc[0]) if 'platform_number' in df.columns else 'unknown',
#             'num_profiles': int(df['profile_number'].nunique()) if 'profile_number' in df.columns else 1,
#             'num_measurements': len(df),
#             'lat_range': [float(df['latitude'].min()), float(df['latitude'].max())] if 'latitude' in df.columns else [0, 0],
#             'lon_range': [float(df['longitude'].min()), float(df['longitude'].max())] if 'longitude' in df.columns else [0, 0],
#             'time_range': [time_start, time_end],
#             'depth_range': [float(df['depth'].min()), float(df['depth'].max())] if 'depth' in df.columns else [0, 0],
#             'variables': list(df.columns),
#             'regions': regions,
#             'data_source': 'Indian Ocean ARGO'
#         }
        
#         return metadata
    
#     def generate_summary_text(self, df: pd.DataFrame, metadata: Dict) -> str:
#         """Generate enhanced text summary for vector database."""
#         if df.empty:
#             return "Empty dataset"
        
#         # Get region-specific information
#         regions_text = ", ".join(metadata.get('regions', ['Unknown Region']))
        
#         # Calculate additional statistics
#         stats = {}
#         for var in ['temperature', 'salinity', 'dissolved_oxygen', 'chlorophyll']:
#             if var in df.columns and not df[var].isna().all():
#                 stats[var] = {
#                     'mean': df[var].mean(),
#                     'std': df[var].std(),
#                     'min': df[var].min(),
#                     'max': df[var].max()
#                 }
        
#         summary_parts = [
#             f"ARGO Float Data from {regions_text}",
#             f"Location: Latitude {metadata['lat_range'][0]:.2f} to {metadata['lat_range'][1]:.2f}, "
#             f"Longitude {metadata['lon_range'][0]:.2f} to {metadata['lon_range'][1]:.2f}",
#             f"Time Period: {metadata['time_range'][0]} to {metadata['time_range'][1]}",
#             f"Depth Range: {metadata['depth_range'][0]:.1f} to {metadata['depth_range'][1]:.1f} meters",
#             f"Number of Profiles: {metadata['num_profiles']}",
#             f"Total Measurements: {metadata['num_measurements']}"
#         ]
        
#         # Add variable statistics
#         for var, stat in stats.items():
#             if stat:
#                 unit = self._get_unit(var)
#                 summary_parts.append(
#                     f"{var.title()} Range: {stat['min']:.2f} to {stat['max']:.2f} {unit} "
#                     f"(mean: {stat['mean']:.2f} ± {stat['std']:.2f})"
#                 )
        
#         return "\n".join(summary_parts)
    
#     def _get_unit(self, variable: str) -> str:
#         """Get appropriate unit for variable."""
#         units = {
#             'temperature': '°C',
#             'salinity': 'PSU',
#             'pressure': 'dbar',
#             'depth': 'm',
#             'dissolved_oxygen': 'µmol/kg',
#             'nitrate': 'µmol/kg',
#             'phosphate': 'µmol/kg',
#             'silicate': 'µmol/kg',
#             'chlorophyll': 'mg/m³',
#             'density': 'kg/m³',
#             'alkalinity': 'µmol/kg',
#             'pco2': 'µatm',
#             'fluorescence': 'mg/m³',
#             'turbidity': 'NTU',
#             'par': 'µmol photons/m²/s'
#         }
#         return units.get(variable, '')


# def create_sample_data_from_csv(csv_path: str) -> pd.DataFrame:
#     """Create sample data from existing CSV file."""
#     ingester = ARGODataIngester()
#     try:
#         df, _ = ingester.process_csv_file(csv_path)
#         return df.head(1000)  # Return first 1000 rows as sample
#     except Exception as e:
#         logger.error(f"Error creating sample data: {e}")
#         return pd.DataFrame()





















"""
ARGO Data Ingestion Module - Optimized for CSV Input
Handles reading and processing of ARGO CSV files with Indian Ocean data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARGODataIngester:
    """Processes ARGO CSV files and extracts relevant oceanographic data."""
    
    def __init__(self):
        # Updated column mappings based on your CSV structure
        self.column_mappings = {
            'Date': 'time',
            'Depth (m)': 'depth',
            'Temperature (°C)': 'temperature', 
            'Salinity (PSL)': 'salinity',
            'Pressure (dbar)': 'pressure',
            'Ocean Region': 'region',
            'DOXY (A,µmol/kg)': 'dissolved_oxygen',
            'Nitrate (A,µmol/kg)': 'nitrate',
            'Phosphate (A,µmol/kg)': 'phosphate',
            'Silicate (A,µmol/kg)': 'silicate',
            'pH': 'ph',
            'Chlorophyll (mg/m³)': 'chlorophyll',
            'Density (kg/m³)': 'density',
            'Potential Density (kg/m³)': 'potential_density',
            'Total Alkalinity (µmol/kg)': 'alkalinity',
            'pCO2w (A,µatm)': 'pco2',
            'Fluorescence (A,mg/m³)': 'fluorescence',
            'Turbidity (NTU)': 'turbidity',
            'PAR (A,µmol photons/m²/s)': 'par'
        }
        
        self.required_vars = ['time', 'depth', 'temperature', 'salinity', 'pressure']
        self.data_cache = {}
        
    def read_csv(self, filepath: str) -> pd.DataFrame:
        """Read CSV file with proper parsing."""
        try:
            # Read CSV with robust parsing
            df = pd.read_csv(
                filepath,
                parse_dates=['Date'],
                date_parser=lambda x: pd.to_datetime(x, errors='coerce'),
                na_values=['', 'NaN', 'nan', 'NULL', 'null', '-999', -999],
                skipinitialspace=True
            )
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            logger.info(f"Successfully loaded CSV file: {filepath}")
            logger.info(f"Shape: {df.shape}, Columns: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file {filepath}: {e}")
            raise
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and add missing essential columns."""
        df_clean = df.copy()
        
        # Rename columns to standard names
        rename_dict = {}
        for old_col, new_col in self.column_mappings.items():
            if old_col in df_clean.columns:
                rename_dict[old_col] = new_col
        
        df_clean = df_clean.rename(columns=rename_dict)
        
        # Add missing essential columns with estimates
        if 'latitude' not in df_clean.columns:
            # Estimate latitude based on region
            region_lat_mapping = {
                'Arabian Sea': 15.0,
                'South Indian Ocean': -30.0,
                'Eastern Indian Ocean': -10.0,
                'Bay of Bengal': 15.0
            }
            df_clean['latitude'] = df_clean['region'].map(region_lat_mapping).fillna(0.0)
        
        if 'longitude' not in df_clean.columns:
            # Estimate longitude based on region
            region_lon_mapping = {
                'Arabian Sea': 65.0,
                'South Indian Ocean': 80.0,
                'Eastern Indian Ocean': 95.0,
                'Bay of Bengal': 88.0
            }
            df_clean['longitude'] = df_clean['region'].map(region_lon_mapping).fillna(0.0)
        
        # Generate synthetic platform numbers based on region and time grouping
        if 'platform_number' not in df_clean.columns:
            df_clean['platform_number'] = (
                df_clean['region'].astype(str) + '_' + 
                df_clean['time'].dt.strftime('%Y%m').astype(str)
            ).fillna('UNKNOWN_PLATFORM')
        
        # Generate profile numbers (group by platform and date)
        if 'profile_number' not in df_clean.columns:
            df_clean['profile_number'] = (
                df_clean.groupby(['platform_number', df_clean['time'].dt.date])
                .ngroup()
            )
        
        return df_clean
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the data."""
        df_clean = df.copy()
        
        # Remove rows with all NaN values for essential variables
        essential_cols = ['temperature', 'salinity', 'pressure', 'depth']
        existing_essential = [col for col in essential_cols if col in df_clean.columns]
        df_clean = df_clean.dropna(subset=existing_essential, how='all')
        
        # Clean specific columns
        numeric_columns = [
            'depth', 'temperature', 'salinity', 'pressure', 'latitude', 'longitude',
            'dissolved_oxygen', 'nitrate', 'phosphate', 'silicate', 'ph', 
            'chlorophyll', 'density', 'potential_density', 'alkalinity', 
            'pco2', 'fluorescence', 'turbidity', 'par'
        ]
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Convert to numeric, coercing errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Apply reasonable bounds for oceanographic data
                if col == 'temperature':
                    df_clean[col] = df_clean[col].clip(-2, 40)
                elif col == 'salinity':
                    df_clean[col] = df_clean[col].clip(20, 45)
                elif col == 'pressure':
                    df_clean[col] = df_clean[col].clip(0, 11000)
                elif col == 'depth':
                    df_clean[col] = df_clean[col].clip(0, 11000)
                elif col in ['latitude']:
                    df_clean[col] = df_clean[col].clip(-90, 90)
                elif col in ['longitude']:
                    df_clean[col] = df_clean[col].clip(-180, 180)
        
        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def extract_profile_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and structure profile data."""
        df_processed = self.standardize_columns(df)
        df_clean = self.clean_data(df_processed)
        
        # Ensure we have required columns
        required_columns = {
            'platform_number': 'UNKNOWN',
            'profile_number': 0,
            'latitude': 0.0,
            'longitude': 0.0,
            'time': pd.Timestamp.now(),
            'depth': 0.0,
            'temperature': np.nan,
            'salinity': np.nan,
            'pressure': 0.0
        }
        
        for col, default_val in required_columns.items():
            if col not in df_clean.columns:
                df_clean[col] = default_val
        
        return df_clean
    
    def debug_csv_structure(self, filepath: str) -> Dict:
        """Debug function to examine CSV structure before processing."""
        try:
            df = pd.read_csv(filepath, nrows=5)  # Read only first 5 rows
            debug_info = {
                'total_columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'sample_data': df.head().to_dict(),
                'data_types': df.dtypes.to_dict()
            }
            logger.info(f"CSV Debug Info: {debug_info}")
            return debug_info
        except Exception as e:
            logger.error(f"Error debugging CSV: {e}")
            return {}
    
    def process_csv_file(self, filepath: str) -> Tuple[pd.DataFrame, Dict]:
        """Process a single CSV file with enhanced debugging."""
        # First debug the CSV structure
        debug_info = self.debug_csv_structure(filepath)
        
        df_raw = self.read_csv(filepath)
        logger.info(f"Raw CSV shape: {df_raw.shape}")
        logger.info(f"Raw CSV columns: {df_raw.columns.tolist()}")
        
        df = self.extract_profile_data(df_raw)
        logger.info(f"Processed CSV shape: {df.shape}")
        logger.info(f"Processed CSV columns: {df.columns.tolist()}")
        
        # Generate metadata for vector database
        metadata = self.generate_metadata(df, filepath)
        
        return df, metadata
    
    def generate_metadata(self, df: pd.DataFrame, filepath: str) -> Dict:
        """Generate metadata dictionary."""
        if df.empty:
            return {
                'file_path': filepath,
                'platform_number': 'unknown',
                'num_profiles': 0,
                'num_measurements': 0,
                'lat_range': [0, 0],
                'lon_range': [0, 0],
                'time_range': ['', ''],
                'depth_range': [0, 0],
                'variables': []
            }
        
        # Calculate time range safely
        time_col = df['time'].dropna()
        if len(time_col) > 0:
            time_start = time_col.min().isoformat()
            time_end = time_col.max().isoformat()
        else:
            time_start = time_end = ''
        
        # Get unique regions for richer metadata
        regions = df['region'].unique().tolist() if 'region' in df.columns else []
        
        metadata = {
            'file_path': filepath,
            'platform_number': str(df['platform_number'].iloc[0]) if 'platform_number' in df.columns else 'unknown',
            'num_profiles': int(df['profile_number'].nunique()) if 'profile_number' in df.columns else 1,
            'num_measurements': len(df),
            'lat_range': [float(df['latitude'].min()), float(df['latitude'].max())] if 'latitude' in df.columns else [0, 0],
            'lon_range': [float(df['longitude'].min()), float(df['longitude'].max())] if 'longitude' in df.columns else [0, 0],
            'time_range': [time_start, time_end],
            'depth_range': [float(df['depth'].min()), float(df['depth'].max())] if 'depth' in df.columns else [0, 0],
            'variables': list(df.columns),
            'regions': regions,
            'data_source': 'Indian Ocean ARGO'
        }
        
        return metadata
    
    def generate_summary_text(self, df: pd.DataFrame, metadata: Dict) -> str:
        """Generate enhanced text summary for vector database."""
        if df.empty:
            return "Empty dataset"
        
        # Get region-specific information
        regions_text = ", ".join(metadata.get('regions', ['Unknown Region']))
        
        # Calculate additional statistics
        stats = {}
        for var in ['temperature', 'salinity', 'dissolved_oxygen', 'chlorophyll']:
            if var in df.columns and not df[var].isna().all():
                stats[var] = {
                    'mean': df[var].mean(),
                    'std': df[var].std(),
                    'min': df[var].min(),
                    'max': df[var].max()
                }
        
        summary_parts = [
            f"ARGO Float Data from {regions_text}",
            f"Location: Latitude {metadata['lat_range'][0]:.2f} to {metadata['lat_range'][1]:.2f}, "
            f"Longitude {metadata['lon_range'][0]:.2f} to {metadata['lon_range'][1]:.2f}",
            f"Time Period: {metadata['time_range'][0]} to {metadata['time_range'][1]}",
            f"Depth Range: {metadata['depth_range'][0]:.1f} to {metadata['depth_range'][1]:.1f} meters",
            f"Number of Profiles: {metadata['num_profiles']}",
            f"Total Measurements: {metadata['num_measurements']}"
        ]
        
        # Add variable statistics
        for var, stat in stats.items():
            if stat:
                unit = self._get_unit(var)
                summary_parts.append(
                    f"{var.title()} Range: {stat['min']:.2f} to {stat['max']:.2f} {unit} "
                    f"(mean: {stat['mean']:.2f} ± {stat['std']:.2f})"
                )
        
        return "\n".join(summary_parts)
    
    def _get_unit(self, variable: str) -> str:
        """Get appropriate unit for variable."""
        units = {
            'temperature': '°C',
            'salinity': 'PSU',
            'pressure': 'dbar',
            'depth': 'm',
            'dissolved_oxygen': 'µmol/kg',
            'nitrate': 'µmol/kg',
            'phosphate': 'µmol/kg',
            'silicate': 'µmol/kg',
            'chlorophyll': 'mg/m³',
            'density': 'kg/m³',
            'alkalinity': 'µmol/kg',
            'pco2': 'µatm',
            'fluorescence': 'mg/m³',
            'turbidity': 'NTU',
            'par': 'µmol photons/m²/s'
        }
        return units.get(variable, '')


def create_sample_data_from_csv(csv_path: str) -> pd.DataFrame:
    """Create sample data from existing CSV file."""
    ingester = ARGODataIngester()
    try:
        df, _ = ingester.process_csv_file(csv_path)
        return df.head(1000)  # Return first 1000 rows as sample
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return pd.DataFrame()