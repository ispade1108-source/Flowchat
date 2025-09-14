"""
ARGO Data Ingestion Module
Handles reading and processing of ARGO NetCDF files
"""

import xarray as xr
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARGODataIngester:
    """Processes ARGO NetCDF files and extracts relevant oceanographic data."""
    
    def __init__(self):
        self.required_vars = ['TEMP', 'PSAL', 'PRES', 'LATITUDE', 'LONGITUDE', 'JULD']
        self.data_cache = {}
        
    def read_netcdf(self, filepath: str) -> xr.Dataset:
        """Read NetCDF file using xarray."""
        try:
            ds = xr.open_dataset(filepath, decode_times=True)
            logger.info(f"Successfully loaded NetCDF file: {filepath}")
            return ds
        except Exception as e:
            logger.error(f"Error reading NetCDF file {filepath}: {e}")
            raise
            
    def extract_profile_data(self, ds: xr.Dataset) -> pd.DataFrame:
        """Extract profile data from ARGO dataset."""
        profiles = []
        
        try:
            # Get dimensions
            n_prof = len(ds.N_PROF) if 'N_PROF' in ds.dims else 1
            n_levels = len(ds.N_LEVELS) if 'N_LEVELS' in ds.dims else len(ds.PRES)
            
            # Extract metadata
            platform_number = str(ds.PLATFORM_NUMBER.values[0]) if 'PLATFORM_NUMBER' in ds else 'unknown'
            
            for prof_idx in range(n_prof):
                # Get profile location and time
                lat = float(ds.LATITUDE[prof_idx].values) if 'LATITUDE' in ds else 0.0
                lon = float(ds.LONGITUDE[prof_idx].values) if 'LONGITUDE' in ds else 0.0
                
                # Handle time - JULD is Julian days since 1950-01-01
                if 'JULD' in ds:
                    juld = float(ds.JULD[prof_idx].values)
                    # Convert Julian days to datetime
                    time = pd.Timestamp('1950-01-01') + pd.Timedelta(days=juld)
                else:
                    time = pd.Timestamp.now()
                
                # Extract vertical profiles
                if 'TEMP' in ds and 'PSAL' in ds and 'PRES' in ds:
                    temp = ds.TEMP[prof_idx, :].values if 'N_PROF' in ds.TEMP.dims else ds.TEMP.values
                    psal = ds.PSAL[prof_idx, :].values if 'N_PROF' in ds.PSAL.dims else ds.PSAL.values
                    pres = ds.PRES[prof_idx, :].values if 'N_PROF' in ds.PRES.dims else ds.PRES.values
                    
                    # Create profile records for each depth level
                    for level in range(len(pres)):
                        if not np.isnan(pres[level]):  # Skip NaN values
                            profiles.append({
                                'platform_number': platform_number,
                                'profile_number': prof_idx,
                                'latitude': lat,
                                'longitude': lon,
                                'time': time,
                                'pressure': float(pres[level]),
                                'temperature': float(temp[level]) if not np.isnan(temp[level]) else None,
                                'salinity': float(psal[level]) if not np.isnan(psal[level]) else None,
                                'depth': float(pres[level])  # Approximate depth from pressure
                            })
            
            df = pd.DataFrame(profiles)
            logger.info(f"Extracted {len(df)} data points from {n_prof} profiles")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting profile data: {e}")
            # Return minimal dataframe if extraction fails
            return pd.DataFrame()
    
    def process_file(self, filepath: str) -> Tuple[pd.DataFrame, Dict]:
        """Process a single ARGO NetCDF file."""
        ds = self.read_netcdf(filepath)
        df = self.extract_profile_data(ds)
        
        # Generate metadata for vector database
        metadata = {
            'file_path': filepath,
            'platform_number': df['platform_number'].iloc[0] if not df.empty else 'unknown',
            'num_profiles': df['profile_number'].nunique() if not df.empty else 0,
            'num_measurements': len(df),
            'lat_range': [df['latitude'].min(), df['latitude'].max()] if not df.empty else [0, 0],
            'lon_range': [df['longitude'].min(), df['longitude'].max()] if not df.empty else [0, 0],
            'time_range': [
                df['time'].min().isoformat() if not df.empty else '',
                df['time'].max().isoformat() if not df.empty else ''
            ],
            'depth_range': [df['depth'].min(), df['depth'].max()] if not df.empty else [0, 0],
            'variables': list(df.columns)
        }
        
        ds.close()
        return df, metadata
    
    def generate_summary_text(self, df: pd.DataFrame, metadata: Dict) -> str:
        """Generate text summary for vector database."""
        if df.empty:
            return "Empty dataset"
            
        summary = f"""
        ARGO Float Platform: {metadata['platform_number']}
        Location: Latitude {metadata['lat_range'][0]:.2f} to {metadata['lat_range'][1]:.2f}, 
                  Longitude {metadata['lon_range'][0]:.2f} to {metadata['lon_range'][1]:.2f}
        Time Period: {metadata['time_range'][0]} to {metadata['time_range'][1]}
        Depth Range: {metadata['depth_range'][0]:.1f} to {metadata['depth_range'][1]:.1f} meters
        Number of Profiles: {metadata['num_profiles']}
        Total Measurements: {metadata['num_measurements']}
        
        Temperature Range: {df['temperature'].min():.2f} to {df['temperature'].max():.2f} °C
        Salinity Range: {df['salinity'].min():.2f} to {df['salinity'].max():.2f} PSU
        """
        
        return summary.strip()


def create_sample_data() -> pd.DataFrame:
    """Create sample ARGO-like data for testing."""
    np.random.seed(42)
    
    # Generate sample profiles
    n_profiles = 5
    n_depths = 50
    
    data = []
    for prof in range(n_profiles):
        lat = np.random.uniform(-20, 20)  # Indian Ocean latitudes
        lon = np.random.uniform(40, 100)   # Indian Ocean longitudes
        time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=prof*30)
        
        depths = np.linspace(0, 2000, n_depths)
        
        for i, depth in enumerate(depths):
            # Simulate realistic ocean profiles
            temp = 28 - (depth/100) + np.random.normal(0, 0.5)  # Temperature decreases with depth
            sal = 35 + (depth/1000) + np.random.normal(0, 0.1)   # Salinity varies slightly
            
            data.append({
                'platform_number': f'SAMPLE_{prof:03d}',
                'profile_number': prof,
                'latitude': lat,
                'longitude': lon,
                'time': time,
                'pressure': depth,
                'depth': depth,
                'temperature': temp,
                'salinity': sal
            })
    
    return pd.DataFrame(data)
