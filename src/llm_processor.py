"""
LLM Query Processor Module
Handles natural language to SQL conversion and RAG-based responses
"""

import re
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processes natural language queries into SQL and generates responses."""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.query_patterns = self._init_query_patterns()
        
    def _init_query_patterns(self) -> Dict:
        """Initialize common query patterns for text-to-SQL conversion."""
        return {
            'location': {
                'patterns': [
                    r'near\s+([\-\d.]+),?\s*([\-\d.]+)',
                    r'at\s+latitude\s+([\-\d.]+)\s+longitude\s+([\-\d.]+)',
                    r'around\s+([\-\d.]+)\s+lat\s+([\-\d.]+)\s+lon',
                    r'near the equator',
                    r'in the (.*?) (ocean|sea)',
                ],
                'sql_template': "latitude BETWEEN {lat_min} AND {lat_max} AND longitude BETWEEN {lon_min} AND {lon_max}"
            },
            'time': {
                'patterns': [
                    r'in\s+(\w+)\s+(\d{4})',
                    r'from\s+([\d\-/]+)\s+to\s+([\d\-/]+)',
                    r'last\s+(\d+)\s+(days?|months?|years?)',
                    r'on\s+([\d\-/]+)',
                    r'recent',
                ],
                'sql_template': "time BETWEEN '{time_start}' AND '{time_end}'"
            },
            'depth': {
                'patterns': [
                    r'at\s+(\d+)\s*m(?:eters)?',
                    r'depth\s+(\d+)',
                    r'between\s+(\d+)\s+and\s+(\d+)\s*m',
                    r'surface',
                    r'deep',
                ],
                'sql_template': "depth BETWEEN {depth_min} AND {depth_max}"
            },
            'variable': {
                'patterns': [
                    r'(temperature|temp)',
                    r'(salinity|salt)',
                    r'(pressure|pres)',
                ],
                'columns': {
                    'temperature': 'temperature',
                    'temp': 'temperature',
                    'salinity': 'salinity',
                    'salt': 'salinity',
                    'pressure': 'pressure',
                    'pres': 'pressure'
                }
            }
        }
    
    def parse_natural_query(self, query: str) -> Dict:
        """Parse natural language query into structured components."""
        query_lower = query.lower()
        components = {
            'variables': [],
            'conditions': [],
            'aggregations': [],
            'limit': 1000
        }
        
        # Extract variables
        for pattern in self.query_patterns['variable']['patterns']:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                var_name = self.query_patterns['variable']['columns'].get(match, match)
                if var_name not in components['variables']:
                    components['variables'].append(var_name)
        
        # Extract location conditions
        if 'near the equator' in query_lower:
            components['conditions'].append("latitude BETWEEN -10 AND 10")
        elif 'arabian sea' in query_lower:
            components['conditions'].append("latitude BETWEEN 0 AND 25 AND longitude BETWEEN 50 AND 75")
        elif 'indian ocean' in query_lower:
            components['conditions'].append("latitude BETWEEN -60 AND 30 AND longitude BETWEEN 20 AND 120")
        else:
            # Check for coordinate patterns
            coord_pattern = r'(?:near|at|around)\s+([\-\d.]+)[,\s]+([\-\d.]+)'
            coord_match = re.search(coord_pattern, query_lower)
            if coord_match:
                lat, lon = float(coord_match.group(1)), float(coord_match.group(2))
                components['conditions'].append(
                    f"latitude BETWEEN {lat-5} AND {lat+5} AND longitude BETWEEN {lon-5} AND {lon+5}"
                )
        
        # Extract time conditions
        month_map = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        for month, month_num in month_map.items():
            if month in query_lower:
                year_match = re.search(rf'{month}\s+(\d{{4}})', query_lower)
                if year_match:
                    year = year_match.group(1)
                    components['conditions'].append(
                        f"strftime('%Y-%m', time) = '{year}-{month_num}'"
                    )
                break
        
        # Extract depth conditions
        if 'surface' in query_lower:
            components['conditions'].append("depth <= 10")
        elif 'deep' in query_lower:
            components['conditions'].append("depth >= 1000")
        else:
            depth_match = re.search(r'at\s+(\d+)\s*m', query_lower)
            if depth_match:
                depth = int(depth_match.group(1))
                components['conditions'].append(f"depth BETWEEN {depth-50} AND {depth+50}")
        
        # Extract aggregations
        if 'average' in query_lower or 'mean' in query_lower:
            components['aggregations'].append('AVG')
        elif 'maximum' in query_lower or 'max' in query_lower:
            components['aggregations'].append('MAX')
        elif 'minimum' in query_lower or 'min' in query_lower:
            components['aggregations'].append('MIN')
        
        # Extract comparison requests
        if 'compare' in query_lower:
            components['comparison'] = True
        
        # Extract profile/plot requests
        if 'profile' in query_lower or 'plot' in query_lower or 'show' in query_lower:
            components['visualization'] = True
        
        return components
    
    def generate_sql_query(self, components: Dict) -> str:
        """Generate SQL query from parsed components."""
        # Default columns if none specified
        if not components['variables']:
            components['variables'] = ['temperature', 'salinity', 'pressure', 'depth', 
                                      'latitude', 'longitude', 'time', 'platform_number']
        
        # Build SELECT clause
        select_cols = ', '.join(components['variables'])
        
        # Add aggregations if present
        if components['aggregations']:
            agg_func = components['aggregations'][0]
            select_cols = ', '.join([f"{agg_func}({col}) as {col}_avg" 
                                    for col in components['variables'] 
                                    if col in ['temperature', 'salinity']])
            select_cols += ', COUNT(*) as count'
        
        # Build WHERE clause
        where_clause = ' AND '.join(components['conditions']) if components['conditions'] else '1=1'
        
        # Build final query
        query = f"SELECT {select_cols} FROM profiles WHERE {where_clause}"
        
        # Add GROUP BY for aggregations
        if components['aggregations']:
            query += " GROUP BY platform_number, profile_number"
        
        # Add LIMIT
        query += f" LIMIT {components['limit']}"
        
        return query
    
    def process_query(self, user_query: str) -> Tuple[pd.DataFrame, str, Dict]:
        """Process user query and return results with explanation."""
        try:
            # Parse the natural language query
            components = self.parse_natural_query(user_query)
            
            # Use vector search to find relevant context
            if self.db_manager:
                relevant_docs = self.db_manager.query_vector_db(user_query, n_results=3)
            else:
                relevant_docs = []
            
            # Generate SQL query
            sql_query = self.generate_sql_query(components)
            logger.info(f"Generated SQL: {sql_query}")
            
            # Execute query
            if self.db_manager:
                results_df = self.db_manager.execute_sql(sql_query)
            else:
                results_df = pd.DataFrame()
            
            # Generate response text
            response = self._generate_response(user_query, results_df, components, relevant_docs)
            
            return results_df, response, components
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return pd.DataFrame(), f"I encountered an error processing your query: {str(e)}", {}
    
    def _generate_response(self, query: str, results: pd.DataFrame, 
                          components: Dict, relevant_docs: List) -> str:
        """Generate natural language response based on query results."""
        if results.empty:
            return "No data found matching your query criteria. Try adjusting your search parameters."
        
        response_parts = []
        
        # Summary of results
        response_parts.append(f"Found {len(results)} measurements matching your query.")
        
        # Provide specific insights based on query type
        if 'temperature' in components['variables'] and not results.empty:
            temp_stats = results['temperature'].describe() if 'temperature' in results.columns else None
            if temp_stats is not None:
                response_parts.append(
                    f"Temperature ranges from {temp_stats['min']:.2f}°C to {temp_stats['max']:.2f}°C "
                    f"with an average of {temp_stats['mean']:.2f}°C."
                )
        
        if 'salinity' in components['variables'] and not results.empty:
            if 'salinity' in results.columns:
                sal_stats = results['salinity'].describe()
                response_parts.append(
                    f"Salinity ranges from {sal_stats['min']:.2f} to {sal_stats['max']:.2f} PSU "
                    f"with an average of {sal_stats['mean']:.2f} PSU."
                )
        
        # Add information about platforms
        if 'platform_number' in results.columns:
            platforms = results['platform_number'].unique()
            if len(platforms) <= 5:
                response_parts.append(f"Data from platforms: {', '.join(platforms[:5])}")
            else:
                response_parts.append(f"Data from {len(platforms)} different ARGO floats.")
        
        # Add relevant context from vector search
        if relevant_docs:
            response_parts.append("\nRelated information from the database:")
            for doc in relevant_docs[:2]:
                if 'document' in doc:
                    # Extract key info from document
                    lines = doc['document'].split('\n')
                    relevant_lines = [l.strip() for l in lines if l.strip() and not l.startswith('ARGO')][:2]
                    if relevant_lines:
                        response_parts.append("- " + '. '.join(relevant_lines))
        
        return '\n'.join(response_parts)
    
    def suggest_queries(self) -> List[str]:
        """Suggest example queries for users."""
        return [
            "Show me temperature profiles near the equator in March 2023",
            "What is the average salinity in the Arabian Sea?",
            "Find all measurements at 100m depth",
            "Compare temperature and salinity profiles in the Indian Ocean",
            "Show recent data from the last 30 days",
            "What are the surface temperature measurements?",
            "Find deep water salinity values below 1000m",
            "Show all data near 10.5, 75.3",
        ]
