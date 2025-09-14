"""
Enhanced LLM Query Processor Module
Optimized for Indian Ocean ARGO data with biogeochemical parameters
Handles natural language to SQL conversion and RAG-based responses
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import logging
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryProcessor:
    """Enhanced query processor for Indian Ocean ARGO data with BGC parameters."""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.query_patterns = self._init_enhanced_patterns()
        self.region_mappings = self._init_region_mappings()
        self.variable_mappings = self._init_variable_mappings()
        
    def _init_enhanced_patterns(self) -> Dict:
        """Initialize enhanced query patterns for Indian Ocean data."""
        return {
            'location': {
                'patterns': [
                    # Coordinate patterns
                    r'near\s+([\-\d.]+)[,\s]+([\-\d.]+)',
                    r'at\s+(?:latitude\s+)?([\-\d.]+)[,\s]+(?:longitude\s+)?([\-\d.]+)',
                    # Regional patterns
                    r'in\s+the\s+(arabian\s+sea|bay\s+of\s+bengal|south\s+indian\s+ocean|eastern\s+indian\s+ocean)',
                    r'(arabian\s+sea|bay\s+of\s+bengal|south\s+indian\s+ocean|eastern\s+indian\s+ocean)',
                    r'near\s+(india|sri\s+lanka|maldives|seychelles|madagascar)',
                    r'equatorial\s+indian\s+ocean',
                    r'tropical\s+indian\s+ocean',
                    r'monsoon\s+region',
                ],
                'sql_template': "latitude BETWEEN {lat_min} AND {lat_max} AND longitude BETWEEN {lon_min} AND {lon_max}"
            },
            'time': {
                'patterns': [
                    # Specific dates
                    r'in\s+(\w+)\s+(\d{4})',
                    r'from\s+([\d\-/]+)\s+to\s+([\d\-/]+)',
                    r'on\s+([\d\-/]+)',
                    # Relative time
                    r'last\s+(\d+)\s+(days?|months?|years?)',
                    r'past\s+(\d+)\s+(days?|months?|years?)',
                    r'recent(?:ly)?',
                    # Seasonal patterns
                    r'(monsoon|winter|summer|pre-monsoon|post-monsoon)',
                    r'during\s+(\w+\s+monsoon)',
                ],
                'sql_template': "time BETWEEN '{time_start}' AND '{time_end}'"
            },
            'depth': {
                'patterns': [
                    r'at\s+(\d+)\s*m(?:eters?)?(?:\s+depth)?',
                    r'depth\s+(?:of\s+)?(\d+)(?:\s*m)?',
                    r'between\s+(\d+)\s*(?:m\s+)?and\s+(\d+)\s*m',
                    r'from\s+(\d+)\s*(?:m\s+)?to\s+(\d+)\s*m',
                    r'surface|near\s+surface',
                    r'deep|deep\s+water|below\s+(\d+)\s*m',
                    r'mixed\s+layer',
                    r'thermocline',
                    r'oxygen\s+minimum\s+zone|omz',
                    r'(\d+)-(\d+)\s*m(?:eters?)?',
                ],
                'sql_template': "depth BETWEEN {depth_min} AND {depth_max}"
            },
            'biogeochemical': {
                'patterns': [
                    # Oxygen
                    r'(dissolved\s+)?oxygen|o2|hypoxic|anoxic',
                    r'oxygen\s+minimum\s+zone|omz',
                    # Nutrients
                    r'nitrate|no3|nitrogen',
                    r'phosphate|po4|phosphorus', 
                    r'silicate|si|silicon',
                    # Carbon system
                    r'ph|acidity|alkalinity',
                    r'carbon|co2|pco2|carbonate',
                    # Biological
                    r'chlorophyll|chl|phytoplankton|primary\s+production',
                    r'fluorescence|biomass',
                    r'productivity|biological',
                ],
                'variables': {
                    'oxygen': 'dissolved_oxygen',
                    'o2': 'dissolved_oxygen',
                    'nitrate': 'nitrate',
                    'no3': 'nitrate',
                    'phosphate': 'phosphate',
                    'po4': 'phosphate',
                    'silicate': 'silicate',
                    'si': 'silicate',
                    'ph': 'ph',
                    'alkalinity': 'alkalinity',
                    'carbon': 'pco2',
                    'co2': 'pco2',
                    'pco2': 'pco2',
                    'chlorophyll': 'chlorophyll',
                    'chl': 'chlorophyll',
                    'fluorescence': 'fluorescence'
                }
            },
            'comparison': {
                'patterns': [
                    r'compare|comparison|versus|vs\.|against',
                    r'difference|differ(?:ent)?',
                    r'correlation|relationship|relate',
                    r'trend|pattern|variability',
                ],
                'operations': ['compare', 'correlate', 'trend']
            },
            'aggregation': {
                'patterns': [
                    r'average|mean',
                    r'maximum|max|highest',
                    r'minimum|min|lowest',
                    r'median|middle',
                    r'standard\s+deviation|std|variability',
                    r'range|spread',
                    r'sum|total',
                    r'count|number\s+of',
                ],
                'functions': {
                    'average': 'AVG', 'mean': 'AVG',
                    'maximum': 'MAX', 'max': 'MAX', 'highest': 'MAX',
                    'minimum': 'MIN', 'min': 'MIN', 'lowest': 'MIN',
                    'median': 'MEDIAN',
                    'sum': 'SUM', 'total': 'SUM',
                    'count': 'COUNT', 'number': 'COUNT'
                }
            }
        }
    
    def _init_region_mappings(self) -> Dict:
        """Initialize region coordinate mappings."""
        return {
            'arabian sea': {'lat': (8, 25), 'lon': (50, 75)},
            'bay of bengal': {'lat': (5, 22), 'lon': (78, 100)},
            'south indian ocean': {'lat': (-60, -20), 'lon': (20, 120)},
            'eastern indian ocean': {'lat': (-20, 10), 'lon': (90, 120)},
            'equatorial indian ocean': {'lat': (-10, 10), 'lon': (40, 100)},
            'tropical indian ocean': {'lat': (-25, 25), 'lon': (40, 120)},
            'monsoon region': {'lat': (-10, 25), 'lon': (40, 100)},
            'indian ocean': {'lat': (-60, 30), 'lon': (20, 120)}
        }
    
    def _init_variable_mappings(self) -> Dict:
        """Initialize comprehensive variable mappings."""
        return {
            'physical': {
                'temperature': ['temp', 'sst', 'sea surface temperature'],
                'salinity': ['sal', 'psal', 'salt'],
                'pressure': ['pres', 'press'],
                'density': ['dens', 'sigma'],
                'potential_density': ['pot_density', 'sigma_theta']
            },
            'biogeochemical': {
                'dissolved_oxygen': ['oxygen', 'o2', 'doxy'],
                'nitrate': ['no3', 'nitrogen'],
                'phosphate': ['po4', 'phosphorus'],
                'silicate': ['si', 'silicon', 'silica'],
                'ph': ['acidity', 'hydrogen'],
                'alkalinity': ['alk', 'total_alkalinity'],
                'pco2': ['carbon_dioxide', 'co2', 'carbon']
            },
            'optical': {
                'chlorophyll': ['chl', 'chla', 'phytoplankton'],
                'fluorescence': ['fluor', 'biomass'],
                'turbidity': ['turb'],
                'par': ['photosynthetically_active_radiation', 'light']
            }
        }
    
    def parse_enhanced_query(self, query: str) -> Dict:
        """Enhanced query parsing with BGC parameter support."""
        query_lower = query.lower().strip()
        components = {
            'variables': [],
            'conditions': [],
            'aggregations': [],
            'comparisons': [],
            'time_filters': [],
            'spatial_filters': [],
            'depth_filters': [],
            'limit': 1000,
            'visualization_type': None,
            'analysis_type': 'basic'
        }
        
        # Extract variables from all categories
        all_variables = set()
        for category, var_dict in self.variable_mappings.items():
            for standard_name, aliases in var_dict.items():
                # Check for standard name
                if standard_name in query_lower:
                    all_variables.add(standard_name)
                # Check for aliases
                for alias in aliases:
                    if alias in query_lower:
                        all_variables.add(standard_name)
        
        # Add BGC variables from patterns
        for var_pattern in self.query_patterns['biogeochemical']['patterns']:
            matches = re.findall(var_pattern, query_lower)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match).strip()
                    # Map to standard variable name
                    standard_var = self.query_patterns['biogeochemical']['variables'].get(
                        match.replace(' ', '_'), None
                    )
                    if standard_var:
                        all_variables.add(standard_var)
        
        components['variables'] = list(all_variables)
        
        # Parse spatial conditions
        components['spatial_filters'] = self._parse_spatial_conditions(query_lower)
        
        # Parse temporal conditions  
        components['time_filters'] = self._parse_temporal_conditions(query_lower)
        
        # Parse depth conditions
        components['depth_filters'] = self._parse_depth_conditions(query_lower)
        
        # Parse aggregations
        for pattern, func_dict in [(self.query_patterns['aggregation']['patterns'], 
                                   self.query_patterns['aggregation']['functions'])]:
            for agg_pattern in pattern:
                if re.search(agg_pattern, query_lower):
                    for keyword, sql_func in func_dict.items():
                        if keyword in query_lower:
                            components['aggregations'].append(sql_func)
                            break
        
        # Detect comparison/analysis requests
        if any(re.search(pattern, query_lower) for pattern in self.query_patterns['comparison']['patterns']):
            components['analysis_type'] = 'comparison'
            components['comparisons'] = self._extract_comparison_variables(query_lower)
        
        # Detect visualization requests
        viz_keywords = {
            'profile': 'profile',
            'plot': 'time_series', 
            'map': 'spatial',
            'chart': 'time_series',
            'graph': 'time_series',
            'show': 'profile',
            'visualize': 'profile'
        }
        
        for keyword, viz_type in viz_keywords.items():
            if keyword in query_lower:
                components['visualization_type'] = viz_type
                break
        
        return components
    
    def _parse_spatial_conditions(self, query: str) -> List[str]:
        """Parse spatial/regional conditions."""
        conditions = []
        
        # Check for specific regions
        for region, bounds in self.region_mappings.items():
            if region in query:
                conditions.append(
                    f"region = '{region.title()}' OR "
                    f"(latitude BETWEEN {bounds['lat'][0]} AND {bounds['lat'][1]} AND "
                    f"longitude BETWEEN {bounds['lon'][0]} AND {bounds['lon'][1]})"
                )
        
        # Check for coordinate patterns
        coord_patterns = [
            r'near\s+([\-\d.]+)[,\s]+([\-\d.]+)',
            r'at\s+(?:latitude\s+)?([\-\d.]+)[,\s]+(?:longitude\s+)?([\-\d.]+)'
        ]
        
        for pattern in coord_patterns:
            match = re.search(pattern, query)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
                # Add 2-degree buffer around point
                conditions.append(
                    f"latitude BETWEEN {lat-2} AND {lat+2} AND "
                    f"longitude BETWEEN {lon-2} AND {lon+2}"
                )
        
        return conditions
    
    def _parse_temporal_conditions(self, query: str) -> List[str]:
        """Parse temporal conditions with monsoon season support."""
        conditions = []
        
        # Monsoon season mappings
        seasons = {
            'southwest monsoon': ("06-01", "09-30"),
            'northeast monsoon': ("10-01", "02-28"),
            'pre-monsoon': ("03-01", "05-31"),
            'post-monsoon': ("10-01", "12-31"),
            'winter': ("12-01", "02-28"),
            'summer': ("03-01", "05-31")
        }
        
        for season, (start, end) in seasons.items():
            if season in query:
                conditions.append(f"strftime('%m-%d', time) BETWEEN '{start}' AND '{end}'")
        
        # Specific month/year patterns
        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        for month, month_num in months.items():
            pattern = rf'{month}\s+(\d{{4}})'
            match = re.search(pattern, query)
            if match:
                year = match.group(1)
                conditions.append(f"strftime('%Y-%m', time) = '{year}-{month_num}'")
        
        # Recent data patterns
        recent_patterns = [
            (r'last\s+(\d+)\s+days?', 'days'),
            (r'past\s+(\d+)\s+months?', 'months'),
            (r'recent', 'months')
        ]
        
        for pattern, unit in recent_patterns:
            match = re.search(pattern, query)
            if match:
                if pattern == r'recent':
                    days = 30
                else:
                    num = int(match.group(1))
                    days = num if unit == 'days' else num * 30
                
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                conditions.append(f"time >= '{start_date}'")
        
        return conditions
    
    def _parse_depth_conditions(self, query: str) -> List[str]:
        """Parse depth conditions with oceanographic zone support."""
        conditions = []
        
        # Oceanographic zones
        zones = {
            'surface': (0, 10),
            'mixed layer': (0, 100),
            'thermocline': (100, 500),
            'oxygen minimum zone': (200, 1000),
            'omz': (200, 1000),
            'deep': (1000, 5000),
            'abyssal': (3000, 6000)
        }
        
        for zone, (min_depth, max_depth) in zones.items():
            if zone in query:
                conditions.append(f"depth BETWEEN {min_depth} AND {max_depth}")
        
        # Specific depth patterns
        depth_patterns = [
            (r'at\s+(\d+)\s*m', lambda d: f"depth BETWEEN {int(d)-25} AND {int(d)+25}"),
            (r'between\s+(\d+)\s*(?:m\s+)?and\s+(\d+)\s*m', lambda d1, d2: f"depth BETWEEN {d1} AND {d2}"),
            (r'(\d+)-(\d+)\s*m', lambda d1, d2: f"depth BETWEEN {d1} AND {d2}"),
            (r'below\s+(\d+)\s*m', lambda d: f"depth > {d}")
        ]
        
        for pattern, condition_func in depth_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                try:
                    if len(match.groups()) == 1:
                        conditions.append(condition_func(match.group(1)))
                    else:
                        conditions.append(condition_func(match.group(1), match.group(2)))
                except:
                    continue
        
        return conditions
    
    def _extract_comparison_variables(self, query: str) -> List[str]:
        """Extract variables for comparison analysis."""
        # Common comparison pairs in oceanography
        comparison_pairs = [
            ['temperature', 'salinity'],
            ['dissolved_oxygen', 'depth'],
            ['chlorophyll', 'nitrate'],
            ['temperature', 'dissolved_oxygen'],
            ['salinity', 'density']
        ]
        
        query_vars = set()
        for category in self.variable_mappings.values():
            for var, aliases in category.items():
                if var in query or any(alias in query for alias in aliases):
                    query_vars.add(var)
        
        # Find best matching comparison pair
        for pair in comparison_pairs:
            if all(var in query_vars for var in pair):
                return pair
        
        # Return any available variables for comparison
        return list(query_vars)[:2] if len(query_vars) >= 2 else list(query_vars)
    
    def generate_enhanced_sql(self, components: Dict) -> str:
        """Generate optimized SQL query from parsed components."""
        # Determine columns to select
        if not components['variables']:
            # Default comprehensive selection
            select_columns = [
                'platform_number', 'profile_number', 'time', 'latitude', 'longitude',
                'depth', 'temperature', 'salinity', 'pressure', 'region'
            ]
            # Add available BGC variables
            bgc_vars = ['dissolved_oxygen', 'chlorophyll', 'nitrate', 'phosphate', 'ph']
            select_columns.extend(bgc_vars)
        else:
            select_columns = ['platform_number', 'time', 'latitude', 'longitude', 'depth']
            select_columns.extend(components['variables'])
        
        # Build SELECT clause with aggregations
        if components['aggregations']:
            agg_func = components['aggregations'][0]
            numeric_cols = [col for col in select_columns if col in [
                'temperature', 'salinity', 'dissolved_oxygen', 'chlorophyll', 
                'nitrate', 'phosphate', 'ph', 'pressure', 'depth'
            ]]
            
            select_parts = [f"{agg_func}({col}) as avg_{col}" for col in numeric_cols]
            select_parts.extend(['region', 'COUNT(*) as measurement_count'])
            select_clause = ', '.join(select_parts)
        else:
            # Remove duplicates and ensure core columns
            unique_columns = []
            seen = set()
            for col in select_columns:
                if col not in seen:
                    unique_columns.append(col)
                    seen.add(col)
            select_clause = ', '.join(unique_columns)
        
        # Build WHERE clause
        all_conditions = []
        all_conditions.extend(components['spatial_filters'])
        all_conditions.extend(components['time_filters']) 
        all_conditions.extend(components['depth_filters'])
        
        # Add data quality filters
        quality_conditions = []
        if 'temperature' in components['variables']:
            quality_conditions.append("temperature BETWEEN -5 AND 40")
        if 'salinity' in components['variables']:
            quality_conditions.append("salinity BETWEEN 25 AND 42")
        if 'dissolved_oxygen' in components['variables']:
            quality_conditions.append("dissolved_oxygen >= 0")
        
        all_conditions.extend(quality_conditions)
        
        where_clause = ' AND '.join(all_conditions) if all_conditions else '1=1'
        
        # Build complete query
        query = f"SELECT {select_clause} FROM profiles WHERE {where_clause}"
        
        # Add GROUP BY for aggregations
        if components['aggregations']:
            query += " GROUP BY region"
        
        # Add ORDER BY
        if components['analysis_type'] == 'comparison':
            query += " ORDER BY time DESC, depth ASC"
        else:
            query += " ORDER BY time DESC"
        
        # Add LIMIT
        query += f" LIMIT {components['limit']}"
        
        return query
    
    def process_enhanced_query(self, user_query: str) -> Tuple[pd.DataFrame, str, Dict]:
        """Process user query with enhanced capabilities."""
        try:
            # Parse the natural language query
            components = self.parse_enhanced_query(user_query)
            logger.info(f"Parsed components: {components}")
            
            # Use vector search for context
            relevant_docs = []
            if self.db_manager:
                relevant_docs = self.db_manager.query_vector_db(user_query, n_results=3)
            
            # Generate and execute SQL
            sql_query = self.generate_enhanced_sql(components)
            logger.info(f"Generated SQL: {sql_query}")
            
            if self.db_manager:
                results_df = self.db_manager.execute_sql(sql_query)
            else:
                results_df = pd.DataFrame()
            
            # Generate enhanced response
            response = self._generate_enhanced_response(user_query, results_df, components, relevant_docs)
            
            return results_df, response, components
            
        except Exception as e:
            logger.error(f"Error processing enhanced query: {e}")
            return pd.DataFrame(), f"Error processing query: {str(e)}", {}
    
    def _generate_enhanced_response(self, query: str, results: pd.DataFrame, 
                                  components: Dict, relevant_docs: List) -> str:
        """Generate comprehensive natural language response."""
        if results.empty:
            return ("No data found matching your criteria. This might be due to "
                   "specific parameter combinations or temporal/spatial constraints. "
                   "Try broadening your search parameters.")
        
        response_parts = []
        
        # Basic summary
        response_parts.append(f"Found {len(results)} measurements matching your query.")
        
        # Regional distribution
        if 'region' in results.columns:
            region_counts = results['region'].value_counts()
            if len(region_counts) > 1:
                top_regions = region_counts.head(3)
                region_text = ", ".join([f"{region} ({count} measurements)" 
                                       for region, count in top_regions.items()])
                response_parts.append(f"Data distribution: {region_text}")
        
        # Variable-specific insights
        self._add_variable_insights(response_parts, results, components)
        
        # Temporal insights
        if 'time' in results.columns:
            time_span = (pd.to_datetime(results['time']).max() - 
                        pd.to_datetime(results['time']).min()).days
            if time_span > 0:
                response_parts.append(f"Data spans {time_span} days.")
        
        # Platform information
        if 'platform_number' in results.columns:
            platform_count = results['platform_number'].nunique()
            response_parts.append(f"Data from {platform_count} ARGO float(s).")
        
        # Add context from vector database
        if relevant_docs:
            context_info = self._extract_relevant_context(relevant_docs, components['variables'])
            if context_info:
                response_parts.append(f"\nContext: {context_info}")
        
        return '\n'.join(response_parts)
    
    def _add_variable_insights(self, response_parts: List[str], results: pd.DataFrame, components: Dict):
        """Add variable-specific statistical insights."""
        variables_to_analyze = components['variables'] if components['variables'] else []
        
        # Priority variables for analysis
        priority_vars = ['temperature', 'salinity', 'dissolved_oxygen', 'chlorophyll', 'nitrate']
        
        for var in priority_vars:
            if var in results.columns and not results[var].isna().all():
                stats = results[var].describe()
                unit = self._get_variable_unit(var)
                
                response_parts.append(
                    f"{var.title()}: {stats['min']:.2f} to {stats['max']:.2f} {unit} "
                    f"(mean: {stats['mean']:.2f} {unit}, std: {stats['std']:.2f})"
                )
    
    def _get_variable_unit(self, variable: str) -> str:
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
            'ph': '',
            'chlorophyll': 'mg/m³',
            'density': 'kg/m³',
            'potential_density': 'kg/m³',
            'alkalinity': 'µmol/kg',
            'pco2': 'µatm',
            'fluorescence': 'mg/m³',
            'turbidity': 'NTU',
            'par': 'µmol photons/m²/s'
        }
        return units.get(variable, '')
    
    def _extract_relevant_context(self, docs: List[Dict], query_variables: List[str]) -> str:
        """Extract relevant context from vector search results."""
        if not docs:
            return ""
        
        context_parts = []
        for doc in docs[:2]:  # Use top 2 most relevant documents
            if 'document' in doc:
                # Extract key information that relates to query variables
                doc_text = doc['document']
                relevant_sentences = []
                
                for sentence in doc_text.split('\n'):
                    sentence = sentence.strip()
                    if not sentence or sentence.startswith('ARGO'):
                        continue
                    
                    # Check if sentence contains information about queried variables
                    sentence_lower = sentence.lower()
                    if any(var in sentence_lower for var in query_variables):
                        relevant_sentences.append(sentence)
                    elif any(keyword in sentence_lower for keyword in 
                           ['range', 'average', 'temperature', 'salinity', 'ocean']):
                        relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    context_parts.extend(relevant_sentences[:2])  # Max 2 sentences per doc
        
        return '. '.join(context_parts) if context_parts else ""
    
    def suggest_enhanced_queries(self) -> List[str]:
        """Suggest enhanced example queries for Indian Ocean data."""
        return [
            # Physical oceanography
            "Show temperature and salinity profiles in the Arabian Sea during monsoon season",
            "Compare surface temperature between Bay of Bengal and Arabian Sea",
            "Find deep water salinity below 1000m in the South Indian Ocean",
            
            # Biogeochemical queries
            "What is the oxygen minimum zone depth in the Arabian Sea?",
            "Show chlorophyll distribution in the equatorial Indian Ocean",
            "Find areas with low oxygen and high nitrate concentrations",
            
            # Temporal analysis
            "Compare pre-monsoon and post-monsoon temperature profiles",
            "Show seasonal variation in chlorophyll near Sri Lanka",
            "Find recent measurements from the last 30 days",
            
            # Regional comparisons
            "Compare biogeochemical conditions between different Indian Ocean regions",
            "Show the relationship between temperature and dissolved oxygen",
            "Find correlations between chlorophyll and nutrient concentrations",
            
            # Depth-specific queries
            "Analyze mixed layer properties in tropical Indian Ocean",
            "Show thermocline characteristics during different seasons",
            "Find surface ocean acidification trends"
        ]
    
    def get_query_suggestions_by_category(self) -> Dict[str, List[str]]:
        """Get query suggestions organized by category."""
        return {
            'Physical Oceanography': [
                "Temperature and salinity in the Arabian Sea",
                "Deep water formation in South Indian Ocean", 
                "Mixed layer depth variations",
                "Seasonal temperature patterns"
            ],
            'Biogeochemistry': [
                "Oxygen minimum zones in Indian Ocean",
                "Chlorophyll distribution patterns",
                "Nutrient cycling in tropical regions",
                "Ocean acidification indicators"
            ],
            'Regional Studies': [
                "Bay of Bengal vs Arabian Sea comparison",
                "Equatorial Indian Ocean dynamics", 
                "Monsoon impacts on ocean properties",
                "Island nation surrounding waters"
            ],
            'Climate & Variability': [
                "Seasonal monsoon effects",
                "Inter-annual variability patterns",
                "Long-term trend analysis",
                "Extreme event characterization"
            ]
        }