"""
FloatChat MVP - Enhanced AI-Powered ARGO Data Explorer
Optimized for Indian Ocean data with biogeochemical parameters
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import optimized modules
from src.data_ingestion_test import ARGODataIngester
from src.database_test import DatabaseManager
from src.llm_processor_test import QueryProcessor

# Configure Streamlit page
st.set_page_config(
    page_title="FloatChat - Indian Ocean ARGO Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .query-suggestion {
        background: #e3f2fd;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        border-left: 3px solid #1976d2;
    }
    .status-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 4px;
        padding: 0.75rem;
        color: #155724;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = QueryProcessor(st.session_state.db_manager)
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = pd.DataFrame()
    if 'last_components' not in st.session_state:
        st.session_state.last_components = {}

def load_csv_data(uploaded_file):
    """Load and process CSV data."""
    with st.spinner("Processing Indian Ocean ARGO CSV data..."):
        try:
            # Save uploaded file
            uploads_dir = Path('data/uploads')
            uploads_dir.mkdir(parents=True, exist_ok=True)
            file_path = uploads_dir / uploaded_file.name
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the CSV file
            ingester = ARGODataIngester()
            df, metadata = ingester.process_csv_file(str(file_path))
            
            if df.empty:
                st.error("No valid data found in the CSV file.")
                return False
            
            # Insert into database
            st.session_state.db_manager.insert_profile_data(df, metadata)
            
            # Add to vector database
            summary_text = ingester.generate_summary_text(df, metadata)
            st.session_state.db_manager.add_to_vector_db(
                doc_id=f"csv_{uploaded_file.name}",
                text=summary_text,
                metadata=metadata
            )
            
            st.session_state.csv_data = df
            st.session_state.data_loaded = True
            
            st.success(f"Successfully processed {len(df)} measurements from {uploaded_file.name}")
            
            # Show data summary
            st.info(f"""
            **Data Summary:**
            - Records: {len(df):,}
            - Regions: {', '.join(df['region'].unique()) if 'region' in df.columns else 'N/A'}
            - Time range: {df['time'].min().strftime('%Y-%m-%d') if 'time' in df.columns else 'N/A'} to {df['time'].max().strftime('%Y-%m-%d') if 'time' in df.columns else 'N/A'}
            - Depth range: {df['depth'].min():.1f} to {df['depth'].max():.1f} m
            """)
            
            return True
            
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            return False

def create_enhanced_profile_plot(df: pd.DataFrame):
    """Create enhanced vertical profile plots with BGC parameters."""
    if df.empty:
        return None
    
    # Determine available variables for plotting
    plot_vars = []
    var_configs = {
        'temperature': {'title': 'Temperature (°C)', 'color': 'red'},
        'salinity': {'title': 'Salinity (PSU)', 'color': 'blue'},
        'dissolved_oxygen': {'title': 'Dissolved Oxygen (µmol/kg)', 'color': 'green'},
        'chlorophyll': {'title': 'Chlorophyll (mg/m³)', 'color': 'darkgreen'},
        'nitrate': {'title': 'Nitrate (µmol/kg)', 'color': 'purple'},
        'ph': {'title': 'pH', 'color': 'orange'}
    }
    
    for var, config in var_configs.items():
        if var in df.columns and not df[var].isna().all():
            plot_vars.append((var, config))
    
    if not plot_vars:
        return None
    
    # Create subplots dynamically based on available variables
    n_cols = min(3, len(plot_vars))
    n_rows = (len(plot_vars) + n_cols - 1) // n_cols
    
    subplot_titles = [config['title'] for _, config in plot_vars]
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )
    
    # Add traces for each variable
    for idx, (var, config) in enumerate(plot_vars):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        # Group by region if available, otherwise by platform
        group_col = 'region' if 'region' in df.columns else 'platform_number'
        
        for group in df[group_col].unique()[:5]:  # Limit to 5 groups for clarity
            group_df = df[df[group_col] == group]
            if not group_df[var].isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=group_df[var],
                        y=-group_df['depth'],  # Negative for depth
                        mode='lines+markers',
                        name=f'{group}',
                        line=dict(color=config['color']),
                        showlegend=(idx == 0),  # Show legend only for first subplot
                        legendgroup=group
                    ),
                    row=row, col=col
                )
    
    # Update layout
    fig.update_xaxes(title_text="Value")
    fig.update_yaxes(title_text="Depth (m)")
    fig.update_layout(
        height=300 * n_rows,
        title_text="Indian Ocean ARGO Profiles",
        showlegend=True
    )
    
    return fig

def create_enhanced_map(df: pd.DataFrame):
    """Create enhanced interactive map with BGC data coloring."""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    # Aggregate data by location
    location_data = df.groupby(['latitude', 'longitude']).agg({
        'temperature': 'mean',
        'dissolved_oxygen': 'mean',
        'chlorophyll': 'mean',
        'platform_number': 'first',
        'region': 'first'
    }).reset_index()
    
    center_lat = location_data['latitude'].mean()
    center_lon = location_data['longitude'].mean()
    
    # Create map with different base layers
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles=None
    )
    
    # Add multiple tile layers
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB Positron').add_to(m)
    folium.TileLayer('Stamen Terrain').add_to(m)
    
    # Color code by dissolved oxygen if available, otherwise temperature
    if 'dissolved_oxygen' in location_data.columns and not location_data['dissolved_oxygen'].isna().all():
        color_var = 'dissolved_oxygen'
        colormap = folium.LinearColormap(['red', 'yellow', 'green'], 
                                       vmin=location_data[color_var].min(),
                                       vmax=location_data[color_var].max())
        colormap.caption = 'Dissolved Oxygen (µmol/kg)'
    else:
        color_var = 'temperature'
        colormap = folium.LinearColormap(['blue', 'green', 'yellow', 'red'],
                                       vmin=location_data[color_var].min(),
                                       vmax=location_data[color_var].max())
        colormap.caption = 'Temperature (°C)'
    
    # Add markers
    for _, row in location_data.iterrows():
        popup_text = f"""
        <b>Platform:</b> {row['platform_number']}<br>
        <b>Region:</b> {row.get('region', 'N/A')}<br>
        <b>Location:</b> {row['latitude']:.2f}°N, {row['longitude']:.2f}°E<br>
        <b>Temperature:</b> {row['temperature']:.2f}°C<br>
        """
        
        if not pd.isna(row.get('dissolved_oxygen')):
            popup_text += f"<b>Oxygen:</b> {row['dissolved_oxygen']:.1f} µmol/kg<br>"
        
        if not pd.isna(row.get('chlorophyll')):
            popup_text += f"<b>Chlorophyll:</b> {row['chlorophyll']:.3f} mg/m³<br>"
        
        color = colormap(row[color_var])
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            weight=1,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)
    
    # Add colormap to map
    colormap.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_bgc_time_series(df: pd.DataFrame, variable: str = 'dissolved_oxygen'):
    """Create time series plot for biogeochemical variables."""
    if df.empty or variable not in df.columns:
        return None
    
    # Filter surface data (0-50m) for better time series visualization
    surface_data = df[df['depth'] <= 50].copy()
    
    if surface_data.empty:
        return None
    
    surface_data['time'] = pd.to_datetime(surface_data['time'])
    
    # Group by region if available
    group_col = 'region' if 'region' in surface_data.columns else 'platform_number'
    
    fig = px.line(
        surface_data, 
        x='time', 
        y=variable,
        color=group_col,
        title=f'Surface {variable.replace("_", " ").title()} Time Series',
        labels={
            'time': 'Date',
            variable: f'{variable.replace("_", " ").title()} ({get_unit(variable)})'
        }
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title=f'{variable.replace("_", " ").title()} ({get_unit(variable)})'
    )
    
    return fig

def get_unit(variable: str) -> str:
    """Get unit for variable."""
    units = {
        'temperature': '°C',
        'salinity': 'PSU',
        'dissolved_oxygen': 'µmol/kg',
        'chlorophyll': 'mg/m³',
        'nitrate': 'µmol/kg',
        'phosphate': 'µmol/kg',
        'ph': ''
    }
    return units.get(variable, '')

def display_data_quality_metrics(df: pd.DataFrame):
    """Display data quality and coverage metrics."""
    if df.empty:
        return
    
    st.subheader("📊 Data Quality & Coverage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Measurements", f"{len(df):,}")
        st.metric("Time Span", f"{(df['time'].max() - df['time'].min()).days} days" if 'time' in df.columns else "N/A")
    
    with col2:
        if 'region' in df.columns:
            st.metric("Regions Covered", df['region'].nunique())
        st.metric("Depth Range", f"{df['depth'].min():.0f} - {df['depth'].max():.0f} m" if 'depth' in df.columns else "N/A")
    
    with col3:
        if 'platform_number' in df.columns:
            st.metric("ARGO Platforms", df['platform_number'].nunique())
        
        # Calculate BGC data coverage
        bgc_vars = ['dissolved_oxygen', 'chlorophyll', 'nitrate', 'phosphate', 'ph']
        bgc_coverage = sum(1 for var in bgc_vars if var in df.columns and not df[var].isna().all())
        st.metric("BGC Parameters", f"{bgc_coverage}/{len(bgc_vars)}")
    
    # Data completeness table
    completeness_data = []
    key_vars = ['temperature', 'salinity', 'dissolved_oxygen', 'chlorophyll', 'nitrate', 'ph']
    
    for var in key_vars:
        if var in df.columns:
            completeness = (1 - df[var].isna().sum() / len(df)) * 100
            completeness_data.append({
                'Variable': var.replace('_', ' ').title(),
                'Completeness (%)': f"{completeness:.1f}%",
                'Records': f"{(~df[var].isna()).sum():,}"
            })
    
    if completeness_data:
        st.subheader("Data Completeness")
        st.dataframe(pd.DataFrame(completeness_data), use_container_width=True)

def main():
    """Enhanced main application function."""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌊 FloatChat - Indian Ocean ARGO Explorer</h1>
        <p>AI-Powered Conversational Interface for Biogeochemical Ocean Data Discovery</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Data Management")
        
        # CSV Upload Section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Indian Ocean ARGO CSV",
            type=['csv'],
            help="Upload your Indian Ocean ARGO dataset CSV file"
        )
        
        if uploaded_file is not None and not st.session_state.data_loaded:
            if st.button("Process CSV Data", type="primary"):
                success = load_csv_data(uploaded_file)
                if success:
                    st.rerun()
        
        if st.session_state.data_loaded:
            st.markdown('<div class="status-success">✅ Data loaded successfully</div>', unsafe_allow_html=True)
            
            # Display enhanced database summary
            summary = st.session_state.db_manager.get_data_summary()
            
            st.subheader("📈 Database Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", f"{summary['total_records']:,}")
                st.metric("Regions", summary['total_regions'])
            with col2:
                st.metric("Platforms", summary['total_platforms'])
                st.metric("BGC Records", f"{summary['bgc_coverage']['oxygen_records']:,}")
            
            # Regional summary
            if st.button("Show Regional Summary"):
                regional_df = st.session_state.db_manager.get_regional_summary()
                if not regional_df.empty:
                    st.subheader("🌍 Regional Statistics")
                    st.dataframe(regional_df, use_container_width=True)
        
        # # Enhanced query suggestions
        # st.divider()
        # st.subheader("💡 Query Categories")
        
        # query_categories = st.session_state.query_processor.get_query_suggestions_by_category()
        
        # for category, queries in query_categories.items():
        #     with st.expander(category):
        #         for query in queries:
        #             if st.button(query, key=f"cat_{category}_{query[:20]}"):
        #                 st.session_state.messages.append({"role": "user", "content": query})
        #                 st.rerun()
        # Enhanced query suggestions
        st.divider()
        st.subheader("💡 Query Categories")
        
        # Fallback query suggestions if the method doesn't exist
        try:
            if hasattr(st.session_state.query_processor, 'get_query_suggestions_by_category'):
                query_categories = st.session_state.query_processor.get_query_suggestions_by_category()
                
                for category, queries in query_categories.items():
                    with st.expander(category):
                        for query in queries:
                            if st.button(query, key=f"cat_{category}_{query[:20]}"):
                                st.session_state.messages.append({"role": "user", "content": query})
                                st.rerun()
            else:
                raise AttributeError("Method not found")
                
        except (AttributeError, Exception):
            # Fallback to simple query suggestions
            query_categories = {
                "Physical Oceanography": [
                    "Show temperature profiles in the Arabian Sea",
                    "Compare salinity between Bay of Bengal and Arabian Sea",
                    "Find deep water salinity below 1000m"
                ],
                "Biogeochemistry": [
                    "Show oxygen minimum zones in Indian Ocean", 
                    "Find chlorophyll distribution patterns",
                    "Analyze nutrient concentrations in surface waters"
                ],
                "Regional Studies": [
                    "Compare monsoon impacts on ocean properties",
                    "Show seasonal temperature variations",
                    "Find measurements near Sri Lanka"
                ]
            }
            
            for category, queries in query_categories.items():
                with st.expander(category):
                    for query in queries:
                        if st.button(query, key=f"fallback_{category}_{query[:15]}"):
                            st.session_state.messages.append({"role": "user", "content": query})
                            st.rerun()
            
    # Main content area with enhanced layout
    if not st.session_state.data_loaded:
        st.info("👆 Please upload your Indian Ocean ARGO CSV data to get started!")
        
        # Show sample query categories even without data
        st.subheader("🔍 Example Query Categories")
        sample_categories = {
            "Physical Oceanography": ["Temperature profiles", "Salinity distributions", "Mixed layer analysis"],
            "Biogeochemistry": ["Oxygen minimum zones", "Chlorophyll patterns", "Nutrient cycles"],
            "Regional Studies": ["Arabian Sea vs Bay of Bengal", "Monsoon impacts", "Island waters"],
            "Climate Analysis": ["Seasonal variations", "Long-term trends", "Extreme events"]
        }
        
        cols = st.columns(2)
        for idx, (category, examples) in enumerate(sample_categories.items()):
            with cols[idx % 2]:
                st.markdown(f"**{category}**")
                for example in examples:
                    st.markdown(f"• {example}")
        return
    
    # Main interface with data loaded
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
    
    with col2:
        st.header("📊 Visualizations & Analysis")
        
        # Display visualizations if we have results
        if not st.session_state.last_results.empty:
            df = st.session_state.last_results
            components = st.session_state.last_components
            
            # Tabs for different visualization types
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Data", "📈 Profiles", "🗺️ Map", "📊 Analysis"])
            
            with tab1:
                st.subheader("Data Preview")
                st.dataframe(df.head(200), use_container_width=True)
                
                # Data quality metrics
                display_data_quality_metrics(df)
            
            with tab2:
                st.subheader("Vertical Profiles")
                profile_fig = create_enhanced_profile_plot(df)
                if profile_fig:
                    st.plotly_chart(profile_fig, use_container_width=True)
                else:
                    st.info("No profile data available for visualization.")
            
            with tab3:
                st.subheader("Geographic Distribution")
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    float_map = create_enhanced_map(df)
                    if float_map:
                        st_folium(float_map, height=500, width=None)
                else:
                    st.info("No geographic data available.")
            
            with tab4:
                st.subheader("Time Series Analysis")
                
                # Variable selector for time series
                available_bgc_vars = [col for col in ['dissolved_oxygen', 'chlorophyll', 'nitrate', 'ph'] 
                                    if col in df.columns and not df[col].isna().all()]
                
                if available_bgc_vars:
                    selected_var = st.selectbox("Select variable for time series:", available_bgc_vars)
                    ts_fig = create_bgc_time_series(df, selected_var)
                    if ts_fig:
                        st.plotly_chart(ts_fig, use_container_width=True)
                else:
                    st.info("No time series data available.")
        else:
            st.info("💡 Ask a question to see visualizations and analysis here!")
    
    # Chat input (always at bottom)
    prompt = st.chat_input("Ask about Indian Ocean ARGO data (e.g., 'Show oxygen levels in Arabian Sea during monsoon')")
    
    if prompt and st.session_state.query_processor:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process query with enhanced processor
        with st.spinner("Analyzing your query..."):
            try:
                results_df, response, components = st.session_state.query_processor.process_enhanced_query(prompt)
            except AttributeError:
                # Fallback to basic processing if enhanced method doesn't exist
                results_df, response, components = st.session_state.query_processor.process_query(prompt)
            except Exception as e:
                results_df = pd.DataFrame()
                response = f"I encountered an error processing your query: {str(e)}"
                components = {}
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Store results for visualization
            st.session_state.last_results = results_df
            st.session_state.last_components = components
        
        st.rerun()


if __name__ == "__main__":
    main()