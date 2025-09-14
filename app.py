"""
FloatChat MVP - AI-Powered ARGO Data Explorer
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

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_ingestion import ARGODataIngester, create_sample_data
from database import DatabaseManager
from llm_processor import QueryProcessor

# Configure Streamlit page
st.set_page_config(
    page_title="FloatChat - ARGO Ocean Data Explorer",
    page_icon="🌊",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if 'query_processor' not in st.session_state:
    st.session_state.query_processor = QueryProcessor(st.session_state.db_manager)
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def load_sample_data():
    """Load sample data into the database."""
    with st.spinner("Loading sample ARGO data..."):
        # Create sample data
        sample_df = create_sample_data()
        
        # Create metadata
        metadata = {
            'platform_number': 'SAMPLE_001',
            'num_profiles': sample_df['profile_number'].nunique(),
            'num_measurements': len(sample_df),
            'lat_range': [sample_df['latitude'].min(), sample_df['latitude'].max()],
            'lon_range': [sample_df['longitude'].min(), sample_df['longitude'].max()],
            'time_range': [
                sample_df['time'].min().isoformat(),
                sample_df['time'].max().isoformat()
            ],
            'depth_range': [sample_df['depth'].min(), sample_df['depth'].max()],
            'variables': list(sample_df.columns)
        }
        
        # Insert into database
        st.session_state.db_manager.insert_profile_data(sample_df, metadata)
        
        # Add to vector database
        ingester = ARGODataIngester()
        summary_text = ingester.generate_summary_text(sample_df, metadata)
        st.session_state.db_manager.add_to_vector_db(
            doc_id='sample_001',
            text=summary_text,
            metadata=metadata
        )
        
        st.session_state.data_loaded = True
        st.success("Sample data loaded successfully!")


def create_profile_plot(df: pd.DataFrame):
    """Create vertical profile plots for temperature and salinity."""
    if df.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Temperature Profile', 'Salinity Profile'),
        horizontal_spacing=0.15
    )
    
    # Add temperature profile
    if 'temperature' in df.columns and 'depth' in df.columns:
        for platform in df['platform_number'].unique():
            platform_df = df[df['platform_number'] == platform]
            fig.add_trace(
                go.Scatter(
                    x=platform_df['temperature'],
                    y=-platform_df['depth'],  # Negative for depth
                    mode='lines+markers',
                    name=f'{platform} (T)',
                    showlegend=True
                ),
                row=1, col=1
            )
    
    # Add salinity profile
    if 'salinity' in df.columns and 'depth' in df.columns:
        for platform in df['platform_number'].unique():
            platform_df = df[df['platform_number'] == platform]
            fig.add_trace(
                go.Scatter(
                    x=platform_df['salinity'],
                    y=-platform_df['depth'],
                    mode='lines+markers',
                    name=f'{platform} (S)',
                    showlegend=True
                ),
                row=1, col=2
            )
    
    # Update layout
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
    fig.update_yaxes(title_text="Depth (m)", row=1, col=1)
    fig.update_yaxes(title_text="Depth (m)", row=1, col=2)
    
    fig.update_layout(height=600, title_text="Vertical Profiles")
    
    return fig


def create_map(df: pd.DataFrame):
    """Create interactive map showing float locations."""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    # Get unique locations
    locations = df.groupby(['platform_number', 'latitude', 'longitude']).size().reset_index()
    
    # Create base map centered on data
    center_lat = locations['latitude'].mean()
    center_lon = locations['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Add markers for each platform
    for _, row in locations.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Platform: {row['platform_number']}",
            tooltip=f"{row['platform_number']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    return m


# def main():
#     """Main application function."""
    
#     # Header
#     st.title("🌊 FloatChat - ARGO Ocean Data Explorer")
#     st.markdown("### AI-Powered Conversational Interface for Ocean Data Discovery")
    
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Settings")
        
#         # Data loading section
#         if not st.session_state.data_loaded:
#             if st.button("Load Sample Data", type="primary"):
#                 load_sample_data()
#         else:
#             st.success("✅ Sample data loaded")
            
#             # Display database summary
#             summary = st.session_state.db_manager.get_data_summary()
#             st.metric("Total Records", summary['total_records'])
#             st.metric("Total Platforms", summary['total_platforms'])
        
#         # File upload
#         st.divider()
#         uploaded_file = st.file_uploader(
#             "Upload NetCDF File",
#             type=['nc', 'nc4'],
#             help="Upload ARGO NetCDF files for analysis"
#         )
        
#         if uploaded_file is not None:
#             # Save uploaded file
#             file_path = Path('data/uploads') / uploaded_file.name
#             with open(file_path, 'wb') as f:
#                 f.write(uploaded_file.getbuffer())
            
#             # Process the file
#             with st.spinner("Processing NetCDF file..."):
#                 ingester = ARGODataIngester()
#                 try:
#                     df, metadata = ingester.process_file(str(file_path))
#                     st.session_state.db_manager.insert_profile_data(df, metadata)
#                     summary_text = ingester.generate_summary_text(df, metadata)
#                     st.session_state.db_manager.add_to_vector_db(
#                         doc_id=uploaded_file.name,
#                         text=summary_text,
#                         metadata=metadata
#                     )
#                     st.success(f"Successfully processed {uploaded_file.name}")
#                 except Exception as e:
#                     st.error(f"Error processing file: {e}")
        
#         # Example queries
#         st.divider()
#         st.subheader("💡 Example Queries")
#         example_queries = st.session_state.query_processor.suggest_queries()
#         for query in example_queries[:5]:
#             if st.button(query, key=f"example_{query[:20]}"):
#                 st.session_state.messages.append({"role": "user", "content": query})
    
#     # Main content area
#     col1, col2 = st.columns([3, 2])
    
#     with col1:
#         st.header("💬 Chat Interface")
        
#         # Display chat messages
#         message_container = st.container()
#         with message_container:
#             for message in st.session_state.messages:
#                 with st.chat_message(message["role"]):
#                     st.write(message["content"])
        
#         # Chat input
#         if prompt := st.chat_input("Ask about ARGO ocean data..."):
#             # Add user message
#             st.session_state.messages.append({"role": "user", "content": prompt})
            
#             # Process query
#             with st.spinner("Processing query..."):
#                 results_df, response, components = st.session_state.query_processor.process_query(prompt)
                
#                 # Add assistant response
#                 st.session_state.messages.append({"role": "assistant", "content": response})
                
#                 # Store results for visualization
#                 st.session_state.last_results = results_df
#                 st.session_state.last_components = components
            
#             st.rerun()
    
#     with col2:
#         st.header("📊 Visualizations")
        
#         # Display visualizations if we have results
#         if hasattr(st.session_state, 'last_results') and not st.session_state.last_results.empty:
#             df = st.session_state.last_results
#             components = st.session_state.last_components
            
#             # Show data preview
#             with st.expander("📋 Data Preview", expanded=False):
#                 st.dataframe(df.head(100), use_container_width=True)
            
#             # Create appropriate visualizations
#             if components.get('visualization', False) or 'profile' in str(components):
#                 # Profile plots
#                 profile_fig = create_profile_plot(df)
#                 if profile_fig:
#                     st.plotly_chart(profile_fig, use_container_width=True)
            
#             # Always show map if we have location data
#             if 'latitude' in df.columns and 'longitude' in df.columns:
#                 st.subheader("🗺️ Float Locations")
#                 float_map = create_map(df)
#                 if float_map:
#                     st_folium(float_map, height=400, width=None)
            
#             # Time series if available
#             if 'time' in df.columns and len(df) > 1:
#                 st.subheader("📈 Time Series")
#                 # Convert time to datetime if needed
#                 df['time'] = pd.to_datetime(df['time'])
                
#                 # Create time series plot
#                 if 'temperature' in df.columns:
#                     fig = px.line(df, x='time', y='temperature', 
#                                  color='platform_number' if 'platform_number' in df.columns else None,
#                                  title="Temperature Over Time")
#                     st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Ask a question to see visualizations here!")
            
#             # Show sample visualization with sample data
#             if st.session_state.data_loaded:
#                 sample_df = st.session_state.db_manager.execute_sql(
#                     "SELECT * FROM profiles LIMIT 100"
#                 )
#                 if not sample_df.empty:
#                     st.subheader("📊 Sample Data Overview")
#                     profile_fig = create_profile_plot(sample_df)
#                     if profile_fig:
#                         st.plotly_chart(profile_fig, use_container_width=True)
















































def main():
    """Main application function."""
    
    # Header
    st.title("🌊 FloatChat - ARGO Ocean Data Explorer")
    st.markdown("### AI-Powered Conversational Interface for Ocean Data Discovery")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Data loading section
        if not st.session_state.data_loaded:
            if st.button("Load Sample Data", type="primary"):
                load_sample_data()
        else:
            st.success("✅ Sample data loaded")
            
            # Display database summary
            summary = st.session_state.db_manager.get_data_summary()
            st.metric("Total Records", summary['total_records'])
            st.metric("Total Platforms", summary['total_platforms'])
        
        # File upload
        st.divider()
        uploaded_file = st.file_uploader(
            "Upload NetCDF File",
            type=['nc', 'nc4'],
            help="Upload ARGO NetCDF files for analysis"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            file_path = Path('data/uploads') / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file
            with st.spinner("Processing NetCDF file..."):
                ingester = ARGODataIngester()
                try:
                    df, metadata = ingester.process_file(str(file_path))
                    st.session_state.db_manager.insert_profile_data(df, metadata)
                    summary_text = ingester.generate_summary_text(df, metadata)
                    st.session_state.db_manager.add_to_vector_db(
                        doc_id=uploaded_file.name,
                        text=summary_text,
                        metadata=metadata
                    )
                    st.success(f"Successfully processed {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        # Example queries
        st.divider()
        st.subheader("💡 Example Queries")
        example_queries = st.session_state.query_processor.suggest_queries()
        for query in example_queries[:5]:
            if st.button(query, key=f"example_{query[:20]}"):
                st.session_state.messages.append({"role": "user", "content": query})
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        with st.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
    
    with col2:
        st.header("📊 Visualizations")
        
        # Display visualizations if we have results
        if hasattr(st.session_state, 'last_results') and not st.session_state.last_results.empty:
            df = st.session_state.last_results
            components = st.session_state.last_components
            
            # Show data preview
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)
            
            # Create appropriate visualizations
            if components.get('visualization', False) or 'profile' in str(components):
                # Profile plots
                profile_fig = create_profile_plot(df)
                if profile_fig:
                    st.plotly_chart(profile_fig, use_container_width=True)
            
            # Always show map if we have location data
            if 'latitude' in df.columns and 'longitude' in df.columns:
                st.subheader("🗺️ Float Locations")
                float_map = create_map(df)
                if float_map:
                    st_folium(float_map, height=400, width=None)
            
            # Time series if available
            if 'time' in df.columns and len(df) > 1:
                st.subheader("📈 Time Series")
                # Convert time to datetime if needed
                df['time'] = pd.to_datetime(df['time'])
                
                # Create time series plot
                if 'temperature' in df.columns:
                    fig = px.line(
                        df, x='time', y='temperature', 
                        color='platform_number' if 'platform_number' in df.columns else None,
                        title="Temperature Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ask a question to see visualizations here!")
            
            # Show sample visualization with sample data
            if st.session_state.data_loaded:
                sample_df = st.session_state.db_manager.execute_sql(
                    "SELECT * FROM profiles LIMIT 100"
                )
                if not sample_df.empty:
                    st.subheader("📊 Sample Data Overview")
                    profile_fig = create_profile_plot(sample_df)
                    if profile_fig:
                        st.plotly_chart(profile_fig, use_container_width=True)
    
    # --- Chat input (outside columns, always at bottom) ---
    prompt = st.chat_input("Ask about ARGO ocean data...")
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process query
        with st.spinner("Processing query..."):
            results_df, response, components = st.session_state.query_processor.process_query(prompt)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Store results for visualization
            st.session_state.last_results = results_df
            st.session_state.last_components = components
        
        st.rerun()








import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px

# Import your own modules
from database import DatabaseManager
from llm_processor import QueryProcessor    # ✅ use this instead
from data_ingestion import ARGODataIngester
# from visualizations import create_profile_plot, create_map

# --- Initialize session state ---
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if "query_processor" not in st.session_state:
    st.session_state.query_processor = QueryProcessor(st.session_state.db_manager)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_results" not in st.session_state:
    st.session_state.last_results = pd.DataFrame()

if "last_components" not in st.session_state:
    st.session_state.last_components = {}



if __name__ == "__main__":
    main()
