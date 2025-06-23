import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import pymongo
import hashlib
import uuid
import json


#streamlit run main4.py --server.port 8502

# MongoDB Connection
def connect_to_mongodb():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["railway_defect_detection"]
        return client, db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None, None

# Initialize MongoDB connection
mongo_client, db = connect_to_mongodb()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'init_db' not in st.session_state:
    st.session_state.init_db = False

# Initialize database collections if they don't exist
def initialize_database():
    if not st.session_state.init_db and mongo_client is not None:
        # Check if users collection exists and has admin
        users_collection = db["users"]
        if users_collection.count_documents({}) == 0:
            # Create default admin user
            admin_user = {
                "username": "admin",
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "is_admin": True,
                "created_at": datetime.now()
            }
            users_collection.insert_one(admin_user)
            st.session_state.init_db = True
            
            # Create indexes
            users_collection.create_index("username", unique=True)
            
            # Create defects collection with indexes
            defects_collection = db["defects"]
            defects_collection.create_index("detection_id")
            defects_collection.create_index("detected_by")
            defects_collection.create_index("detection_date")
            
            st.success("Database initialized with default admin account (username: admin, password: admin123)")
        else:
            st.session_state.init_db = True

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Authentication functions
def login(username, password):
    if mongo_client is None:
        st.error("Cannot connect to database")
        return False
    
    users_collection = db["users"]
    user = users_collection.find_one({
        "username": username,
        "password": hash_password(password)
    })
    
    if user:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.is_admin = user.get("is_admin", False)
        return True
    return False

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.is_admin = False

def create_user(username, password, is_admin=False):
    if mongo_client is None:
        st.error("Cannot connect to database")
        return False
    
    users_collection = db["users"]
    
    # Check if user already exists
    if users_collection.find_one({"username": username}):
        return False
    
    # Create new user
    new_user = {
        "username": username,
        "password": hash_password(password),
        "is_admin": is_admin,
        "created_at": datetime.now()
    }
    
    users_collection.insert_one(new_user)
    return True

# Set page configuration
st.set_page_config(
    page_title="Railway Track Defect Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚂"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .status-ok {
        color: #10B981;
        font-weight: bold;
    }
    .status-warning {
        color: #F59E0B;
        font-weight: bold;
    }
    .defect-detected {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .no-defect {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .image-container {
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        color: #6B7280;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #E5E7EB;
    }
    /* Enhanced login styling */
    .login-container {
        max-width: 450px;
        margin: 2rem auto;
        padding: 2.5rem;
        background-color: #FFFFFF;
        border-radius: 1rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-header img {
        max-width: 80px;
        margin-bottom: 1rem;
    }
    .login-header h2 {
        color: #1E3A8A;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .login-header p {
        color: #6B7280;
        font-size: 1rem;
    }
    .login-input {
        margin-bottom: 1.5rem;
    }
    .login-input label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 500;
        color: #374151;
    }
    .login-input input {
        width: 100%;
        padding: 0.75rem 1rem;
        border: 1px solid #D1D5DB;
        border-radius: 0.5rem;
        font-size: 1rem;
        transition: all 0.3s;
    }
    .login-input input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
        outline: none;
    }
    .login-button {
        width: 100%;
        padding: 0.75rem 1rem;
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s;
    }
    .login-button:hover {
        background-color: #1D4ED8;
    }
    .login-footer {
        text-align: center;
        margin-top: 1.5rem;
        font-size: 0.875rem;
        color: #6B7280;
    }
    /* Split screen for login */
    .login-split-screen {
        display: flex;
        min-height: 80vh;
        margin-top: -4rem;
    }
    .login-split-left {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }
    .login-split-right {
        flex: 1;
        background-color: #EFF6FF;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        border-radius: 0 1rem 1rem 0;
    }
    .login-feature {
        margin-bottom: 1.5rem;
    }
    .login-feature h3 {
        display: flex;
        align-items: center;
        font-size: 1.25rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .login-feature h3 svg {
        margin-right: 0.5rem;
    }
    .login-feature p {
        color: #4B5563;
        padding-left: 2rem;
    }
    .admin-section {
        background-color: #FFEDD5;
        border-left: 5px solid #F97316;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .auth-tabs .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .auth-tabs .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        border: 1px solid #E5E7EB;
        border-bottom: none;
        padding: 10px 16px;
    }
    .auth-tabs .stTabs [aria-selected="true"] {
        background-color: #EFF6FF;
        border-top: 3px solid #3B82F6;
    }
    /* Error message styling */
    .error-message {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        color: #B91C1C;
    }
    .success-message {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        color: #065F46;
    }
    /* Report card styling */
    .report-card {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    .report-card:hover {
        box-shadow: 0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22);
    }
    .report-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .report-id {
        font-size: 0.875rem;
        color: #6B7280;
    }
    .report-date {
        font-size: 0.875rem;
        color: #6B7280;
    }
    .report-content {
        padding: 0.5rem 0;
    }
    .report-footer {
        display: flex;
        justify-content: flex-end;
        padding-top: 0.5rem;
        border-top: 1px solid #E5E7EB;
    }
    .view-button {
        padding: 0.25rem 0.75rem;
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the database
initialize_database()

# Load the model with caching to improve performance
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    model = load_model("./yolo11x/best.pt")
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Error loading model: {e}")

# Authentication UI
def show_login_ui():
    st.markdown('<div class="main-header">🚂 Railway Track Defect Detection System</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="auth-tabs">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Login", "About"])
    
    with tab1:
        # st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.subheader("Login to Access the System")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Login", use_container_width=True):
                if login(username, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="info-box">
        <h3>Railway Track Defect Detection System</h3>
        
        This application uses AI to detect defects in railway tracks using the YOLO model.
        <br><br>
        To access the system, please login with your credentials. If you don't have an account, 
        please contact the system administrator.
        <br><br>
        <b>Developers:</b><br>
        Moshiur Rahman Sayem - <a href="https://www.facebook.com/moshiurrahmansayembd" target="_blank">Facebook</a><br>
        Md. Mehedi Hassan - <a href="https://www.facebook.com/md.hassan.mehedi.s____" target="_blank">Facebook</a><br>            
        <b>Created:</b> May 2025
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# User management UI (admin only)
def show_user_management():
    st.markdown('<div class="admin-section">', unsafe_allow_html=True)
    st.subheader("👤 User Management")
    
    tab1, tab2, tab3 = st.tabs(["Create User", "View Users", "System Stats"])
    
    with tab1:
        st.subheader("Create New User")
        new_username = st.text_input("Username", key="new_username")
        new_password = st.text_input("Password", type="password", key="new_password")
        is_admin = st.checkbox("Admin privileges", key="is_admin")
        
        if st.button("Create User"):
            if new_username and new_password:
                if create_user(new_username, new_password, is_admin):
                    st.success(f"User '{new_username}' created successfully!")
                else:
                    st.error(f"Username '{new_username}' already exists!")
            else:
                st.warning("Please enter both username and password")
    
    with tab2:
        if mongo_client is not None:
            users = list(db["users"].find({}, {"_id": 0, "password": 0}))
            if users:
                users_df = pd.DataFrame(users)
                # Convert datetime to string for display
                if "created_at" in users_df.columns:
                    users_df["created_at"] = users_df["created_at"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))
                st.dataframe(users_df, use_container_width=True)
            else:
                st.info("No users found")
    
    with tab3:
        if mongo_client is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                user_count = db["users"].count_documents({})
                admin_count = db["users"].count_documents({"is_admin": True})
                
                st.metric("Total Users", user_count)
                st.metric("Admin Users", admin_count)
            
            with col2:
                defect_count = db["defects"].count_documents({})
                latest_detection = db["defects"].find_one(
                    {}, 
                    sort=[("detection_date", pymongo.DESCENDING)]
                )
                
                st.metric("Total Detection Records", defect_count)
                if latest_detection:
                    st.metric(
                        "Latest Detection", 
                        latest_detection.get("detection_date", "Unknown").strftime("%Y-%m-%d %H:%M")
                    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Save detection results to MongoDB
def save_detection_to_db(detection_data, detection_id=None):
    if mongo_client is None:
        return None
    
    defects_collection = db["defects"]
    
    if detection_id is None:
        detection_id = str(uuid.uuid4())
    
    detection_record = {
        "detection_id": detection_id,
        "detected_by": st.session_state.username,
        "detection_date": datetime.now(),
        "results": detection_data
    }
    
    defects_collection.insert_one(detection_record)
    return detection_id

# Main application UI
def show_main_app():
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown(f"""
        <div class="info-box">
        <h3>Welcome, {st.session_state.username}!</h3>
        {"<span class='status-warning'>Admin Mode</span>" if st.session_state.is_admin else ""}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-header">Detection Settings</div>', unsafe_allow_html=True)
        
        # Input methods
        st.subheader("Input Source")
        input_method = st.radio(
            "Select input method:",
            # ["Folder with Images", "Upload Images", "Sample Images"],
            ["Folder with Images"],
            key="input_method"
        )
        
        # Folder path input
        if input_method == "Folder with Images":
            folder_path = st.text_input("📂 Enter Folder Path:", placeholder="/path/to/images")
        
        # File uploader
        elif input_method == "Upload Images":
            uploaded_files = st.file_uploader(
                "Upload image files", 
                accept_multiple_files=True,
                type=["jpg", "jpeg", "png"]
            )
        
        # Sample images
        elif input_method == "Sample Images":
            st.info("Using built-in sample images for demonstration")
        
        # Advanced settings
        st.markdown('<div class="sidebar-header">Advanced Settings</div>', unsafe_allow_html=True)
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for detection"
        )
        
        display_options = st.multiselect(
            "Display Options",
            options=["Show confidence scores", "Show bounding boxes", "Show detection labels"],
            default=["Show confidence scores", "Show bounding boxes", "Show detection labels"]
        )
        
        # User options
        st.markdown('<div class="sidebar-header">User Options</div>', unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            logout()
            st.rerun()

    # Main content
    st.markdown('<div class="main-header">🚂 Railway Track Defect Detection</div>', unsafe_allow_html=True)
    
    # Show admin panel if user is admin
    if st.session_state.is_admin:
        show_user_management()

    # # Interactive dashboard stats (placeholder for real data)
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     st.metric(label="Total Images", value="0", delta=None)
    # with col2:
    #     st.metric(label="Defects Found", value="0", delta=None)
    # with col3:
    #     st.metric(label="Avg. Confidence", value="0%", delta=None)
    # with col4:
    #     st.metric(label="Processing Time", value="0s", delta=None)

    # st.markdown("""
    # This intelligent system analyzes railway track images to detect potential defects using YOLO, 
    # a state-of-the-art object detection model. Upload your images or provide a folder path to begin analysis.
    # """)

    # Recent detections
    if mongo_client is not None:
        recent_detections = list(db["defects"].find({}, {"_id": 0, "results": 0}).sort("detection_date", -1).limit(5))
        if recent_detections:
            st.markdown('<div class="sub-header">Recent Detection Sessions</div>', unsafe_allow_html=True)
            recent_df = pd.DataFrame(recent_detections)
            recent_df["detection_date"] = recent_df["detection_date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
    
    # Process function with progress tracking
    def process_images(image_list, image_paths=None, is_upload=False):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        summary_data = []
        all_detections = []
        detection_details = []
        
        tabs = st.tabs([f"Image {i+1}: {getattr(img, 'name', os.path.basename(img)) if is_upload else os.path.basename(img)}" 
                       for i, img in enumerate(image_list)])
        
        for idx, image_item in enumerate(image_list):
            # Update progress
            progress = (idx + 1) / len(image_list)
            progress_bar.progress(progress)
            status_text.text(f"Processing image {idx + 1} of {len(image_list)}...")
            
            # Load image based on source
            if is_upload:
                image = Image.open(image_item)
                image_name = image_item.name
            else:
                image_path = os.path.join(image_paths, image_item)
                image = Image.open(image_path)
                image_name = os.path.basename(image_path)
            
            # Convert to OpenCV format for processing
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Perform inference
            results = model(image_bgr, conf=confidence_threshold)
            
            # Annotate image
            annotated_image = results[0].plot()
            
            # Process detections
            detections = []
            for detection in results[0].boxes:
                label = int(detection.cls)
                confidence = float(detection.conf)
                bbox = detection.xyxy[0].tolist()  # Get bounding box coordinates
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox
                })
                all_detections.append(confidence)
                
            detection_status = "Defects Detected" if detections else "No Defects"
            
            # Prepare detection data for MongoDB
            detection_data = {
                "image_name": image_name,
                "status": detection_status,
                "defects_count": len(detections),
                "detections": [
                    {
                        "label": d["label"],
                        "confidence": d["confidence"],
                        "bbox": d["bbox"]
                    } for d in detections
                ],
                "processing_time": time.time() - start_time
            }
            detection_details.append(detection_data)
            
            # Add to summary
            summary_data.append({
                "Image Name": image_name,
                "Status": detection_status,
                "Defects Count": len(detections),
                "Avg Confidence": sum([d["confidence"] for d in detections]) / len(detections) if detections else 0,
                "Processing Time": f"{(time.time() - start_time) / (idx + 1):.2f}s"
            })
            
            # Display in tab
            with tabs[idx]:
                st.markdown(f"#### {image_name}")
                
                # Status indicator
                if detection_status == "Defects Detected":
                    st.markdown(f'''
                    <div class="defect-detected">
                        ⚠️ <span class="status-warning">{detection_status}</span> - Found {len(detections)} potential issues
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="no-defect">
                        ✅ <span class="status-ok">{detection_status}</span> - Track appears normal
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Image comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(image, caption="Original Image", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(annotated_image, caption="Detected Issues", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                # Detection details
                if detections:
                    st.markdown('<div class="sub-header">Detection Details</div>', unsafe_allow_html=True)
                    
                    # Create a dataframe for better visualization
                    df_detections = pd.DataFrame([
                        {"Label": d["label"], "Confidence": f"{d['confidence']:.2f}", "Confidence_Value": d["confidence"]} 
                        for d in detections
                    ])
                    
                    # Bar chart visualization
                    if not df_detections.empty:
                        fig = px.bar(
                            df_detections, 
                            x="Label", 
                            y="Confidence_Value", 
                            color="Label",
                            labels={"Confidence_Value": "Confidence Score", "Label": "Defect Type"},
                            title="Defect Detection Confidence Scores",
                            height=300
                        )
                        fig.update_layout(yaxis_range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed table
                    st.dataframe(
                        df_detections[["Label", "Confidence"]],
                        use_container_width=True,
                        hide_index=True
                    )
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        
        # Save to MongoDB
        detection_id = save_detection_to_db({
            "summary": summary_data,
            "details": detection_details
        })
        
        if detection_id:
            st.success(f"Results saved to database with ID: {detection_id}")
        
        # Update metrics
        total_time = time.time() - start_time
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Images", value=str(len(image_list)), delta=None)
        with col2:
            defect_count = sum(1 for item in summary_data if item["Status"] == "Defects Detected")
            st.metric(label="Defects Found", value=str(defect_count), delta=None)
        with col3:
            avg_conf = f"{sum(all_detections) / len(all_detections) * 100:.1f}%" if all_detections else "0%"
            st.metric(label="Avg. Confidence", value=avg_conf, delta=None)
        with col4:
            st.metric(label="Processing Time", value=f"{total_time:.2f}s", delta=None)
        
        # Display Summary Table
        st.markdown('<div class="sub-header">Detection Summary</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame(summary_data)
        
        # Add styling to the dataframe
        def highlight_defects(row):
            if row["Status"] == "Defects Detected":
                return ['background-color: #FEF2F2'] * len(row)
            return ['background-color: #ECFDF5'] * len(row)
        
        styled_df = summary_df.style.apply(highlight_defects, axis=1)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Create a downloadable CSV
        csv = summary_df.to_csv(index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"railway_detection_results_{timestamp}.csv",
            mime="text/csv"
        )
        
        # Summary visualization if multiple images
        if len(image_list) > 1:
            st.markdown('<div class="sub-header">Analysis Results</div>', unsafe_allow_html=True)
            
            # Create a pie chart for defect distribution
            status_count = summary_df["Status"].value_counts().reset_index()
            status_count.columns = ["Status", "Count"]
            
            fig = px.pie(
                status_count,
                values="Count",
                names="Status",
                title="Defect Detection Results",
                color="Status",
                color_discrete_map={
                    "Defects Detected": "#EF4444",
                    "No Defects": "#10B981"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

    # Handle different input methods
    if model_loaded:
        if input_method == "Folder with Images" and folder_path:
            if os.path.isdir(folder_path):
                st.success(f"📁 Found folder: `{folder_path}`")
                image_files = [
                    f for f in os.listdir(folder_path)
                    if f.lower().endswith(("jpg", "jpeg", "png"))
                ]
                
                if image_files:
                    st.info(f"🔍 Found {len(image_files)} image(s)")
                    
                    # Start processing button
                    if st.button("▶️ Start Processing", key="start_folder"):
                        process_images(image_files, folder_path, is_upload=False)
                else:
                    st.warning("⚠️ No valid images found in the folder.")
            else:
                st.error("🚫 The specified folder does not exist.")
        
        elif input_method == "Upload Images" and uploaded_files:
            st.success(f"📁 {len(uploaded_files)} image(s) uploaded successfully")
            
            # Start processing button
            if st.button("▶️ Start Processing", key="start_upload"):
                process_images(uploaded_files, is_upload=True)
        
        elif input_method == "Sample Images":
            # In a real app, you would include sample images in your project
            st.info("This would use built-in sample images. Demo mode - no processing will occur.")
            
            # Demo button (in a real app, this would process sample images)
            if st.button("🔍 Run Demo", key="run_demo"):
                st.warning("Demo mode: This would process sample images in a real deployment")
        
        elif input_method == "Folder with Images":
            st.info("📝 Please enter a folder path to get started.")
        
        elif input_method == "Upload Images":
            st.info("📤 Please upload image files to get started.")
    else:
        st.error("⚠️ Model could not be loaded. Please check if the model file exists at './best.pt'")

# Main application flow
if not st.session_state.logged_in:
    show_login_ui()
else:
    show_main_app()

# Footer
st.markdown("""
<div class="footer">
    Railway Track Defect Detection System | 2025
</div>
""", unsafe_allow_html=True)