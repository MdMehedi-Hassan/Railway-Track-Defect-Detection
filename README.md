# Railway-Track-Defect-Detection

## Overview

This is a comprehensive web application for detecting defects in railway tracks using YOLO (You Only Look Once) object detection model. The system features:

- **AI-powered defect detection** using YOLOv8 model
- **User authentication** with admin privileges
- **MongoDB integration** for data storage and retrieval
- **Interactive dashboard** with visualizations
- **Batch processing** of multiple images
- **Detailed reporting** with confidence scores

## Features

### Core Functionality
- 🚂 YOLO-based defect detection for railway tracks
- 📂 Supports folder-based image processing
- 📊 Interactive results visualization with Plotly
- 💾 Automatic saving of detection results to MongoDB
- 📥 Downloadable reports in CSV format

### User Management
- 👤 Role-based access (admin vs regular users)
- 🔐 Secure password hashing
- 📝 User creation and management (admin only)
- 📊 System statistics dashboard

### Technical Highlights
- 🐍 Python backend with Streamlit for web interface
- 🍃 MongoDB for data persistence
- 🔄 Asynchronous processing with progress tracking
- 🎨 Custom CSS for enhanced UI/UX
- 📈 Data visualization with Plotly and Pandas

## Installation

### Prerequisites
- Python 3.8+
- MongoDB (running locally on default port 27017)
- Git

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/railway-defect-detection.git
   cd railway-defect-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the YOLO model weights file (`best.pt`) and place it in the project root.

5. Start MongoDB service (ensure it's running on localhost:27017)

6. Run the application:
   ```bash
   streamlit run main4.py --server.port 8502
   ```

## Usage

### Default Credentials
- Admin account: `admin` / `admin123`
- Regular users can be created by admin

### Workflow
1. Login with your credentials
2. Select input method (Folder with Images recommended)
3. Enter the path to your images folder
4. Adjust detection settings as needed
5. Click "Start Processing"
6. View results and download reports

## File Structure

```
.
├── main4.py                # Main application file
├── yolo11x/                # YOLO model directory
│   └── best.pt             # YOLO model weights
├── README.md               # This documentation
└── requirements.txt        # Python dependencies
```

## Dependencies

- Python 3.8+
- Streamlit
- Ultralytics (YOLOv8)
- OpenCV
- Pillow
- NumPy
- Pandas
- Plotly
- PyMongo

## Configuration

The application is configured to use MongoDB running locally on the default port. To change this, modify the connection string in the `connect_to_mongodb()` function.

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Ensure `best.pt` exists in the correct location
   - Verify the model is compatible with YOLOv8

2. **MongoDB connection issues**:
   - Check if MongoDB service is running
   - Verify connection string in code

3. **Dependency conflicts**:
   - Use the exact versions specified in requirements.txt
   - Create a fresh virtual environment

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLO team for the object detection model
- Streamlit for the web framework
- MongoDB for the database solution

## Contact

For questions or support, please contact:
- Md. Mehedi Hassan - [Facebook](https://www.facebook.com/share/1CDCvcHq4J/)
- Moshiur Rahman Sayem - [Facebook](https://www.facebook.com/share/192c8qrS3v/)

---

**Note**: This is a prototype system intended for demonstration purposes. For production use, additional security measures and testing are recommended.
