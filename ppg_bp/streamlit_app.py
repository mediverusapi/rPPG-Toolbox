#!/usr/bin/env python3
"""
Streamlit Web App for rPPG Blood Pressure Estimation

This web application allows users to upload video files (.mp4) and get
real-time blood pressure estimations using rPPG technology.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import torch
import tempfile
import os
import json
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import local modules
try:
    from model import M5_fusion_transformer
    from preprocess import normalize_min_max, filter_ppg_signal, remove_artifact
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure you're running from the ppg_bp directory.")
    st.stop()

class VideoProcessor:
    """Process video files to extract PPG signals"""
    
    def __init__(self):
        self.fps = 30  # Default FPS for processing
        
    def extract_face_region(self, frame):
        """Extract face region from frame using OpenCV cascade"""
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Extract forehead region (upper 1/3 of face)
            forehead_y = y + int(h * 0.1)
            forehead_h = int(h * 0.3)
            forehead_region = frame[forehead_y:forehead_y + forehead_h, x:x + w]
            
            return forehead_region, (x, y, w, h)
        else:
            return None, None
    
    def extract_ppg_from_video(self, video_path: str, progress_callback=None) -> np.ndarray:
        """Extract PPG signal from video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        st.info(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        green_values = []
        frame_count = 0
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract face region
            face_region, face_coords = self.extract_face_region(frame)
            
            if face_region is not None:
                # Extract green channel (most sensitive to blood volume changes)
                green_channel = face_region[:, :, 1]  # Green channel in BGR
                
                # Calculate mean green value
                mean_green = np.mean(green_channel)
                green_values.append(mean_green)
            else:
                # If no face detected, use previous value or zero
                if green_values:
                    green_values.append(green_values[-1])
                else:
                    green_values.append(0)
            
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            if progress_callback:
                progress_callback(progress)
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        if len(green_values) == 0:
            raise ValueError("No face detected in video")
        
        return np.array(green_values)

class BloodPressurePredictor:
    """Blood pressure prediction using pre-trained models"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._setup_device(device)
        
        # Default configuration
        self.config = {
            "chunk_length": 512,
            "filter_lpf": 0.7,
            "filter_hpf": 10,
            "frequency_sample": 60,
            "derivative_input": False
        }
        
        # Blood pressure normalization constants
        self.sys_min, self.sys_max = 100, 180
        self.dia_min, self.dia_max = 55, 100
        
        # Demographics normalization
        self.age_min, self.age_max = 34, 96
        self.bmi_min, self.bmi_max = 17, 41
        
        # Initialize with a dummy model (will be replaced when model is loaded)
        self.model = None
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device"""
        if device == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        return device
    
    def load_model(self, model_path: str) -> bool:
        """Load pre-trained model"""
        try:
            if not os.path.exists(model_path):
                return False
            
            # Create model
            self.model = M5_fusion_transformer(n_input=1, n_output=2)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()
            self.model.to(self.device)
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def preprocess_ppg(self, ppg_signal: np.ndarray) -> np.ndarray:
        """Preprocess PPG signal"""
        # Remove artifacts
        ppg_clean = remove_artifact(ppg_signal)
        
        # Apply bandpass filter
        ppg_filtered = filter_ppg_signal(
            ppg_clean, 
            self.config["filter_lpf"], 
            self.config["filter_hpf"], 
            self.config["frequency_sample"]
        )
        
        # Normalize
        ppg_normalized = normalize_min_max(ppg_filtered)
        
        return ppg_normalized
    
    def normalize_demographics(self, age: float, bmi: float) -> Tuple[float, float]:
        """Normalize demographics"""
        age_norm = max(0, min(1, (age - self.age_min) / (self.age_max - self.age_min)))
        bmi_norm = max(0, min(1, (bmi - self.bmi_min) / (self.bmi_max - self.bmi_min)))
        return age_norm, bmi_norm
    
    def predict(self, ppg_signal: np.ndarray, age: float, bmi: float) -> Dict:
        """Predict blood pressure from PPG signal"""
        if self.model is None:
            raise ValueError("No model loaded")
        
        start_time = time.time()
        
        # Preprocess signal
        ppg_processed = self.preprocess_ppg(ppg_signal)
        
        # Prepare input
        chunk_length = self.config["chunk_length"]
        
        if len(ppg_processed) < chunk_length:
            # Pad with mean
            pad_length = chunk_length - len(ppg_processed)
            mean_val = np.mean(ppg_processed)
            ppg_processed = np.concatenate([ppg_processed, np.full(pad_length, mean_val)])
        elif len(ppg_processed) > chunk_length:
            # Take middle section
            start_idx = (len(ppg_processed) - chunk_length) // 2
            ppg_processed = ppg_processed[start_idx:start_idx + chunk_length]
        
        # Convert to tensors
        ppg_tensor = torch.tensor(ppg_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        age_norm, bmi_norm = self.normalize_demographics(age, bmi)
        age_tensor = torch.tensor([[age_norm]], dtype=torch.float32).to(self.device)
        bmi_tensor = torch.tensor([[bmi_norm]], dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            bp_pred = self.model(ppg_tensor, age_tensor, bmi_tensor)
        
        # Denormalize
        bp_numpy = bp_pred.squeeze().cpu().numpy()
        sys_pred = bp_numpy[0] * (self.sys_max - self.sys_min) + self.sys_min
        dia_pred = bp_numpy[1] * (self.dia_max - self.dia_min) + self.dia_min
        
        inference_time = time.time() - start_time
        
        return {
            'systolic': float(sys_pred),
            'diastolic': float(dia_pred),
            'inference_time': inference_time,
            'signal_length': len(ppg_signal),
            'processed_length': len(ppg_processed)
        }

def create_demo_model():
    """Create a demo model for testing when no real model is available"""
    model = M5_fusion_transformer(n_input=1, n_output=2)
    
    # Initialize with random weights (for demo purposes)
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    return model

def plot_ppg_signal(ppg_signal: np.ndarray, title: str = "PPG Signal"):
    """Plot PPG signal using Plotly"""
    fig = go.Figure()
    
    time_axis = np.arange(len(ppg_signal)) / 30  # Assuming 30 FPS
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=ppg_signal,
        mode='lines',
        name='PPG Signal',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (seconds)',
        yaxis_title='Signal Amplitude',
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    st.set_page_config(
        page_title="rPPG Blood Pressure Estimation",
        page_icon="🩺",
        layout="wide"
    )
    
    st.title("🩺 rPPG Blood Pressure Estimation")
    st.markdown("Upload a video file to estimate blood pressure using remote photoplethysmography")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Settings")
    model_option = st.sidebar.selectbox(
        "Model Type",
        ["Demo Model (No file needed)", "Upload Custom Model"]
    )
    
    # Initialize predictor
    predictor = BloodPressurePredictor()
    
    if model_option == "Upload Custom Model":
        model_file = st.sidebar.file_uploader(
            "Upload Model File (.pth)",
            type=['pth'],
            help="Upload a pre-trained PyTorch model file"
        )
        
        if model_file:
            # Save uploaded model temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(model_file.read())
                tmp_model_path = tmp_file.name
            
            if predictor.load_model(tmp_model_path):
                st.sidebar.success("✅ Model loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load model")
                st.stop()
            
            # Clean up
            os.unlink(tmp_model_path)
        else:
            st.sidebar.warning("Please upload a model file")
            st.stop()
    else:
        # Use demo model
        st.sidebar.info("Using demo model (random predictions)")
        predictor.model = create_demo_model()
        predictor.model.to(predictor.device)
    
    # Demographics input
    st.sidebar.subheader("Demographics")
    age = st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=30)
    bmi = st.sidebar.number_input("BMI", min_value=15.0, max_value=50.0, value=24.0, step=0.1)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Video Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video file containing a person's face"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_video_path = tmp_file.name
            
            # Display video info
            st.video(uploaded_file)
            
            # Process video button
            if st.button("🚀 Process Video & Predict BP", type="primary"):
                try:
                    with st.spinner("Processing video..."):
                        # Extract PPG signal
                        processor = VideoProcessor()
                        ppg_signal = processor.extract_ppg_from_video(tmp_video_path)
                        
                        st.success(f"✅ Extracted PPG signal with {len(ppg_signal)} samples")
                        
                        # Plot raw signal
                        fig_raw = plot_ppg_signal(ppg_signal, "Raw PPG Signal from Video")
                        st.plotly_chart(fig_raw, use_container_width=True)
                        
                        # Predict blood pressure
                        with st.spinner("Predicting blood pressure..."):
                            result = predictor.predict(ppg_signal, age, bmi)
                        
                        # Display results in col2
                        with col2:
                            st.header("Results")
                            
                            # Blood pressure results
                            st.subheader("📊 Blood Pressure Estimation")
                            
                            # Create metrics display
                            metric_col1, metric_col2 = st.columns(2)
                            
                            with metric_col1:
                                st.metric(
                                    label="Systolic BP",
                                    value=f"{result['systolic']:.0f} mmHg",
                                    help="Upper blood pressure reading"
                                )
                            
                            with metric_col2:
                                st.metric(
                                    label="Diastolic BP", 
                                    value=f"{result['diastolic']:.0f} mmHg",
                                    help="Lower blood pressure reading"
                                )
                            
                            # Blood pressure category
                            sys_val = result['systolic']
                            dia_val = result['diastolic']
                            
                            if sys_val < 120 and dia_val < 80:
                                category = "Normal"
                                color = "green"
                            elif sys_val < 130 and dia_val < 80:
                                category = "Elevated"
                                color = "orange"
                            elif sys_val < 140 or dia_val < 90:
                                category = "High BP Stage 1"
                                color = "red"
                            else:
                                category = "High BP Stage 2"
                                color = "darkred"
                            
                            st.markdown(f"**Category:** :{color}[{category}]")
                            
                            # Technical details
                            st.subheader("🔧 Technical Details")
                            st.write(f"**Processing Time:** {result['inference_time']:.3f} seconds")
                            st.write(f"**Signal Length:** {result['signal_length']} samples")
                            st.write(f"**Processed Length:** {result['processed_length']} samples")
                            st.write(f"**Device:** {predictor.device}")
                            
                            # Download results
                            results_df = pd.DataFrame([{
                                'Age': age,
                                'BMI': bmi,
                                'Systolic_BP_mmHg': result['systolic'],
                                'Diastolic_BP_mmHg': result['diastolic'],
                                'Category': category,
                                'Processing_Time_seconds': result['inference_time'],
                                'Signal_Length': result['signal_length']
                            }])
                            
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"bp_results_{int(time.time())}.csv",
                                mime="text/csv"
                            )
                
                except Exception as e:
                    st.error(f"❌ Error processing video: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_video_path):
                        os.unlink(tmp_video_path)
    
    with col2:
        if uploaded_file is None:
            st.header("Instructions")
            st.markdown("""
            ### How to use:
            
            1. **Upload a video** (.mp4, .avi, .mov) containing a person's face
            2. **Enter demographics** (age and BMI) in the sidebar
            3. **Select model type** in the sidebar:
               - Demo Model: Uses random predictions for testing
               - Custom Model: Upload your own trained .pth model file
            4. **Click "Process Video"** to extract PPG signal and predict BP
            
            ### Tips for best results:
            - Ensure good lighting on the face
            - Minimize head movement
            - Video should be at least 30 seconds long
            - Person should face the camera directly
            
            ### Note:
            This is a demonstration application. For medical use, please consult 
            with healthcare professionals and use validated medical devices.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🩺 **Disclaimer:** This application is for research and demonstration purposes only. "
        "Do not use for medical diagnosis. Always consult healthcare professionals for medical advice."
    )

if __name__ == "__main__":
    main() 