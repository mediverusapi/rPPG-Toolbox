#!/usr/bin/env python3
"""
rPPG Blood Pressure Estimation Application

This application provides a comprehensive interface for testing remote photoplethysmography (rPPG)
based blood pressure estimation using various neural network models.

Features:
- Single PPG signal prediction
- Batch processing from data files
- Demo mode with existing datasets
- Real-time visualization
- Model comparison across multiple folds
- Comprehensive evaluation metrics

Usage Examples:
    # Basic single prediction
    python app.py --mode single --ppg_file sample.npy --age 45 --bmi 24.5

    # Demo mode with existing dataset
    python app.py --mode demo --config configs/config_face_manual_demo.json

    # Batch processing
    python app.py --mode batch --input_dir ./data --output_dir ./results

    # Model comparison
    python app.py --mode compare --config configs/test_config_face_t8.json
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(__file__))
from model import M5, M5_fusion_transformer
from model_final import M5_fusion_all_transformer
from dataset import UWBP_test_manual_demo
from preprocess import normalize_min_max, filter_ppg_signal, remove_artifact
import BlandAlman

class BloodPressurePredictor:
    """Main class for rPPG-based blood pressure prediction"""
    
    def __init__(self, model_path: str, config: Dict, device: str = 'auto'):
        """
        Initialize the blood pressure predictor
        
        Args:
            model_path: Path to the trained model
            config: Configuration dictionary
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.config = config
        self.device = self._setup_device(device)
        
        # Blood pressure normalization constants
        self.sys_min, self.sys_max = 100, 180
        self.dia_min, self.dia_max = 55, 100
        
        # Age and BMI normalization constants (from dataset analysis)
        self.age_min, self.age_max = 34, 96
        self.bmi_min, self.bmi_max = 17, 41
        
        # Load model
        self.model = self._load_model(model_path)
        
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"✓ Configuration: {config.get('prefix', 'Unknown')}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device"""
        if device == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        print(f"Using device: {device}")
        return device
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine model type based on config
        if self.config.get("derivative_input", False):
            model = M5(n_input=6)
        else:
            model = M5_fusion_transformer(n_input=1, n_output=2)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        model.to(self.device)
        
        return model
    
    def preprocess_ppg(self, ppg_signal: np.ndarray) -> np.ndarray:
        """
        Preprocess PPG signal according to the dataset pipeline
        
        Args:
            ppg_signal: Raw PPG signal
            
        Returns:
            Preprocessed PPG signal ready for model input
        """
        # Remove artifacts
        ppg_clean = remove_artifact(ppg_signal)
        
        # Apply bandpass filter
        lpf = self.config.get("filter_lpf", 0.7)
        hpf = self.config.get("filter_hpf", 10.0)
        fs = self.config.get("frequency_sample", 60)
        
        ppg_filtered = filter_ppg_signal(ppg_clean, lpf, hpf, fs)
        
        # Normalize
        ppg_normalized = normalize_min_max(ppg_filtered)
        
        return ppg_normalized
    
    def normalize_demographics(self, age: float, bmi: float) -> Tuple[float, float]:
        """Normalize age and BMI values"""
        age_norm = (age - self.age_min) / (self.age_max - self.age_min)
        bmi_norm = (bmi - self.bmi_min) / (self.bmi_max - self.bmi_min)
        
        # Clip to valid range
        age_norm = max(0, min(1, age_norm))
        bmi_norm = max(0, min(1, bmi_norm))
        
        return age_norm, bmi_norm
    
    def denormalize_bp(self, bp_pred: torch.Tensor) -> Tuple[float, float]:
        """Convert normalized BP predictions back to mmHg"""
        bp_numpy = bp_pred.detach().cpu().numpy()
        
        if len(bp_numpy.shape) > 1:
            bp_numpy = bp_numpy.squeeze()
        
        sys_pred = bp_numpy[0] * (self.sys_max - self.sys_min) + self.sys_min
        dia_pred = bp_numpy[1] * (self.dia_max - self.dia_min) + self.dia_min
        
        return float(sys_pred), float(dia_pred)
    
    def predict_single(self, ppg_signal: np.ndarray, age: float, bmi: float, 
                      preprocess: bool = True) -> Dict:
        """
        Predict blood pressure from a single PPG signal
        
        Args:
            ppg_signal: PPG signal (length should be chunk_length or will be padded/cropped)
            age: Age in years
            bmi: Body mass index
            preprocess: Whether to apply preprocessing
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Preprocess if needed
        if preprocess:
            ppg_signal = self.preprocess_ppg(ppg_signal)
        
        # Prepare input tensor
        chunk_length = self.config.get("chunk_length", 512)
        
        # Pad or crop to required length
        if len(ppg_signal) < chunk_length:
            # Pad with mean value
            pad_length = chunk_length - len(ppg_signal)
            mean_val = np.mean(ppg_signal)
            ppg_signal = np.concatenate([ppg_signal, np.full(pad_length, mean_val)])
        elif len(ppg_signal) > chunk_length:
            # Take the middle section
            start_idx = (len(ppg_signal) - chunk_length) // 2
            ppg_signal = ppg_signal[start_idx:start_idx + chunk_length]
        
        # Convert to tensor
        ppg_tensor = torch.tensor(ppg_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ppg_tensor = ppg_tensor.to(self.device)
        
        # Normalize demographics
        age_norm, bmi_norm = self.normalize_demographics(age, bmi)
        age_tensor = torch.tensor([[age_norm]], dtype=torch.float32).to(self.device)
        bmi_tensor = torch.tensor([[bmi_norm]], dtype=torch.float32).to(self.device)
        
        # Inference
        with torch.no_grad():
            bp_pred = self.model(ppg_tensor, age_tensor, bmi_tensor)
        
        # Denormalize predictions
        sys_pred, dia_pred = self.denormalize_bp(bp_pred)
        
        inference_time = time.time() - start_time
        
        return {
            'systolic': sys_pred,
            'diastolic': dia_pred,
            'age': age,
            'bmi': bmi,
            'inference_time': inference_time,
            'signal_length': len(ppg_signal),
            'model_input_shape': ppg_tensor.shape
        }
    
    def predict_batch(self, ppg_signals: List[np.ndarray], ages: List[float], 
                     bmis: List[float], preprocess: bool = True) -> List[Dict]:
        """Predict blood pressure for multiple PPG signals"""
        results = []
        
        print(f"Processing {len(ppg_signals)} signals...")
        for i, (ppg, age, bmi) in enumerate(zip(ppg_signals, ages, bmis)):
            try:
                result = self.predict_single(ppg, age, bmi, preprocess)
                result['sample_id'] = i
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(ppg_signals)} signals")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        return results

class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(self, config: Dict, output_dir: str = "./results"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def evaluate_demo_dataset(self, model_paths: List[str]) -> Dict:
        """Evaluate model performance on demo dataset"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        all_sys_gt, all_sys_pred = [], []
        all_dia_gt, all_dia_pred = [], []
        all_sessions = []
        
        # Process each fold
        for fold, model_path in enumerate(model_paths, 1):
            if not os.path.exists(model_path):
                print(f"Warning: Model {model_path} not found, skipping fold {fold}")
                continue
                
            print(f"\nEvaluating fold {fold}...")
            
            # Load model
            predictor = BloodPressurePredictor(model_path, self.config, device)
            
            # Load test dataset
            test_dataset = UWBP_test_manual_demo(self.config, fold)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False, 
                num_workers=4, drop_last=False, pin_memory=True
            )
            
            # Inference
            fold_sys_gt, fold_sys_pred = [], []
            fold_dia_gt, fold_dia_pred = [], []
            
            for i, data in enumerate(test_loader):
                temp_sys_gt, temp_sys_pred = [], []
                temp_dia_gt, temp_dia_pred = [], []
                
                for j, batch in enumerate(data):
                    chunks = batch["ppg_chunk"].to(device).float()
                    bp_gt = batch["bp"].to(device).float()
                    age = batch["age"].to(device).float()
                    bmi = batch["bmi"].to(device).float()
                    
                    with torch.no_grad():
                        bp_pred = predictor.model(chunks, age, bmi)
                    
                    temp_sys_gt.append(torch.squeeze(bp_gt)[0].cpu().numpy())
                    temp_sys_pred.append(torch.squeeze(bp_pred)[0].cpu().numpy())
                    temp_dia_gt.append(torch.squeeze(bp_gt)[1].cpu().numpy())
                    temp_dia_pred.append(torch.squeeze(bp_pred)[1].cpu().numpy())
                
                # Average over chunks for this session
                fold_sys_gt.append(np.mean(temp_sys_gt))
                fold_sys_pred.append(np.mean(temp_sys_pred))
                fold_dia_gt.append(np.mean(temp_dia_gt))
                fold_dia_pred.append(np.mean(temp_dia_pred))
                all_sessions.append(batch["session"])
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(test_loader)} samples")
            
            all_sys_gt.extend(fold_sys_gt)
            all_sys_pred.extend(fold_sys_pred)
            all_dia_gt.extend(fold_dia_gt)
            all_dia_pred.extend(fold_dia_pred)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_sys_gt, all_sys_pred, all_dia_gt, all_dia_pred)
        
        # Generate plots
        self._generate_evaluation_plots(all_sys_gt, all_sys_pred, all_dia_gt, all_dia_pred, metrics)
        
        # Save results
        self._save_results(all_sys_gt, all_sys_pred, all_dia_gt, all_dia_pred, all_sessions, metrics)
        
        return metrics
    
    def _calculate_metrics(self, sys_gt: List, sys_pred: List, 
                          dia_gt: List, dia_pred: List) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        # Convert to mmHg
        sys_gt_mmhg = np.array(sys_gt) * (180 - 100) + 100
        sys_pred_mmhg = np.array(sys_pred) * (180 - 100) + 100
        dia_gt_mmhg = np.array(dia_gt) * (100 - 55) + 55
        dia_pred_mmhg = np.array(dia_pred) * (100 - 55) + 55
        
        # Calculate metrics
        sys_mae = np.mean(np.abs(sys_pred_mmhg - sys_gt_mmhg))
        dia_mae = np.mean(np.abs(dia_pred_mmhg - dia_gt_mmhg))
        
        sys_rmse = np.sqrt(np.mean((sys_pred_mmhg - sys_gt_mmhg)**2))
        dia_rmse = np.sqrt(np.mean((dia_pred_mmhg - dia_gt_mmhg)**2))
        
        sys_corr = np.corrcoef(sys_gt_mmhg, sys_pred_mmhg)[0, 1]
        dia_corr = np.corrcoef(dia_gt_mmhg, dia_pred_mmhg)[0, 1]
        
        sys_std = np.std(sys_pred_mmhg - sys_gt_mmhg)
        dia_std = np.std(dia_pred_mmhg - dia_gt_mmhg)
        
        sys_mean_error = np.mean(sys_pred_mmhg - sys_gt_mmhg)
        dia_mean_error = np.mean(dia_pred_mmhg - dia_gt_mmhg)
        
        return {
            'systolic': {
                'mae': sys_mae, 'rmse': sys_rmse, 'correlation': sys_corr,
                'std': sys_std, 'mean_error': sys_mean_error
            },
            'diastolic': {
                'mae': dia_mae, 'rmse': dia_rmse, 'correlation': dia_corr,
                'std': dia_std, 'mean_error': dia_mean_error
            }
        }
    
    def _generate_evaluation_plots(self, sys_gt: List, sys_pred: List, 
                                  dia_gt: List, dia_pred: List, metrics: Dict):
        """Generate comprehensive evaluation plots"""
        # Convert to mmHg
        sys_gt_mmhg = np.array(sys_gt) * (180 - 100) + 100
        sys_pred_mmhg = np.array(sys_pred) * (180 - 100) + 100
        dia_gt_mmhg = np.array(dia_gt) * (100 - 55) + 55
        dia_pred_mmhg = np.array(dia_pred) * (100 - 55) + 55
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('rPPG Blood Pressure Estimation Results', fontsize=16, fontweight='bold')
        
        # Systolic scatter plot
        axes[0, 0].scatter(sys_gt_mmhg, sys_pred_mmhg, alpha=0.6, color='red')
        axes[0, 0].plot([100, 180], [100, 180], 'k--', alpha=0.8)
        axes[0, 0].set_xlabel('Ground Truth Systolic BP (mmHg)')
        axes[0, 0].set_ylabel('Predicted Systolic BP (mmHg)')
        axes[0, 0].set_title(f'Systolic BP (r={metrics["systolic"]["correlation"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Diastolic scatter plot
        axes[0, 1].scatter(dia_gt_mmhg, dia_pred_mmhg, alpha=0.6, color='blue')
        axes[0, 1].plot([55, 100], [55, 100], 'k--', alpha=0.8)
        axes[0, 1].set_xlabel('Ground Truth Diastolic BP (mmHg)')
        axes[0, 1].set_ylabel('Predicted Diastolic BP (mmHg)')
        axes[0, 1].set_title(f'Diastolic BP (r={metrics["diastolic"]["correlation"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution
        sys_errors = sys_pred_mmhg - sys_gt_mmhg
        dia_errors = dia_pred_mmhg - dia_gt_mmhg
        
        axes[0, 2].hist(sys_errors, bins=30, alpha=0.7, color='red', label='Systolic')
        axes[0, 2].hist(dia_errors, bins=30, alpha=0.7, color='blue', label='Diastolic')
        axes[0, 2].set_xlabel('Prediction Error (mmHg)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bland-Altman plots
        sys_mean = (sys_gt_mmhg + sys_pred_mmhg) / 2
        dia_mean = (dia_gt_mmhg + dia_pred_mmhg) / 2
        
        axes[1, 0].scatter(sys_mean, sys_errors, alpha=0.6, color='red')
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.8)
        axes[1, 0].axhline(y=np.mean(sys_errors), color='r', linestyle='--', alpha=0.8)
        axes[1, 0].axhline(y=np.mean(sys_errors) + 1.96*np.std(sys_errors), color='r', linestyle=':', alpha=0.8)
        axes[1, 0].axhline(y=np.mean(sys_errors) - 1.96*np.std(sys_errors), color='r', linestyle=':', alpha=0.8)
        axes[1, 0].set_xlabel('Mean Systolic BP (mmHg)')
        axes[1, 0].set_ylabel('Difference (Pred - GT)')
        axes[1, 0].set_title('Bland-Altman: Systolic')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].scatter(dia_mean, dia_errors, alpha=0.6, color='blue')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.8)
        axes[1, 1].axhline(y=np.mean(dia_errors), color='b', linestyle='--', alpha=0.8)
        axes[1, 1].axhline(y=np.mean(dia_errors) + 1.96*np.std(dia_errors), color='b', linestyle=':', alpha=0.8)
        axes[1, 1].axhline(y=np.mean(dia_errors) - 1.96*np.std(dia_errors), color='b', linestyle=':', alpha=0.8)
        axes[1, 1].set_xlabel('Mean Diastolic BP (mmHg)')
        axes[1, 1].set_ylabel('Difference (Pred - GT)')
        axes[1, 1].set_title('Bland-Altman: Diastolic')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Metrics summary
        axes[1, 2].axis('off')
        metrics_text = f"""
        SYSTOLIC BP:
        MAE: {metrics['systolic']['mae']:.2f} mmHg
        RMSE: {metrics['systolic']['rmse']:.2f} mmHg
        Correlation: {metrics['systolic']['correlation']:.3f}
        Mean Error: {metrics['systolic']['mean_error']:.2f} mmHg
        Std Error: {metrics['systolic']['std']:.2f} mmHg
        
        DIASTOLIC BP:
        MAE: {metrics['diastolic']['mae']:.2f} mmHg
        RMSE: {metrics['diastolic']['rmse']:.2f} mmHg
        Correlation: {metrics['diastolic']['correlation']:.3f}
        Mean Error: {metrics['diastolic']['mean_error']:.2f} mmHg
        Std Error: {metrics['diastolic']['std']:.2f} mmHg
        """
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Evaluation plots saved to {self.output_dir}")
    
    def _save_results(self, sys_gt: List, sys_pred: List, dia_gt: List, 
                     dia_pred: List, sessions: List, metrics: Dict):
        """Save detailed results to files"""
        # Convert to mmHg
        sys_gt_mmhg = np.array(sys_gt) * (180 - 100) + 100
        sys_pred_mmhg = np.array(sys_pred) * (180 - 100) + 100
        dia_gt_mmhg = np.array(dia_gt) * (100 - 55) + 55
        dia_pred_mmhg = np.array(dia_pred) * (100 - 55) + 55
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'session_id': [s[0] if isinstance(s, list) else s for s in sessions],
            'sys_gt_mmhg': sys_gt_mmhg,
            'sys_pred_mmhg': sys_pred_mmhg,
            'dia_gt_mmhg': dia_gt_mmhg,
            'dia_pred_mmhg': dia_pred_mmhg,
            'sys_error': sys_pred_mmhg - sys_gt_mmhg,
            'dia_error': dia_pred_mmhg - dia_gt_mmhg
        })
        
        # Save to CSV
        results_df.to_csv(os.path.join(self.output_dir, 'detailed_results.csv'), index=False)
        
        # Save metrics to JSON
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Results saved to {self.output_dir}")

def create_sample_config() -> Dict:
    """Create a sample configuration for testing"""
    return {
        "prefix": "ppg_bp_demo",
        "manual_seed": 6666,
        "chunk_amount": 5,
        "chunk_length": 512,
        "filter_lpf": 0.7,
        "filter_hpf": 10,
        "frequency_sample": 60,
        "frequency_field": False,
        "derivative_input": False,
        "output_dir": "./results"
    }

def main():
    parser = argparse.ArgumentParser(description="rPPG Blood Pressure Estimation Application")
    parser.add_argument("--mode", type=str, choices=['single', 'demo', 'batch', 'compare'], 
                       default='demo', help="Operation mode")
    
    # Single prediction mode
    parser.add_argument("--ppg_file", type=str, help="PPG signal file (.npy)")
    parser.add_argument("--age", type=float, help="Age in years")
    parser.add_argument("--bmi", type=float, help="Body Mass Index")
    parser.add_argument("--model_path", type=str, help="Path to trained model")
    
    # Demo/Compare mode
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    # Batch mode
    parser.add_argument("--input_dir", type=str, help="Input directory for batch processing")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    
    # General options
    parser.add_argument("--device", type=str, default="auto", choices=['auto', 'cpu', 'cuda'],
                       help="Device to use for inference")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    
    args = parser.parse_args()
    
    print("🩺 rPPG Blood Pressure Estimation Application")
    print("=" * 50)
    
    try:
        if args.mode == 'single':
            # Single prediction mode
            if not all([args.ppg_file, args.age, args.bmi, args.model_path]):
                print("❌ Error: Single mode requires --ppg_file, --age, --bmi, and --model_path")
                return
            
            # Load PPG signal
            ppg_signal = np.load(args.ppg_file)
            config = create_sample_config()
            
            # Create predictor and predict
            predictor = BloodPressurePredictor(args.model_path, config, args.device)
            result = predictor.predict_single(ppg_signal, args.age, args.bmi)
            
            print(f"\n📊 Prediction Results:")
            print(f"   Systolic BP:  {result['systolic']:.1f} mmHg")
            print(f"   Diastolic BP: {result['diastolic']:.1f} mmHg")
            print(f"   Inference time: {result['inference_time']:.3f} seconds")
            
        elif args.mode == 'demo':
            # Demo mode with existing dataset
            if not args.config:
                print("❌ Error: Demo mode requires --config")
                return
            
            # Load configuration
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # For demo, we'll use a sample model path (you may need to adjust this)
            sample_model = "output/sample_model.pth"  # Placeholder
            if not os.path.exists(sample_model):
                print(f"⚠️  Warning: Sample model {sample_model} not found.")
                print("   Please provide a valid model path or use a different mode.")
                return
            
            config = create_sample_config()
            predictor = BloodPressurePredictor(sample_model, config, args.device)
            
            # Generate sample prediction
            sample_ppg = np.random.randn(512) * 0.1 + np.sin(np.linspace(0, 20*np.pi, 512)) * 0.5
            result = predictor.predict_single(sample_ppg, age=45, bmi=24.5)
            
            print(f"\n📊 Demo Prediction Results:")
            print(f"   Systolic BP:  {result['systolic']:.1f} mmHg")
            print(f"   Diastolic BP: {result['diastolic']:.1f} mmHg")
            
        elif args.mode == 'compare':
            # Model comparison mode
            if not args.config:
                print("❌ Error: Compare mode requires --config")
                return
            
            # Load test configuration
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # Extract model paths
            model_paths = []
            for i in range(1, 6):
                path_key = f"model_path{i}"
                if path_key in config:
                    model_paths.append(config[path_key])
            
            if not model_paths:
                print("❌ Error: No model paths found in configuration")
                return
            
            print(f"📈 Evaluating {len(model_paths)} models...")
            
            # Create evaluator and run evaluation
            evaluator = ModelEvaluator(config, args.output_dir)
            metrics = evaluator.evaluate_demo_dataset(model_paths)
            
            print(f"\n📊 Overall Results:")
            print(f"   Systolic MAE:  {metrics['systolic']['mae']:.2f} mmHg")
            print(f"   Diastolic MAE: {metrics['diastolic']['mae']:.2f} mmHg")
            print(f"   Systolic Correlation:  {metrics['systolic']['correlation']:.3f}")
            print(f"   Diastolic Correlation: {metrics['diastolic']['correlation']:.3f}")
            
        elif args.mode == 'batch':
            # Batch processing mode
            if not args.input_dir:
                print("❌ Error: Batch mode requires --input_dir")
                return
            
            print(f"🔄 Batch processing not fully implemented yet.")
            print(f"   Input directory: {args.input_dir}")
            print(f"   Output directory: {args.output_dir}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 