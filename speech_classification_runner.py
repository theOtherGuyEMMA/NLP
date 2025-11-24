#!/usr/bin/env python3
"""
Swahili Speech Classification Runner
===================================
A comprehensive script for Swahili speech classification with terminal output.
Implements data loading, preprocessing, model training, and evaluation.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import colorama
    from colorama import Fore, Back, Style, init
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback to no colors
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Back:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = RESET_ALL = ""

def print_header(text, color=Fore.CYAN):
    """Print a formatted header with colors."""
    print(f"\n{color}{'='*60}")
    print(f"{text.center(60)}")
    print(f"{'='*60}{Style.RESET_ALL}\n")

def print_section(text, color=Fore.YELLOW):
    """Print a section header."""
    print(f"\n{color}{'─'*50}")
    print(f"📊 {text}")
    print(f"{'─'*50}{Style.RESET_ALL}")

def print_success(text):
    """Print success message."""
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")

def print_error(text):
    """Print error message."""
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")

def print_warning(text):
    """Print warning message."""
    print(f"{Fore.YELLOW}⚠️  {text}{Style.RESET_ALL}")

def print_info(text):
    """Print info message."""
    print(f"{Fore.BLUE}ℹ️  {text}{Style.RESET_ALL}")

class SpeechClassificationRunner:
    """Main class for running speech classification pipeline."""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create figures directory
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        print_section("Checking Dependencies")
        
        # Map import names to pip package names for accurate guidance
        required = {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'sklearn': 'scikit-learn',
            'librosa': 'librosa',
            'datasets': 'datasets',
        }

        missing_imports = []
        missing_pips = []

        for import_name, pip_name in required.items():
            try:
                __import__(import_name)
                print_success(f"{import_name} is installed")
            except ImportError:
                missing_imports.append(import_name)
                missing_pips.append(pip_name)
                print_error(f"{import_name} is missing")

        if missing_imports:
            print_error(f"Missing packages: {', '.join(missing_imports)}")
            print_info("Install with: pip install " + " ".join(missing_pips))
            return False
        
        print_success("All dependencies are available!")
        return True
    
    def load_sample_data(self):
        """Load or create sample data for demonstration."""
        print_section("Loading Sample Data")
        
        try:
            # Try to load Common Voice dataset
            print_info("Attempting to load Common Voice Swahili dataset...")
            
            from datasets import load_dataset
            
            # Load a small subset for demonstration
            dataset = load_dataset("mozilla-foundation/common_voice_11_0", "sw", 
                                 split="train[:100]", trust_remote_code=True)
            
            print_success(f"Loaded {len(dataset)} samples from Common Voice")
            
            # Extract basic info
            sample_rate = dataset[0]['audio']['sampling_rate']
            print_info(f"Sample rate: {sample_rate} Hz")
            
            # Create proxy labels from transcripts
            transcripts = [item['sentence'] for item in dataset]
            
            # Simple word-based classification (demo purposes)
            common_words = ['na', 'ya', 'wa', 'ni', 'la']
            labels = []
            
            for transcript in transcripts:
                words = transcript.lower().split()
                label = 0  # default
                for i, word in enumerate(common_words):
                    if word in words:
                        label = i
                        break
                labels.append(label)
            
            self.results['dataset_size'] = len(dataset)
            self.results['sample_rate'] = sample_rate
            self.results['num_classes'] = len(set(labels))
            
            print_success(f"Created {len(set(labels))} classes from transcripts")
            
            return dataset, labels
            
        except Exception as e:
            print_warning(f"Could not load Common Voice: {str(e)}")
            print_info("Creating synthetic data for demonstration...")
            
            # Create synthetic data
            np.random.seed(42)
            n_samples = 100
            sample_rate = 16000
            duration = 2.0
            
            # Generate synthetic audio data
            synthetic_data = []
            synthetic_labels = []
            
            for i in range(n_samples):
                # Create synthetic audio with different frequency patterns
                t = np.linspace(0, duration, int(sample_rate * duration))
                label = i % 5  # 5 classes
                
                # Different frequency patterns for each class
                freq = 200 + label * 100
                audio = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
                
                synthetic_data.append({
                    'audio': {'array': audio, 'sampling_rate': sample_rate},
                    'sentence': f"synthetic_sample_{i}"
                })
                synthetic_labels.append(label)
            
            self.results['dataset_size'] = n_samples
            self.results['sample_rate'] = sample_rate
            self.results['num_classes'] = 5
            
            print_success(f"Created {n_samples} synthetic samples with {5} classes")
            
            return synthetic_data, synthetic_labels
    
    def preprocess_data(self, dataset, labels):
        """Preprocess audio data and extract features."""
        print_section("Preprocessing Data")
        
        try:
            import librosa
            from sklearn.preprocessing import StandardScaler
            
            print_info("Extracting MFCC features...")
            
            features = []
            valid_labels = []
            
            for i, (sample, label) in enumerate(zip(dataset, labels)):
                try:
                    audio = sample['audio']['array']
                    sr = sample['audio']['sampling_rate']
                    
                    # Extract MFCC features
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfccs, axis=1)
                    
                    features.append(mfcc_mean)
                    valid_labels.append(label)
                    
                    if (i + 1) % 20 == 0:
                        print_info(f"Processed {i + 1}/{len(dataset)} samples")
                        
                except Exception as e:
                    print_warning(f"Skipping sample {i}: {str(e)}")
                    continue
            
            features = np.array(features)
            valid_labels = np.array(valid_labels)
            
            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            
            self.results['feature_dim'] = features.shape[1]
            self.results['processed_samples'] = len(features)
            
            print_success(f"Extracted features: {features.shape}")
            print_success(f"Feature dimension: {features.shape[1]}")
            
            return features_normalized, valid_labels, scaler
            
        except ImportError:
            print_error("librosa not available, using simple features")
            
            # Fallback to simple statistical features
            features = []
            for sample in dataset:
                audio = sample['audio']['array']
                # Simple statistical features
                feat = [
                    np.mean(audio), np.std(audio), np.max(audio), np.min(audio),
                    np.median(audio), np.var(audio)
                ]
                features.append(feat)
            
            features = np.array(features)
            labels = np.array(labels)
            
            print_success(f"Extracted simple features: {features.shape}")
            return features, labels, None
    
    def train_model(self, X, y):
        """Train classification models."""
        print_section("Training Models")
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print_info(f"Training set: {X_train.shape[0]} samples")
        print_info(f"Test set: {X_test.shape[0]} samples")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print_info(f"Training {name}...")
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                # Predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'train_time': train_time,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                print_success(f"{name} - Accuracy: {accuracy:.3f}, Time: {train_time:.2f}s")
                
            except Exception as e:
                print_error(f"Failed to train {name}: {str(e)}")
        
        self.results['models'] = results
        return results, X_test, y_test
    
    def evaluate_models(self, model_results):
        """Evaluate and display model performance."""
        print_section("Model Evaluation")
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        for name, result in model_results.items():
            print(f"\n{Fore.MAGENTA}{'─'*30}")
            print(f"📈 {name} Results")
            print(f"{'─'*30}{Style.RESET_ALL}")
            
            y_test = result['y_test']
            y_pred = result['y_pred']
            
            print(f"Accuracy: {Fore.GREEN}{result['accuracy']:.3f}{Style.RESET_ALL}")
            print(f"Training Time: {result['train_time']:.2f}s")
            
            # Classification report
            print(f"\n{Fore.CYAN}Classification Report:{Style.RESET_ALL}")
            report = classification_report(y_test, y_pred)
            print(report)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n{Fore.CYAN}Confusion Matrix:{Style.RESET_ALL}")
            print(cm)
    
    def create_visualizations(self, model_results):
        """Create and save visualizations."""
        print_section("Creating Visualizations")

        try:
            from sklearn.metrics import confusion_matrix
            # Model comparison plot
            plt.figure(figsize=(10, 6))
            
            names = list(model_results.keys())
            accuracies = [result['accuracy'] for result in model_results.values()]
            times = [result['train_time'] for result in model_results.values()]
            
            # Accuracy comparison
            plt.subplot(1, 2, 1)
            bars = plt.bar(names, accuracies, color=['skyblue', 'lightcoral'])
            plt.title('Model Accuracy Comparison')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            # Training time comparison
            plt.subplot(1, 2, 2)
            bars = plt.bar(names, times, color=['lightgreen', 'orange'])
            plt.title('Training Time Comparison')
            plt.ylabel('Time (seconds)')
            
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{time_val:.2f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.figures_dir / "model_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print_success(f"Saved model comparison plot: {plot_path}")
            
            # Confusion matrix for best model
            best_model_name = max(model_results.keys(), 
                                key=lambda k: model_results[k]['accuracy'])
            best_result = model_results[best_model_name]
            
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {best_model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = self.figures_dir / f"confusion_matrix_{best_model_name.lower().replace(' ', '_')}.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print_success(f"Saved confusion matrix: {cm_path}")
            
        except Exception as e:
            print_error(f"Failed to create visualizations: {str(e)}")
    
    def generate_report(self):
        """Generate a summary report."""
        print_section("Generating Summary Report")
        
        report_path = self.output_dir / "classification_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Swahili Speech Classification Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"- Total samples: {self.results.get('dataset_size', 'N/A')}\n")
            f.write(f"- Processed samples: {self.results.get('processed_samples', 'N/A')}\n")
            f.write(f"- Number of classes: {self.results.get('num_classes', 'N/A')}\n")
            f.write(f"- Sample rate: {self.results.get('sample_rate', 'N/A')} Hz\n")
            f.write(f"- Feature dimension: {self.results.get('feature_dim', 'N/A')}\n\n")
            
            if 'models' in self.results:
                f.write("Model Performance:\n")
                for name, result in self.results['models'].items():
                    f.write(f"- {name}: {result['accuracy']:.3f} accuracy, {result['train_time']:.2f}s training\n")
        
        print_success(f"Report saved: {report_path}")
    
    def run_pipeline(self):
        """Run the complete speech classification pipeline."""
        print_header("🎤 Swahili Speech Classification Pipeline", Fore.CYAN)
        
        start_time = time.time()
        
        try:
            # Check dependencies
            if not self.check_dependencies():
                if not self.args.skip_deps:
                    print_error("Dependencies missing. Use --skip-deps to continue anyway.")
                    return False
            
            # Load data
            dataset, labels = self.load_sample_data()
            
            # Preprocess
            X, y, scaler = self.preprocess_data(dataset, labels)
            
            # Train models
            model_results, X_test, y_test = self.train_model(X, y)
            
            # Evaluate
            self.evaluate_models(model_results)
            
            # Create visualizations
            if not self.args.no_plots:
                self.create_visualizations(model_results)
            
            # Generate report
            self.generate_report()
            
            total_time = time.time() - start_time
            
            print_header("🎉 Pipeline Completed Successfully!", Fore.GREEN)
            print_info(f"Total execution time: {total_time:.2f} seconds")
            print_info(f"Results saved in: {self.output_dir}")
            
            return True
            
        except KeyboardInterrupt:
            print_error("\nPipeline interrupted by user")
            return False
        except Exception as e:
            print_error(f"Pipeline failed: {str(e)}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
            return False

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Swahili Speech Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speech_classification_runner.py
  python speech_classification_runner.py --output-dir results --no-plots
  python speech_classification_runner.py --skip-deps --debug
        """
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots and visualizations'
    )
    
    parser.add_argument(
        '--skip-deps',
        action='store_true',
        help='Skip dependency checking'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with full error traces'
    )
    
    args = parser.parse_args()
    
    # Print startup info
    print_header("🚀 Starting Speech Classification Runner")
    print_info(f"Output directory: {args.output_dir}")
    print_info(f"Plots enabled: {not args.no_plots}")
    print_info(f"Debug mode: {args.debug}")
    
    if not COLORS_AVAILABLE:
        print_warning("colorama not installed - using plain text output")
        print_info("Install colorama for colored output: pip install colorama")
    
    # Run pipeline
    runner = SpeechClassificationRunner(args)
    success = runner.run_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()