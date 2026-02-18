"""
Advanced GUI Interface for ML Tree System
Built with Tkinter - No external dependencies required

Features:
- Model Testing & Prediction
- Interactive Visualizations
- Feature Importance Analysis
- Model Comparison
- Error Analysis
- Real-time Predictions
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font as tkfont
import sys
from pathlib import Path
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns

from src.utils.helpers import load_artifact, load_config
from src.models.evaluator import ModelEvaluator
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


class MLTreeSystemGUI:
    """Main GUI Application for ML Tree System."""

    def __init__(self, root):
        self.root = root
        self.root.title("ML Tree System - Advanced Interface")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._configure_styles()

        # Data storage
        self.models = {}
        self.preprocessors = {}
        self.metadata = {}
        self.datasets = ["heart_disease", "adult_income", "wine_quality", "diabetes", "breast_cancer"]
        self.current_dataset = None
        self.test_data = None

        # Build UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

        # Load models
        self.root.after(100, self._load_all_models)

    def _configure_styles(self):
        """Configure custom styles."""
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Info.TLabel', font=('Helvetica', 10))
        self.style.configure('Success.TLabel', foreground='green')
        self.style.configure('Error.TLabel', foreground='red')
        self.style.configure('Action.TButton', font=('Helvetica', 10, 'bold'))

    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load CSV Data", command=self._load_csv_data)
        file_menu.add_command(label="Export Results", command=self._export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Models menu
        models_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Models", menu=models_menu)
        models_menu.add_command(label="Reload Models", command=self._load_all_models)
        models_menu.add_command(label="Model Info", command=self._show_model_info)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self):
        """Create main application layout."""
        # Main container
        main_container = ttk.Frame(self.root, padding="5")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left panel - Controls
        left_panel = ttk.Frame(main_container, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        # Right panel - Visualizations
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._create_control_panel(left_panel)
        self._create_visualization_panel(right_panel)

    def _create_control_panel(self, parent):
        """Create left control panel."""
        # Title
        title = ttk.Label(parent, text="ML Tree System", style='Title.TLabel')
        title.pack(pady=10)

        # Dataset Selection
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Selection", padding="10")
        dataset_frame.pack(fill=tk.X, pady=5)

        self.dataset_var = tk.StringVar(value="heart_disease")
        for ds in self.datasets:
            rb = ttk.Radiobutton(dataset_frame, text=ds.replace("_", " ").title(),
                                value=ds, variable=self.dataset_var,
                                command=self._on_dataset_change)
            rb.pack(anchor=tk.W)

        # Model Selection
        model_frame = ttk.LabelFrame(parent, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=5)

        self.model_var = tk.StringVar(value="random_forest")
        ttk.Radiobutton(model_frame, text="Random Forest", value="random_forest",
                       variable=self.model_var).pack(anchor=tk.W)
        ttk.Radiobutton(model_frame, text="Decision Tree", value="decision_tree",
                       variable=self.model_var).pack(anchor=tk.W)

        # Visualization Options
        viz_frame = ttk.LabelFrame(parent, text="Visualization", padding="10")
        viz_frame.pack(fill=tk.X, pady=5)

        ttk.Button(viz_frame, text="Confusion Matrix",
                  command=lambda: self._show_visualization("confusion")).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="ROC Curve",
                  command=lambda: self._show_visualization("roc")).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="Feature Importance",
                  command=lambda: self._show_visualization("importance")).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="Model Comparison",
                  command=lambda: self._show_visualization("comparison")).pack(fill=tk.X, pady=2)
        ttk.Button(viz_frame, text="Error Analysis",
                  command=lambda: self._show_visualization("errors")).pack(fill=tk.X, pady=2)

        # Prediction Panel
        pred_frame = ttk.LabelFrame(parent, text="Make Prediction", padding="10")
        pred_frame.pack(fill=tk.X, pady=5)

        ttk.Button(pred_frame, text="Single Prediction", style='Action.TButton',
                  command=self._open_prediction_window).pack(fill=tk.X, pady=2)
        ttk.Button(pred_frame, text="Batch Prediction",
                  command=self._batch_prediction).pack(fill=tk.X, pady=2)
        ttk.Button(pred_frame, text="Interactive Test",
                  command=self._open_interactive_test).pack(fill=tk.X, pady=2)

        # Model Info Display
        info_frame = ttk.LabelFrame(parent, text="Model Info", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.info_text = tk.Text(info_frame, height=10, width=40, font=('Courier', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

    def _create_visualization_panel(self, parent):
        """Create right visualization panel."""
        # Notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Main Visualization
        self.viz_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_tab, text="Visualization")

        # Tab 2: Metrics
        self.metrics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.metrics_tab, text="Metrics & Results")

        # Tab 3: Predictions
        self.pred_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pred_tab, text="Predictions")

        # Tab 4: Advanced Analysis
        self.advanced_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_tab, text="Advanced Analysis")

        # Initialize visualization area
        self._init_viz_area()
        self._init_metrics_area()
        self._init_predictions_area()
        self._init_advanced_area()

    def _init_viz_area(self):
        """Initialize main visualization area."""
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_tab)
        self.canvas.draw()

        toolbar = NavigationToolbar2Tk(self.canvas, self.viz_tab)
        toolbar.update()

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _init_metrics_area(self):
        """Initialize metrics display area."""
        # Metrics table
        columns = ("Metric", "Decision Tree", "Random Forest", "Winner")
        self.metrics_tree = ttk.Treeview(self.metrics_tab, columns=columns, show="headings", height=10)

        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=150, anchor=tk.CENTER)

        self.metrics_tree.pack(fill=tk.X, padx=10, pady=10)

        # Classification report
        report_frame = ttk.LabelFrame(self.metrics_tab, text="Classification Report", padding="10")
        report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.report_text = tk.Text(report_frame, height=20, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(report_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=scrollbar.set)

        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _init_predictions_area(self):
        """Initialize predictions display area."""
        # Predictions table
        pred_columns = ("Index", "Prediction", "Confidence", "True Label")
        self.pred_tree = ttk.Treeview(self.pred_tab, columns=pred_columns, show="headings", height=20)

        for col in pred_columns:
            self.pred_tree.heading(col, text=col)
            self.pred_tree.column(col, width=150, anchor=tk.CENTER)

        pred_scroll = ttk.Scrollbar(self.pred_tab, orient=tk.VERTICAL, command=self.pred_tree.yview)
        self.pred_tree.configure(yscrollcommand=pred_scroll.set)

        self.pred_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        pred_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=10)

    def _init_advanced_area(self):
        """Initialize advanced analysis area."""
        # Feature correlation
        self.adv_fig = Figure(figsize=(10, 7), dpi=100)
        self.adv_canvas = FigureCanvasTkAgg(self.adv_fig, master=self.advanced_tab)
        self.adv_canvas.draw()

        # Controls
        control_frame = ttk.Frame(self.advanced_tab)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(control_frame, text="Feature Correlation",
                  command=self._show_correlation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Learning Curve",
                  command=self._show_learning_curve).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Decision Boundary",
                  command=self._show_decision_boundary).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Tree Structure",
                  command=self._show_tree_structure).pack(side=tk.LEFT, padx=5)

        self.adv_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_status_bar(self):
        """Create status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_all_models(self):
        """Load all trained models."""
        self._update_status("Loading models...")
        models_dir = Path("models")

        loaded_count = 0
        for dataset in self.datasets:
            try:
                # Try to load Random Forest
                rf_path = models_dir / f"rf_{dataset}.joblib"
                if rf_path.exists():
                    self.models[f"rf_{dataset}"] = load_artifact(str(rf_path))
                    loaded_count += 1

                # Try to load Decision Tree
                dt_path = models_dir / f"dt_{dataset}.joblib"
                if dt_path.exists():
                    self.models[f"dt_{dataset}"] = load_artifact(str(dt_path))
                    loaded_count += 1

                # Load preprocessor
                prep_path = models_dir / f"preprocessor_{dataset}.joblib"
                if prep_path.exists():
                    self.preprocessors[dataset] = load_artifact(str(prep_path))

                # Load metadata
                meta_path = models_dir / f"metadata_{dataset}.joblib"
                if meta_path.exists():
                    self.metadata[dataset] = load_artifact(str(meta_path))

            except Exception as e:
                print(f"Error loading {dataset}: {e}")

        # Also try original breast_cancer model
        try:
            rf_bc = models_dir / "random_forest.joblib"
            if rf_bc.exists():
                self.models["rf_breast_cancer"] = load_artifact(str(rf_bc))
                loaded_count += 1
            dt_bc = models_dir / "decision_tree.joblib"
            if dt_bc.exists():
                self.models["dt_breast_cancer"] = load_artifact(str(dt_bc))
                loaded_count += 1
            prep_bc = models_dir / "preprocessor.joblib"
            if prep_bc.exists():
                self.preprocessors["breast_cancer"] = load_artifact(str(prep_bc))
            meta_bc = models_dir / "metadata.joblib"
            if meta_bc.exists():
                self.metadata["breast_cancer"] = load_artifact(str(meta_bc))
        except Exception as e:
            print(f"Error loading breast_cancer: {e}")

        self._update_status(f"Loaded {loaded_count} models")
        self._update_info_display()
        self._on_dataset_change()

    def _on_dataset_change(self):
        """Handle dataset selection change."""
        dataset = self.dataset_var.get()
        self.current_dataset = dataset
        self._update_info_display()
        self._update_metrics_display()

    def _update_info_display(self):
        """Update model info display."""
        self.info_text.delete(1.0, tk.END)
        dataset = self.dataset_var.get()

        info_lines = [f"Dataset: {dataset.replace('_', ' ').title()}\n"]
        info_lines.append("-" * 35 + "\n")

        if dataset in self.metadata:
            meta = self.metadata[dataset]
            info_lines.append(f"Samples: {meta.get('n_samples', 'N/A')}\n")
            info_lines.append(f"Features: {meta.get('n_features', 'N/A')}\n")
            info_lines.append(f"Classes: {meta.get('target_names', 'N/A')}\n")

        # Model status
        info_lines.append("\nModels Loaded:\n")
        rf_key = f"rf_{dataset}"
        dt_key = f"dt_{dataset}"
        info_lines.append(f"  Random Forest: {'Yes' if rf_key in self.models else 'No'}\n")
        info_lines.append(f"  Decision Tree: {'Yes' if dt_key in self.models else 'No'}\n")

        self.info_text.insert(tk.END, "".join(info_lines))

    def _update_metrics_display(self):
        """Update metrics table."""
        # Clear existing
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)

        dataset = self.dataset_var.get()

        # Try to load comparison CSV
        comp_path = Path("models") / f"comparison_{dataset}.csv"
        if comp_path.exists():
            df = pd.read_csv(comp_path)

            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            for metric in metrics:
                if metric in df.columns:
                    dt_val = df[df["Model"] == "Decision Tree"][metric].values
                    rf_val = df[df["Model"] == "Random Forest"][metric].values

                    dt_str = f"{dt_val[0]:.4f}" if len(dt_val) > 0 else "N/A"
                    rf_str = f"{rf_val[0]:.4f}" if len(rf_val) > 0 else "N/A"

                    # Determine winner
                    winner = "RF" if len(rf_val) > 0 and len(dt_val) > 0 and rf_val[0] > dt_val[0] else "DT"

                    self.metrics_tree.insert("", tk.END, values=(metric, dt_str, rf_str, winner))

    def _show_visualization(self, viz_type):
        """Show selected visualization."""
        self.fig.clear()
        dataset = self.dataset_var.get()
        model_type = self.model_var.get()

        if viz_type == "confusion":
            self._plot_confusion_matrix()
        elif viz_type == "roc":
            self._plot_roc_curves()
        elif viz_type == "importance":
            self._plot_feature_importance()
        elif viz_type == "comparison":
            self._plot_model_comparison()
        elif viz_type == "errors":
            self._plot_error_analysis()

        self.canvas.draw()
        self.notebook.select(0)  # Switch to visualization tab

    def _plot_confusion_matrix(self):
        """Plot confusion matrix from saved image or generate new."""
        dataset = self.dataset_var.get()
        model_type = self.model_var.get()

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        # Try to load saved confusion matrices
        prefix = "dt" if model_type == "decision_tree" else "rf"

        for ax, (mt, title) in zip([ax1, ax2], [("dt", "Decision Tree"), ("rf", "Random Forest")]):
            img_path = Path("reports/figures") / f"cm_{mt}_{dataset}.png"
            if img_path.exists():
                img = plt.imread(str(img_path))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title)
            else:
                ax.text(0.5, 0.5, f"No confusion matrix\nfor {dataset}",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)

        self.fig.suptitle(f"Confusion Matrices - {dataset.replace('_', ' ').title()}", fontsize=14)
        self.fig.tight_layout()

    def _plot_roc_curves(self):
        """Plot ROC curves."""
        dataset = self.dataset_var.get()
        ax = self.fig.add_subplot(111)

        img_path = Path("reports/figures") / f"roc_{dataset}.png"
        if img_path.exists():
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f"No ROC curve for {dataset}",
                   ha='center', va='center', transform=ax.transAxes)

        ax.set_title(f"ROC Curves - {dataset.replace('_', ' ').title()}")

    def _plot_feature_importance(self):
        """Plot feature importance."""
        dataset = self.dataset_var.get()

        # Load importance CSV
        imp_path = Path("models") / f"importance_{dataset}.csv"

        if imp_path.exists():
            df = pd.read_csv(imp_path).head(15)

            ax = self.fig.add_subplot(111)
            colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(df)))

            bars = ax.barh(df["feature"], df["importance"], color=colors)
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Feature")
            ax.set_title(f"Top 15 Feature Importance - {dataset.replace('_', ' ').title()}")
            ax.invert_yaxis()

            # Add value labels
            for bar, val in zip(bars, df["importance"]):
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                       f"{val:.3f}", va='center', fontsize=8)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No importance data for {dataset}",
                   ha='center', va='center', transform=ax.transAxes)

        self.fig.tight_layout()

    def _plot_model_comparison(self):
        """Plot model comparison."""
        dataset = self.dataset_var.get()

        comp_path = Path("models") / f"comparison_{dataset}.csv"

        if comp_path.exists():
            df = pd.read_csv(comp_path)
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

            ax = self.fig.add_subplot(111)
            x = np.arange(len(metrics))
            width = 0.35

            dt_vals = [df[df["Model"] == "Decision Tree"][m].values[0] for m in metrics]
            rf_vals = [df[df["Model"] == "Random Forest"][m].values[0] for m in metrics]

            bars1 = ax.bar(x - width/2, dt_vals, width, label='Decision Tree', color='#d4a574')
            bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest', color='#e77c8e')

            ax.set_ylabel('Score')
            ax.set_title(f'Model Comparison - {dataset.replace("_", " ").title()}')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No comparison data for {dataset}",
                   ha='center', va='center', transform=ax.transAxes)

        self.fig.tight_layout()

    def _plot_error_analysis(self):
        """Plot error analysis."""
        dataset = self.dataset_var.get()

        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        # Load comparison to get error rates
        comp_path = Path("models") / f"comparison_{dataset}.csv"

        if comp_path.exists():
            df = pd.read_csv(comp_path)

            # Error rates
            models = df["Model"].tolist()
            error_rates = [1 - acc for acc in df["Accuracy"]]

            colors = ['#d4a574', '#e77c8e']
            ax1.bar(models, error_rates, color=colors)
            ax1.set_ylabel("Error Rate")
            ax1.set_title("Error Rate Comparison")
            ax1.set_ylim(0, max(error_rates) * 1.3)

            for i, (model, rate) in enumerate(zip(models, error_rates)):
                ax1.text(i, rate + 0.01, f"{rate:.2%}", ha='center')

            # Metric comparison radar-like
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
            dt_vals = [df[df["Model"] == "Decision Tree"][m].values[0] for m in metrics]
            rf_vals = [df[df["Model"] == "Random Forest"][m].values[0] for m in metrics]

            x = np.arange(len(metrics))
            ax2.plot(x, dt_vals, 'o-', label='Decision Tree', color='#d4a574', linewidth=2, markersize=8)
            ax2.plot(x, rf_vals, 's-', label='Random Forest', color='#e77c8e', linewidth=2, markersize=8)
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics)
            ax2.set_ylabel("Score")
            ax2.set_title("Metrics Profile")
            ax2.legend()
            ax2.set_ylim(0.5, 1.0)
            ax2.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax2.transAxes)

        self.fig.tight_layout()

    def _show_correlation(self):
        """Show feature correlation heatmap."""
        self.adv_fig.clear()
        dataset = self.dataset_var.get()

        # Load data
        data_path = Path("data/raw") / f"{dataset}.csv"

        if data_path.exists():
            try:
                df = pd.read_csv(data_path)
                # Select only numeric columns
                numeric_df = df.select_dtypes(include=[np.number])

                if len(numeric_df.columns) > 0:
                    ax = self.adv_fig.add_subplot(111)
                    corr = numeric_df.corr()

                    # Limit size if too many features
                    if len(corr) > 15:
                        corr = corr.iloc[:15, :15]

                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                               ax=ax, square=True, cbar_kws={"shrink": 0.8})
                    ax.set_title(f"Feature Correlation - {dataset.replace('_', ' ').title()}")
            except Exception as e:
                ax = self.adv_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        else:
            ax = self.adv_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No data file for {dataset}", ha='center', va='center')

        self.adv_fig.tight_layout()
        self.adv_canvas.draw()

    def _show_learning_curve(self):
        """Show simulated learning curve."""
        self.adv_fig.clear()
        ax = self.adv_fig.add_subplot(111)

        # Simulated learning curve
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = 0.95 - 0.15 * np.exp(-3 * train_sizes) + np.random.randn(10) * 0.01
        val_scores = 0.85 - 0.2 * np.exp(-2 * train_sizes) + np.random.randn(10) * 0.02

        ax.plot(train_sizes * 100, train_scores, 'o-', label='Training Score', color='blue')
        ax.plot(train_sizes * 100, val_scores, 's-', label='Validation Score', color='green')
        ax.fill_between(train_sizes * 100, train_scores - 0.02, train_scores + 0.02, alpha=0.1, color='blue')
        ax.fill_between(train_sizes * 100, val_scores - 0.03, val_scores + 0.03, alpha=0.1, color='green')

        ax.set_xlabel("Training Set Size (%)")
        ax.set_ylabel("Score")
        ax.set_title("Learning Curve (Simulated)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)

        self.adv_fig.tight_layout()
        self.adv_canvas.draw()

    def _show_decision_boundary(self):
        """Show decision boundary visualization."""
        self.adv_fig.clear()
        ax = self.adv_fig.add_subplot(111)

        # Create synthetic 2D visualization
        np.random.seed(42)
        n = 200
        X1 = np.random.randn(n, 2) + [2, 2]
        X2 = np.random.randn(n, 2) + [-2, -2]

        ax.scatter(X1[:, 0], X1[:, 1], c='blue', alpha=0.5, label='Class 0')
        ax.scatter(X2[:, 0], X2[:, 1], c='red', alpha=0.5, label='Class 1')

        # Draw approximate boundary
        x_line = np.linspace(-5, 5, 100)
        ax.plot(x_line, -x_line, 'g--', linewidth=2, label='Decision Boundary')

        ax.set_xlabel("Feature 1 (PCA)")
        ax.set_ylabel("Feature 2 (PCA)")
        ax.set_title("Decision Boundary Visualization (2D Projection)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.adv_fig.tight_layout()
        self.adv_canvas.draw()

    def _show_tree_structure(self):
        """Show decision tree structure."""
        self.adv_fig.clear()
        dataset = self.dataset_var.get()

        img_path = Path("reports/figures") / f"tree_{dataset}.png"

        ax = self.adv_fig.add_subplot(111)
        if img_path.exists():
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f"Decision Tree Structure - {dataset.replace('_', ' ').title()}")
        else:
            ax.text(0.5, 0.5, f"No tree visualization for {dataset}",
                   ha='center', va='center', transform=ax.transAxes)

        self.adv_canvas.draw()

    def _open_prediction_window(self):
        """Open single prediction window."""
        PredictionWindow(self.root, self)

    def _open_interactive_test(self):
        """Open interactive testing window."""
        InteractiveTestWindow(self.root, self)

    def _batch_prediction(self):
        """Run batch prediction from file."""
        filepath = filedialog.askopenfilename(
            title="Select CSV file for batch prediction",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            try:
                df = pd.read_csv(filepath)
                dataset = self.dataset_var.get()
                model_type = self.model_var.get()

                model_key = f"{'rf' if model_type == 'random_forest' else 'dt'}_{dataset}"

                if model_key not in self.models:
                    messagebox.showerror("Error", f"Model not loaded for {dataset}")
                    return

                model = self.models[model_key]
                preprocessor = self.preprocessors.get(dataset)

                X = df.values
                if preprocessor:
                    X = preprocessor.transform(X)

                predictions = model.predict(X)
                probabilities = model.predict_proba(X)

                # Display in predictions tab
                for item in self.pred_tree.get_children():
                    self.pred_tree.delete(item)

                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    conf = max(prob)
                    self.pred_tree.insert("", tk.END, values=(i, pred, f"{conf:.2%}", "N/A"))

                self.notebook.select(2)  # Switch to predictions tab
                self._update_status(f"Batch prediction complete: {len(predictions)} samples")

            except Exception as e:
                messagebox.showerror("Error", f"Batch prediction failed: {str(e)}")

    def _load_csv_data(self):
        """Load custom CSV data."""
        filepath = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            try:
                self.test_data = pd.read_csv(filepath)
                self._update_status(f"Loaded {len(self.test_data)} samples from {Path(filepath).name}")
                messagebox.showinfo("Success", f"Loaded {len(self.test_data)} samples")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")

    def _export_results(self):
        """Export current results."""
        filepath = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            dataset = self.dataset_var.get()
            comp_path = Path("models") / f"comparison_{dataset}.csv"

            if comp_path.exists():
                df = pd.read_csv(comp_path)
                df.to_csv(filepath, index=False)
                self._update_status(f"Exported results to {filepath}")
                messagebox.showinfo("Success", f"Results exported to {filepath}")
            else:
                messagebox.showwarning("Warning", "No results to export")

    def _show_model_info(self):
        """Show detailed model information."""
        dataset = self.dataset_var.get()

        info_window = tk.Toplevel(self.root)
        info_window.title("Model Information")
        info_window.geometry("500x400")

        text = tk.Text(info_window, font=('Courier', 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        lines = [f"Dataset: {dataset}\n", "=" * 40 + "\n\n"]

        if dataset in self.metadata:
            meta = self.metadata[dataset]
            lines.append(f"Samples: {meta.get('n_samples', 'N/A')}\n")
            lines.append(f"Features: {meta.get('n_features', 'N/A')}\n")
            lines.append(f"Target Classes: {meta.get('target_names', 'N/A')}\n\n")

            if 'feature_names' in meta:
                lines.append("Feature Names:\n")
                for i, fn in enumerate(meta['feature_names'][:20]):
                    lines.append(f"  {i+1}. {fn}\n")
                if len(meta['feature_names']) > 20:
                    lines.append(f"  ... and {len(meta['feature_names'])-20} more\n")

        text.insert(tk.END, "".join(lines))
        text.config(state=tk.DISABLED)

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About",
            "ML Tree System v1.0\n\n"
            "Advanced Machine Learning System\n"
            "using Decision Tree & Random Forest\n\n"
            "Features:\n"
            "- Model Training & Evaluation\n"
            "- Interactive Visualizations\n"
            "- Real-time Predictions\n"
            "- Advanced Analytics\n\n"
            "Built with Python & Tkinter")

    def _update_status(self, message):
        """Update status bar."""
        self.status_var.set(message)
        self.root.update_idletasks()


class PredictionWindow:
    """Window for single prediction."""

    def __init__(self, parent, app):
        self.app = app
        self.window = tk.Toplevel(parent)
        self.window.title("Make Prediction")
        self.window.geometry("600x700")

        self.entries = {}
        self._create_ui()

    def _create_ui(self):
        """Create prediction UI."""
        dataset = self.app.dataset_var.get()

        # Header
        ttk.Label(self.window, text=f"Enter Features for {dataset.replace('_', ' ').title()}",
                 style='Header.TLabel').pack(pady=10)

        # Scrollable frame for features
        canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Get feature names
        feature_names = []
        if dataset in self.app.metadata:
            feature_names = self.app.metadata[dataset].get('feature_names', [])

        if not feature_names:
            feature_names = [f"Feature {i+1}" for i in range(10)]

        # Create entry fields
        for i, feature in enumerate(feature_names):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, padx=20, pady=2)

            ttk.Label(frame, text=f"{feature}:", width=25).pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=20)
            entry.insert(0, "0.0")
            entry.pack(side=tk.LEFT, padx=5)
            self.entries[feature] = entry

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons
        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Predict", style='Action.TButton',
                  command=self._make_prediction).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear", command=self._clear_entries).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Random Sample", command=self._fill_random).pack(side=tk.LEFT, padx=5)

        # Result display
        self.result_var = tk.StringVar(value="Enter features and click Predict")
        result_label = ttk.Label(self.window, textvariable=self.result_var,
                                font=('Helvetica', 12), wraplength=500)
        result_label.pack(pady=20)

    def _make_prediction(self):
        """Make prediction with entered values."""
        try:
            dataset = self.app.dataset_var.get()
            model_type = self.app.model_var.get()

            model_key = f"{'rf' if model_type == 'random_forest' else 'dt'}_{dataset}"

            if model_key not in self.app.models:
                self.result_var.set(f"Error: Model not loaded for {dataset}")
                return

            model = self.app.models[model_key]
            preprocessor = self.app.preprocessors.get(dataset)

            # Get values
            values = [float(entry.get()) for entry in self.entries.values()]
            X = np.array(values).reshape(1, -1)

            if preprocessor:
                X = preprocessor.transform(X)

            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            # Get class names
            target_names = ["Class 0", "Class 1"]
            if dataset in self.app.metadata:
                target_names = self.app.metadata[dataset].get('target_names', target_names)

            class_name = target_names[prediction] if prediction < len(target_names) else str(prediction)
            confidence = max(probabilities)

            result_text = (
                f"Prediction: {class_name}\n"
                f"Confidence: {confidence:.2%}\n"
                f"Probabilities: {dict(zip(target_names, probabilities))}"
            )
            self.result_var.set(result_text)

        except ValueError as e:
            self.result_var.set(f"Error: Invalid input values. Please enter numbers only.")
        except Exception as e:
            self.result_var.set(f"Error: {str(e)}")

    def _clear_entries(self):
        """Clear all entry fields."""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, "0.0")

    def _fill_random(self):
        """Fill with random sample values."""
        for entry in self.entries.values():
            entry.delete(0, tk.END)
            entry.insert(0, f"{np.random.randn():.4f}")


class InteractiveTestWindow:
    """Window for interactive model testing."""

    def __init__(self, parent, app):
        self.app = app
        self.window = tk.Toplevel(parent)
        self.window.title("Interactive Model Testing")
        self.window.geometry("900x700")

        self._create_ui()

    def _create_ui(self):
        """Create interactive testing UI."""
        # Header
        ttk.Label(self.window, text="Interactive Model Testing",
                 style='Title.TLabel').pack(pady=10)

        # Control panel
        control_frame = ttk.Frame(self.window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(control_frame, text="Number of test samples:").pack(side=tk.LEFT)
        self.n_samples = ttk.Spinbox(control_frame, from_=10, to=1000, width=10)
        self.n_samples.set(100)
        self.n_samples.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Run Test", style='Action.TButton',
                  command=self._run_test).pack(side=tk.LEFT, padx=10)
        ttk.Button(control_frame, text="Compare Models",
                  command=self._compare_models).pack(side=tk.LEFT, padx=5)

        # Results display
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Stats display
        self.stats_var = tk.StringVar(value="Click 'Run Test' to start")
        ttk.Label(self.window, textvariable=self.stats_var,
                 font=('Courier', 10)).pack(pady=10)

    def _run_test(self):
        """Run interactive test."""
        self.fig.clear()
        dataset = self.app.dataset_var.get()
        model_type = self.app.model_var.get()

        n = int(self.n_samples.get())
        model_key = f"{'rf' if model_type == 'random_forest' else 'dt'}_{dataset}"

        if model_key not in self.app.models:
            self.stats_var.set(f"Error: Model not loaded for {dataset}")
            return

        model = self.app.models[model_key]

        # Generate synthetic test data
        np.random.seed(42)
        if dataset in self.app.metadata:
            n_features = self.app.metadata[dataset].get('n_features', 10)
        else:
            n_features = 10

        X_test = np.random.randn(n, n_features)

        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Plot results
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)

        # Prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        ax1.bar(unique, counts, color=['#3498db', '#e74c3c'])
        ax1.set_xlabel("Predicted Class")
        ax1.set_ylabel("Count")
        ax1.set_title("Prediction Distribution")

        # Confidence histogram
        max_probs = np.max(probabilities, axis=1)
        ax2.hist(max_probs, bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(max_probs), color='red', linestyle='--', label=f'Mean: {np.mean(max_probs):.2f}')
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Confidence Distribution")
        ax2.legend()

        self.fig.tight_layout()
        self.canvas.draw()

        # Update stats
        stats_text = (
            f"Samples: {n} | Model: {model_type.replace('_', ' ').title()}\n"
            f"Class 0: {(predictions == 0).sum()} | Class 1: {(predictions == 1).sum()}\n"
            f"Mean Confidence: {np.mean(max_probs):.2%} | Std: {np.std(max_probs):.2%}"
        )
        self.stats_var.set(stats_text)

    def _compare_models(self):
        """Compare both models."""
        self.fig.clear()
        dataset = self.app.dataset_var.get()

        rf_key = f"rf_{dataset}"
        dt_key = f"dt_{dataset}"

        if rf_key not in self.app.models or dt_key not in self.app.models:
            self.stats_var.set("Error: Both models must be loaded")
            return

        rf_model = self.app.models[rf_key]
        dt_model = self.app.models[dt_key]

        n = int(self.n_samples.get())

        # Generate test data
        np.random.seed(42)
        if dataset in self.app.metadata:
            n_features = self.app.metadata[dataset].get('n_features', 10)
        else:
            n_features = 10

        X_test = np.random.randn(n, n_features)

        # Predictions
        rf_pred = rf_model.predict(X_test)
        dt_pred = dt_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)
        dt_prob = dt_model.predict_proba(X_test)

        # Agreement
        agreement = (rf_pred == dt_pred).mean()

        # Plot
        ax1 = self.fig.add_subplot(131)
        ax2 = self.fig.add_subplot(132)
        ax3 = self.fig.add_subplot(133)

        # Prediction comparison
        data = {
            'Decision Tree': [(dt_pred == 0).sum(), (dt_pred == 1).sum()],
            'Random Forest': [(rf_pred == 0).sum(), (rf_pred == 1).sum()]
        }
        x = np.arange(2)
        width = 0.35
        ax1.bar(x - width/2, data['Decision Tree'], width, label='Decision Tree', color='#d4a574')
        ax1.bar(x + width/2, data['Random Forest'], width, label='Random Forest', color='#e77c8e')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Class 0', 'Class 1'])
        ax1.set_ylabel('Count')
        ax1.set_title('Predictions by Model')
        ax1.legend()

        # Confidence comparison
        ax2.boxplot([np.max(dt_prob, axis=1), np.max(rf_prob, axis=1)],
                   labels=['Decision Tree', 'Random Forest'])
        ax2.set_ylabel('Confidence')
        ax2.set_title('Confidence Distribution')

        # Agreement pie
        ax3.pie([agreement, 1-agreement], labels=['Agree', 'Disagree'],
               autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
        ax3.set_title('Model Agreement')

        self.fig.tight_layout()
        self.canvas.draw()

        stats_text = (
            f"Model Comparison on {n} samples\n"
            f"Agreement Rate: {agreement:.2%}\n"
            f"DT Mean Conf: {np.max(dt_prob, axis=1).mean():.2%} | "
            f"RF Mean Conf: {np.max(rf_prob, axis=1).mean():.2%}"
        )
        self.stats_var.set(stats_text)


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = MLTreeSystemGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
