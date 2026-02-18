"""
Advanced CLI Interface for ML Tree System
Interactive command-line interface for model testing and analysis.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.utils.helpers import load_artifact


class MLTreeCLI:
    """Interactive CLI for ML Tree System."""

    def __init__(self):
        self.models = {}
        self.preprocessors = {}
        self.metadata = {}
        self.datasets = ["heart_disease", "adult_income", "wine_quality", "diabetes", "breast_cancer"]
        self.current_dataset = "heart_disease"
        self._load_models()

    def _load_models(self):
        """Load all models."""
        print("\n Loading models...")
        models_dir = Path("models")

        for dataset in self.datasets:
            try:
                rf_path = models_dir / f"rf_{dataset}.joblib"
                if rf_path.exists():
                    self.models[f"rf_{dataset}"] = load_artifact(str(rf_path))

                dt_path = models_dir / f"dt_{dataset}.joblib"
                if dt_path.exists():
                    self.models[f"dt_{dataset}"] = load_artifact(str(dt_path))

                prep_path = models_dir / f"preprocessor_{dataset}.joblib"
                if prep_path.exists():
                    self.preprocessors[dataset] = load_artifact(str(prep_path))

                meta_path = models_dir / f"metadata_{dataset}.joblib"
                if meta_path.exists():
                    self.metadata[dataset] = load_artifact(str(meta_path))
            except Exception as e:
                pass

        # Load breast_cancer original models
        try:
            rf_bc = models_dir / "random_forest.joblib"
            if rf_bc.exists():
                self.models["rf_breast_cancer"] = load_artifact(str(rf_bc))
            dt_bc = models_dir / "decision_tree.joblib"
            if dt_bc.exists():
                self.models["dt_breast_cancer"] = load_artifact(str(dt_bc))
            prep_bc = models_dir / "preprocessor.joblib"
            if prep_bc.exists():
                self.preprocessors["breast_cancer"] = load_artifact(str(prep_bc))
            meta_bc = models_dir / "metadata.joblib"
            if meta_bc.exists():
                self.metadata["breast_cancer"] = load_artifact(str(meta_bc))
        except:
            pass

        print(f" Loaded {len(self.models)} models\n")

    def _clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def _print_header(self):
        """Print application header."""
        print("\n" + "=" * 60)
        print("     ML TREE SYSTEM - Interactive Interface")
        print("     Decision Tree & Random Forest Classifier")
        print("=" * 60)

    def _print_menu(self):
        """Print main menu."""
        print(f"\n Current Dataset: {self.current_dataset.replace('_', ' ').title()}")
        print("-" * 40)
        print(" 1. Change Dataset")
        print(" 2. View Model Comparison")
        print(" 3. View Feature Importance")
        print(" 4. Make Single Prediction")
        print(" 5. Make Batch Prediction")
        print(" 6. Run Model Test")
        print(" 7. View Dataset Info")
        print(" 8. Compare All Datasets")
        print(" 9. Interactive Prediction Mode")
        print(" 0. Exit")
        print("-" * 40)

    def run(self):
        """Run the CLI interface."""
        self._clear_screen()
        self._print_header()

        while True:
            self._print_menu()
            choice = input("\n Enter choice: ").strip()

            if choice == "1":
                self._change_dataset()
            elif choice == "2":
                self._view_comparison()
            elif choice == "3":
                self._view_importance()
            elif choice == "4":
                self._single_prediction()
            elif choice == "5":
                self._batch_prediction()
            elif choice == "6":
                self._run_test()
            elif choice == "7":
                self._view_dataset_info()
            elif choice == "8":
                self._compare_all_datasets()
            elif choice == "9":
                self._interactive_prediction()
            elif choice == "0":
                print("\n Goodbye!\n")
                break
            else:
                print("\n Invalid choice. Try again.")

    def _change_dataset(self):
        """Change current dataset."""
        print("\n Available Datasets:")
        for i, ds in enumerate(self.datasets, 1):
            status = "Loaded" if f"rf_{ds}" in self.models else "Not loaded"
            print(f"  {i}. {ds.replace('_', ' ').title()} [{status}]")

        try:
            choice = int(input("\n Select dataset (1-5): ")) - 1
            if 0 <= choice < len(self.datasets):
                self.current_dataset = self.datasets[choice]
                print(f"\n Switched to: {self.current_dataset}")
        except ValueError:
            print("\n Invalid selection.")

    def _view_comparison(self):
        """View model comparison."""
        comp_path = Path("models") / f"comparison_{self.current_dataset}.csv"

        if comp_path.exists():
            df = pd.read_csv(comp_path)
            print(f"\n Model Comparison - {self.current_dataset.replace('_', ' ').title()}")
            print("-" * 60)
            print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))
        else:
            print(f"\n No comparison data for {self.current_dataset}")

        input("\n Press Enter to continue...")

    def _view_importance(self):
        """View feature importance."""
        imp_path = Path("models") / f"importance_{self.current_dataset}.csv"

        if imp_path.exists():
            df = pd.read_csv(imp_path).head(15)
            print(f"\n Top 15 Feature Importance - {self.current_dataset.replace('_', ' ').title()}")
            print("-" * 60)

            for _, row in df.iterrows():
                bar_len = int(row['importance'] * 40)
                bar = "" * bar_len
                print(f" {row['feature'][:25]:<25} {bar} {row['importance']:.4f}")
        else:
            print(f"\n No importance data for {self.current_dataset}")

        input("\n Press Enter to continue...")

    def _single_prediction(self):
        """Make single prediction."""
        model_key = f"rf_{self.current_dataset}"

        if model_key not in self.models:
            print(f"\n Model not loaded for {self.current_dataset}")
            return

        model = self.models[model_key]
        preprocessor = self.preprocessors.get(self.current_dataset)

        # Get feature names
        feature_names = []
        if self.current_dataset in self.metadata:
            feature_names = self.metadata[self.current_dataset].get('feature_names', [])

        if not feature_names:
            n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 10
            feature_names = [f"Feature {i+1}" for i in range(n_features)]

        print(f"\n Enter values for {len(feature_names)} features:")
        print(" (Enter 'r' for random values, 'q' to cancel)")

        values = []
        for i, fn in enumerate(feature_names):
            while True:
                inp = input(f"  {fn}: ").strip()
                if inp.lower() == 'q':
                    return
                if inp.lower() == 'r':
                    values = np.random.randn(len(feature_names)).tolist()
                    break
                try:
                    values.append(float(inp))
                    break
                except ValueError:
                    print("   Invalid. Enter a number.")

            if len(values) == len(feature_names):
                break

        X = np.array(values).reshape(1, -1)
        if preprocessor:
            X = preprocessor.transform(X)

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        target_names = ["Class 0", "Class 1"]
        if self.current_dataset in self.metadata:
            target_names = self.metadata[self.current_dataset].get('target_names', target_names)

        print("\n" + "=" * 40)
        print(" PREDICTION RESULT")
        print("=" * 40)
        print(f" Predicted Class: {target_names[prediction] if prediction < len(target_names) else prediction}")
        print(f" Confidence: {max(probabilities):.2%}")
        print(f" Probabilities:")
        for name, prob in zip(target_names, probabilities):
            print(f"   {name}: {prob:.4f}")
        print("=" * 40)

        input("\n Press Enter to continue...")

    def _batch_prediction(self):
        """Batch prediction from file."""
        filepath = input("\n Enter CSV file path: ").strip()

        if not Path(filepath).exists():
            print(" File not found.")
            return

        model_key = f"rf_{self.current_dataset}"
        if model_key not in self.models:
            print(f" Model not loaded for {self.current_dataset}")
            return

        try:
            df = pd.read_csv(filepath)
            model = self.models[model_key]
            preprocessor = self.preprocessors.get(self.current_dataset)

            X = df.values
            if preprocessor:
                X = preprocessor.transform(X)

            predictions = model.predict(X)
            probabilities = model.predict_proba(X)

            print(f"\n Batch Prediction Results ({len(predictions)} samples)")
            print("-" * 50)

            results = pd.DataFrame({
                'Sample': range(1, len(predictions) + 1),
                'Prediction': predictions,
                'Confidence': [max(p) for p in probabilities]
            })

            print(tabulate(results.head(20), headers='keys', tablefmt='grid', showindex=False))

            if len(predictions) > 20:
                print(f"\n ... and {len(predictions) - 20} more samples")

            # Save option
            save = input("\n Save results? (y/n): ").strip().lower()
            if save == 'y':
                output_path = filepath.replace('.csv', '_predictions.csv')
                results.to_csv(output_path, index=False)
                print(f" Saved to: {output_path}")

        except Exception as e:
            print(f" Error: {e}")

        input("\n Press Enter to continue...")

    def _run_test(self):
        """Run model test."""
        model_key = f"rf_{self.current_dataset}"
        if model_key not in self.models:
            print(f"\n Model not loaded for {self.current_dataset}")
            return

        model = self.models[model_key]

        try:
            n_samples = int(input("\n Number of test samples (10-1000): "))
            n_samples = max(10, min(1000, n_samples))
        except ValueError:
            n_samples = 100

        n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 10

        print(f"\n Running test with {n_samples} random samples...")

        np.random.seed(42)
        X_test = np.random.randn(n_samples, n_features)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        print("\n" + "=" * 50)
        print(" TEST RESULTS")
        print("=" * 50)
        print(f" Samples tested: {n_samples}")
        print(f" Class 0 predictions: {(predictions == 0).sum()} ({(predictions == 0).mean():.1%})")
        print(f" Class 1 predictions: {(predictions == 1).sum()} ({(predictions == 1).mean():.1%})")
        print(f" Mean confidence: {np.max(probabilities, axis=1).mean():.2%}")
        print(f" Min confidence: {np.max(probabilities, axis=1).min():.2%}")
        print(f" Max confidence: {np.max(probabilities, axis=1).max():.2%}")
        print("=" * 50)

        input("\n Press Enter to continue...")

    def _view_dataset_info(self):
        """View dataset information."""
        print(f"\n Dataset: {self.current_dataset.replace('_', ' ').title()}")
        print("=" * 50)

        if self.current_dataset in self.metadata:
            meta = self.metadata[self.current_dataset]
            print(f" Samples: {meta.get('n_samples', 'N/A')}")
            print(f" Features: {meta.get('n_features', 'N/A')}")
            print(f" Classes: {meta.get('target_names', 'N/A')}")

            if 'feature_names' in meta:
                print(f"\n Feature Names:")
                for i, fn in enumerate(meta['feature_names'], 1):
                    print(f"   {i:2}. {fn}")
        else:
            print(" No metadata available")

        print("\n Models loaded:")
        print(f"   Random Forest: {'Yes' if f'rf_{self.current_dataset}' in self.models else 'No'}")
        print(f"   Decision Tree: {'Yes' if f'dt_{self.current_dataset}' in self.models else 'No'}")

        input("\n Press Enter to continue...")

    def _compare_all_datasets(self):
        """Compare all datasets."""
        print("\n ALL DATASETS COMPARISON")
        print("=" * 80)

        results = []
        for ds in self.datasets:
            comp_path = Path("models") / f"comparison_{ds}.csv"
            if comp_path.exists():
                df = pd.read_csv(comp_path)
                rf_row = df[df["Model"] == "Random Forest"].iloc[0] if len(df[df["Model"] == "Random Forest"]) > 0 else None

                if rf_row is not None:
                    results.append({
                        'Dataset': ds.replace('_', ' ').title(),
                        'Accuracy': rf_row['Accuracy'],
                        'F1-Score': rf_row['F1-Score'],
                        'ROC-AUC': rf_row['ROC-AUC']
                    })

        if results:
            results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
            print(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))
        else:
            print(" No comparison data available")

        input("\n Press Enter to continue...")

    def _interactive_prediction(self):
        """Interactive prediction mode."""
        print("\n INTERACTIVE PREDICTION MODE")
        print(" Type 'exit' to return to main menu")
        print("-" * 40)

        model_key = f"rf_{self.current_dataset}"
        if model_key not in self.models:
            print(f" Model not loaded for {self.current_dataset}")
            return

        model = self.models[model_key]
        preprocessor = self.preprocessors.get(self.current_dataset)
        n_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else 10

        target_names = ["Class 0", "Class 1"]
        if self.current_dataset in self.metadata:
            target_names = self.metadata[self.current_dataset].get('target_names', target_names)

        while True:
            print(f"\n Enter {n_features} comma-separated values (or 'r' for random, 'exit' to quit):")
            inp = input(" > ").strip()

            if inp.lower() == 'exit':
                break

            if inp.lower() == 'r':
                values = np.random.randn(n_features)
            else:
                try:
                    values = np.array([float(x) for x in inp.split(',')])
                    if len(values) != n_features:
                        print(f" Error: Expected {n_features} values, got {len(values)}")
                        continue
                except ValueError:
                    print(" Error: Invalid input. Enter numbers separated by commas.")
                    continue

            X = values.reshape(1, -1)
            if preprocessor:
                X = preprocessor.transform(X)

            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            class_name = target_names[prediction] if prediction < len(target_names) else str(prediction)
            print(f"\n >> Prediction: {class_name} | Confidence: {max(probabilities):.2%}")


def main():
    """Run CLI interface."""
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate...")
        os.system("pip install tabulate -q")
        from tabulate import tabulate

    cli = MLTreeCLI()
    cli.run()


if __name__ == "__main__":
    main()
