"""
Script to compare performance of all trained models from the latest run.
Displays a table with metrics and percentage differences relative to the best model.
"""
import json
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

def compare_models():
    console = Console()
    
    # Load latest run data
    artifact_path = Path('artifacts/latest_run.json')
    if not artifact_path.exists():
        console.print("[bold red]❌ Error: artifacts/latest_run.json not found.[/bold red]")
        console.print("   Run the pipeline first: [cyan]python run_pipeline.py --quick[/cyan]")
        return

    with open(artifact_path, 'r') as f:
        data = json.load(f)
    
    comparison = data.get('model_comparison', [])
    
    if not comparison:
        console.print("[bold yellow]⚠️ No model comparison data found in latest run.[/bold yellow]")
        console.print("   You may need to re-run the pipeline with the updated code.")
        return

    # Create table
    table = Table(title="🏆 Model Performance Comparison (Validation Set)", box=box.ROUNDED)
    
    table.add_column("Rank", style="cyan", justify="center")
    table.add_column("Model", style="white")
    table.add_column("Macro F1", style="green", justify="right")
    table.add_column("Accuracy", style="blue", justify="right")
    table.add_column("Diff from Best", style="red", justify="right")
    table.add_column("Type", style="dim")

    for i, model in enumerate(comparison, 1):
        name = model['model_name']
        f1 = model['macro_f1']
        acc = model['accuracy']
        delta = model['delta_from_best']
        delta_pct = model['delta_pct']
        
        # Format delta
        if i == 1:
            diff_str = "[bold green]BEST[/bold green]"
        else:
            diff_str = f"{delta:.4f} ({delta_pct:.1f}%)"
            
        # Determine type
        m_type = "Baseline" if model.get('is_baseline') else ("Ensemble" if model.get('is_ensemble') else "Model")
        
        table.add_row(
            str(i),
            name,
            f"{f1:.4f}",
            f"{acc:.4f}",
            diff_str,
            m_type
        )

    console.print(table)
    
    # Show per-class breakdown for top 3
    console.print("\n[bold]🔍 Per-Class F1 Score (Top 3 Models)[/bold]")
    
    class_table = Table(box=box.SIMPLE)
    class_table.add_column("Model", style="white")
    class_table.add_column("Dropout", style="red")
    class_table.add_column("Enrolled", style="yellow")
    class_table.add_column("Graduate", style="green")
    
    for model in comparison[:3]:
        per_class = model['per_class_f1']
        class_table.add_row(
            model['model_name'],
            f"{per_class[0]:.3f}",
            f"{per_class[1]:.3f}",
            f"{per_class[2]:.3f}"
        )
        
    console.print(class_table)

if __name__ == "__main__":
    try:
        compare_models()
    except ImportError:
        print("Rich library not installed. Please install it: pip install rich")
