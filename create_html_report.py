#!/usr/bin/env python3
"""
Create an HTML report from experiment results for easy remote viewing.
This combines all results, images, and data into a single HTML file.
"""

import os
import json
import base64
import glob
from pathlib import Path

def encode_image(image_path):
    """Encode image to base64 for embedding in HTML."""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

def create_html_report(results_dir):
    """Create a comprehensive HTML report from experiment results."""
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(results_dir, 'run_*'))
    
    if not run_dirs:
        print(f"No run directories found in {results_dir}")
        return
    
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fault Injection Results - {run_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .experiment {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric {{
            background: #f0f0f0;
            padding: 10px;
            border-radius: 4px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: bold;
            color: #2196F3;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 10px 0;
        }}
        pre {{
            background: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .tab-container {{
            margin: 20px 0;
        }}
        .tab-buttons {{
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .tab-button {{
            padding: 10px 20px;
            background: #e0e0e0;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }}
        .tab-button.active {{
            background: #2196F3;
            color: white;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
    <script>
        function showTab(tabId, buttonId) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab-button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            document.getElementById(buttonId).classList.add('active');
        }}
    </script>
</head>
<body>
    <h1>🔬 Fault Injection Experiment Results</h1>
    <p><strong>Run:</strong> {run_name}</p>
"""
        
        # Add final report if exists
        report_path = os.path.join(run_dir, 'final_report.md')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_content = f.read()
            html_content += f"""
    <div class="experiment">
        <h2>📊 Final Report</h2>
        <pre>{report_content}</pre>
    </div>
"""
        
        # Add summary visualization
        summary_img = os.path.join(run_dir, 'summary_visualizations.png')
        if os.path.exists(summary_img):
            img_data = encode_image(summary_img)
            if img_data:
                html_content += f"""
    <div class="experiment">
        <h2>📈 Summary Visualizations</h2>
        <img src="data:image/png;base64,{img_data}" alt="Summary Visualizations">
    </div>
"""
        
        # Add experiment tabs
        experiment_dirs = sorted(glob.glob(os.path.join(run_dir, 'experiment_*')))
        if experiment_dirs:
            html_content += """
    <div class="experiment">
        <h2>🧪 Individual Experiments</h2>
        <div class="tab-container">
            <div class="tab-buttons">
"""
            for i, exp_dir in enumerate(experiment_dirs):
                exp_name = os.path.basename(exp_dir)
                active = "active" if i == 0 else ""
                html_content += f"""
                <button id="btn-{exp_name}" class="tab-button {active}" 
                        onclick="showTab('{exp_name}', 'btn-{exp_name}')">
                    {exp_name}
                </button>
"""
            html_content += """
            </div>
"""
            
            # Add tab contents
            for i, exp_dir in enumerate(experiment_dirs):
                exp_name = os.path.basename(exp_dir)
                active = "active" if i == 0 else ""
                html_content += f"""
            <div id="{exp_name}" class="tab-content {active}">
                <h3>{exp_name}</h3>
"""
                
                # Load results.json
                results_path = os.path.join(exp_dir, 'results.json')
                if os.path.exists(results_path):
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    
                    # Display key metrics
                    html_content += """
                <div class="metrics">
"""
                    metrics_to_show = [
                        ('fault_type', 'Fault Type'),
                        ('injection_epoch', 'Injection Epoch'),
                        ('injection_step', 'Injection Step'),
                        ('baseline_final_acc', 'Baseline Final Acc'),
                        ('baseline_recovery_rate', 'Baseline Recovery'),
                    ]
                    
                    for key, label in metrics_to_show:
                        if key in results:
                            value = results[key]
                            if isinstance(value, float):
                                value = f"{value:.4f}"
                            html_content += f"""
                    <div class="metric">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
"""
                    html_content += """
                </div>
"""
                
                # Add images
                for img_file in ['recovery_comparison.png', 'degradation_rates.png']:
                    img_path = os.path.join(exp_dir, img_file)
                    if os.path.exists(img_path):
                        img_data = encode_image(img_path)
                        if img_data:
                            html_content += f"""
                <h4>{img_file.replace('_', ' ').title()}</h4>
                <img src="data:image/png;base64,{img_data}" alt="{img_file}">
"""
                
                html_content += """
            </div>
"""
            
            html_content += """
        </div>
    </div>
"""
        
        # Close HTML
        html_content += """
</body>
</html>
"""
        
        # Save HTML file
        output_path = os.path.join(run_dir, 'report.html')
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Created HTML report: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Create index file
    index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Fault Injection Results Index</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .run-link {
            display: block;
            padding: 15px;
            margin: 10px 0;
            background: #f0f0f0;
            border-radius: 4px;
            text-decoration: none;
            color: #333;
        }
        .run-link:hover {
            background: #e0e0e0;
        }
    </style>
</head>
<body>
    <h1>All Experiment Runs</h1>
"""
    
    for run_dir in sorted(run_dirs):
        run_name = os.path.basename(run_dir)
        index_html += f"""
    <a href="{run_name}/report.html" class="run-link">
        📁 {run_name}
    </a>
"""
    
    index_html += """
</body>
</html>
"""
    
    index_path = os.path.join(results_dir, 'index.html')
    with open(index_path, 'w') as f:
        f.write(index_html)
    
    print(f"\n✅ Created index: {index_path}")
    print("\nTo view these reports:")
    print("1. Download the HTML files to your local machine")
    print("2. Or start a simple web server (see view_results.sh)")

if __name__ == "__main__":
    import sys
    
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "./extracted_results/optimizer"
    
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} does not exist")
        print("Usage: python create_html_report.py [results_directory]")
        sys.exit(1)
    
    create_html_report(results_dir)