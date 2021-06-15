#!/usr/bin/env python3
"""
Script to generate a PDF report from a Markdown file.
This requires the markdown, reportlab, and pdfkit libraries.
"""

import os
import sys
import argparse
import datetime
import markdown
import pdfkit
from pathlib import Path


def convert_markdown_to_html(md_file_path):
    """Convert markdown file to HTML."""
    with open(md_file_path, 'r') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content, 
        extensions=['tables', 'fenced_code', 'codehilite']
    )
    
    # Add basic styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Credit Card Fraud Detection - Project Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #2c3e50; text-align: center; }}
            h2 {{ color: #3498db; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            h3 {{ color: #2980b9; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            code {{ background-color: #f8f8f8; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
            pre {{ background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            .date {{ text-align: right; color: #7f8c8d; margin-bottom: 30px; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; }}
            blockquote {{ border-left: 4px solid #ccc; padding-left: 15px; color: #555; }}
            a {{ color: #3498db; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="date">Generated on {datetime.datetime.now().strftime('%B %d, %Y')}</div>
        {html_content}
    </body>
    </html>
    """
    
    return styled_html


def save_html_to_file(html_content, output_html_path):
    """Save HTML content to file."""
    with open(output_html_path, 'w') as f:
        f.write(html_content)
    print(f"HTML file saved to {output_html_path}")


def convert_html_to_pdf(html_path, pdf_path):
    """Convert HTML file to PDF using pdfkit."""
    try:
        # Configure pdfkit options
        options = {
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'encoding': 'UTF-8',
            'no-outline': None,
            'enable-local-file-access': None,
        }
        
        # Convert HTML to PDF
        pdfkit.from_file(html_path, pdf_path, options=options)
        print(f"PDF file saved to {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("If wkhtmltopdf is not installed, please install it from: https://wkhtmltopdf.org/downloads.html")
        sys.exit(1)


def main():
    """Main function to convert markdown to PDF."""
    parser = argparse.ArgumentParser(description='Generate PDF report from Markdown file')
    parser.add_argument('--input', '-i', default='docs/project_report.md', 
                        help='Path to input markdown file')
    parser.add_argument('--output', '-o', default='docs/project_report.pdf',
                        help='Path to output PDF file')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate temporary HTML file path
    html_path = os.path.splitext(args.output)[0] + '.html'
    
    # Convert markdown to HTML
    print(f"Converting {args.input} to HTML...")
    html_content = convert_markdown_to_html(args.input)
    
    # Save HTML to file
    save_html_to_file(html_content, html_path)
    
    # Convert HTML to PDF
    print(f"Converting HTML to PDF...")
    convert_html_to_pdf(html_path, args.output)
    
    # Clean up temporary HTML file
    if os.path.exists(html_path):
        os.remove(html_path)
        print(f"Temporary HTML file removed")


if __name__ == "__main__":
    main()