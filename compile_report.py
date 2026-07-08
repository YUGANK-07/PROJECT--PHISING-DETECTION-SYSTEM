import os
import glob

def compile_markdown():
    output_file = "PhishGuard_Final_Major_Report.md"
    drafts_dir = "report_drafts"
    
    # Get all markdown files in order
    files = sorted(glob.glob(os.path.join(drafts_dir, "*.md")))
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in files:
            with open(fname, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n\n---\n\n<div style='page-break-after: always;'></div>\n\n")
                
    print(f"Successfully compiled {len(files)} chapters into {output_file}")

if __name__ == "__main__":
    compile_markdown()
