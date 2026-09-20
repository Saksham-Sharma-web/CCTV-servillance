import os
import re

files_to_combine = [
    ("Introduction & Requirements", "README.md", False),
    ("Project Explanation", "PROJECT_EXPLANATION.txt", True),
    ("Detailed Technical Architecture", "TECHNICAL_ARCHITECTURE.txt", True),
    ("IBVAP Core Pipeline", "ibvap/README.md", True),
    ("Camera Configuration Guide", "ADMIN_CAMERA_CONFIGURATION_GUIDE.md", True),
    ("Multiple Reference Testing Guide", "MULTIPLE_REFERENCE_TESTING_GUIDE.md", True),
    ("Vehicle ANPR Implementation State", "IBVAP_VEHICLE_ANPR_IMPLEMENTATION_STATE.md", True),
    ("Knowledge Base & Additional Docs", "knowledge/README.md", True),
    ("Reference Faces", "reference_faces/README.md", True)
]

def adjust_headings(content, level_increase=1):
    # This regex matches lines starting with one or more '#' followed by a space
    def replacer(match):
        return '#' * (len(match.group(1)) + level_increase) + ' '
    
    # We only apply this to lines that look like headings
    lines = content.split('\n')
    adjusted = []
    for line in lines:
        if line.startswith('#'):
            adjusted.append(re.sub(r'^(#+)\s', replacer, line))
        else:
            adjusted.append(line)
    return '\n'.join(adjusted)

master_content = []
master_content.append("# IBVAP Master Documentation\n")
master_content.append("This is the master documentation file containing all architectural details, guides, and implementation states for the IBVAP (Intelligent Border Video Analytics Platform).\n")

# Back up the original README just in case, read it from memory
with open("README.md", "r") as f:
    readme_original = f.read()
    
with open("README.md.bak", "w") as f:
    f.write(readme_original)

for section_title, filepath, adjust in files_to_combine:
    if os.path.exists(filepath):
        # We already read README.md above, use original so we don't read the partially written master if anything goes wrong
        if filepath == "README.md":
            content = readme_original
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
        if adjust and filepath.endswith('.md'):
            content = adjust_headings(content, 1)
        elif adjust and filepath.endswith('.txt'):
            # Text files might not have markdown headings, wrap in code block or just append
            pass
            
        master_content.append(f"\n\n---\n\n## {section_title}\n\n")
        
        if filepath.endswith('.txt'):
            master_content.append("```text\n" + content + "\n```")
        else:
            master_content.append(content)

with open("README.md", "w", encoding="utf-8") as f:
    f.write("".join(master_content))

print("Master README generated successfully.")
