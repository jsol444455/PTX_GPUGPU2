#!/usr/bin/env python3
"""Quick script to add debug output to make_ctaid_map"""

import sys

def add_debug_to_file():
    file_path = "ptx_files/ptx_tracing.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find make_ctaid_map function
    for i, line in enumerate(lines):
        if 'def make_ctaid_map(' in line:
            print(f"Found make_ctaid_map at line {i+1}")
            
            # Find the line after docstring and global declarations
            j = i + 1
            while j < len(lines):
                if 'global ctaid_map' in lines[j]:
                    # Insert debug after this line
                    insert_line = j + 1
                    
                    debug_code = '''
    # === DEBUG OUTPUT ===
    print(f"\\n{'='*80}")
    print(f"[make_ctaid_map] 🚀 CALLED")
    print(f"{'='*80}")
    print(f"  formular type: {type(formular)}")
    print(f"  kernel_info: {kernel_info is not None}")
    print(f"  bb_graph: {bb_graph is not None}")
    print(f"  kernel_name: {kernel_name}")
    print(f"  param_dict: {param_dict is not None}")
    print(f"{'='*80}\\n")
'''
                    
                    lines.insert(insert_line, debug_code)
                    
                    # Write back
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
                    
                    print(f"✅ Added debug output at line {insert_line}")
                    return True
                j += 1
    
    print("❌ Could not find insertion point")
    return False

if __name__ == "__main__":
    print("Adding debug output to ptx_tracing.py...")
    if add_debug_to_file():
        print("\n✅ Success! Now run:")
        print("   python3 locality_map_coalescing.py -f GEMM_2_8_32_8")
    else:
        print("\n❌ Failed. Manually add the debug code.")
