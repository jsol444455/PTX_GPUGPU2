#!/usr/bin/env python3
"""
LocalityGuru Loop Evaluation Diagnostic Script

This script checks if your LocalityGuru installation has all the necessary
components for loop evaluation and identifies what might be missing or broken.

Usage: python3 diagnose.py <path_to_ptx_file_base_name>
Example: python3 diagnose.py vector_256_1
"""

import sys
import os
import json
from pathlib import Path

def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def print_check(passed, message):
    icon = "✅" if passed else "❌"
    print(f"{icon} {message}")
    return passed

def check_files_exist(base_name):
    """Check if all required JSON files exist"""
    print_header("STEP 1: Checking Required Files")
    
    base_path = Path(f"syntax_tree/{base_name}")
    
    checks = []
    
    # Check if directory exists
    if not base_path.exists():
        print_check(False, f"Directory not found: {base_path}")
        print(f"\n💡 Hint: Run 'python3 locality_guru.py -f {base_name}.ptx' first!")
        return False
    
    print_check(True, f"Directory exists: {base_path}")
    
    # Check required files
    required_files = [
        (f"{base_name}_st.json", "Syntax tree file"),
        (f"{base_name}_kernel_info.json", "Kernel info file (contains predicates)"),
        (f"{base_name}_bb.json", "Basic block graph file"),
        (f"{base_name}_param.json", "Parameters file"),
        (f"{base_name}_formular.json", "Formula file"),
    ]
    
    all_exist = True
    for filename, description in required_files:
        filepath = base_path / filename
        exists = filepath.exists()
        checks.append(print_check(exists, f"{description}: {filename}"))
        all_exist = all_exist and exists
    
    return all_exist

def check_loop_detection(base_name):
    """Check if loops were detected in the BB graph"""
    print_header("STEP 2: Checking Loop Detection")
    
    bb_file = Path(f"syntax_tree/{base_name}/{base_name}_bb.json")
    
    if not bb_file.exists():
        print_check(False, "BB graph file not found")
        return False
    
    with open(bb_file, 'r') as f:
        bb_graph = json.load(f)
    
    print(f"Loaded BB graph with {len(bb_graph)} kernels")
    
    total_loops = 0
    for kernel_name, bbs in bb_graph.items():
        print(f"\n  Kernel: {kernel_name}")
        
        loop_bbs = []
        for bb_n, bb_info in bbs.items():
            if bb_info.get("is_loop_header", False):
                loop_bbs.append(bb_n)
                print(f"    Loop BB {bb_n}:")
                print(f"      - Predicate: {bb_info.get('loop_predicate', 'MISSING')}")
                print(f"      - Variables: {bb_info.get('loop_variables', [])}")
                print(f"      - Has ld.global: {bb_info.get('has_ld_global', False)}")
        
        if loop_bbs:
            print_check(True, f"Found {len(loop_bbs)} loops in {kernel_name}")
            total_loops += len(loop_bbs)
        else:
            print_check(False, f"No loops detected in {kernel_name}")
    
    return total_loops > 0

def check_predicates(base_name):
    """Check if predicates were parsed and saved"""
    print_header("STEP 3: Checking Predicate Parsing")
    
    ki_file = Path(f"syntax_tree/{base_name}/{base_name}_kernel_info.json")
    
    if not ki_file.exists():
        print_check(False, "Kernel info file not found")
        return False
    
    with open(ki_file, 'r') as f:
        kernel_info = json.load(f)
    
    total_preds = 0
    for kernel_name, info in kernel_info.items():
        print(f"\n  Kernel: {kernel_name}")
        
        if "predicates" in info:
            preds = info["predicates"]
            print_check(True, f"Found {len(preds)} predicates")
            total_preds += len(preds)
            
            # Show first few
            for pred_name in list(preds.keys())[:3]:
                tree = preds[pred_name]
                if tree and len(tree) > 0:
                    opcode = tree[0].get("opcode", "unknown")
                    print(f"      - {pred_name}: {len(tree)} nodes, opcode: {opcode}")
        else:
            print_check(False, f"No predicates found in {kernel_name}")
    
    return total_preds > 0

def check_ptx_tracing_functions():
    """Check if ptx_tracing.py has the required functions"""
    print_header("STEP 4: Checking ptx_tracing.py Functions")
    
    try:
        # Try to import and check for functions
        import ptx_files.ptx_tracing as ptx_tracing
        
        required_functions = [
            ("GetTBAddressMap", "Main algorithm 2 implementation"),
            ("get_loop_addresses", "Loop address evaluation"),
            ("eval_predicate_tree", "Predicate evaluation"),
            ("evaluate_formula", "Formula evaluation"),
            ("make_ctaid_map", "Locality map generation"),
        ]
        
        all_exist = True
        for func_name, description in required_functions:
            exists = hasattr(ptx_tracing, func_name)
            print_check(exists, f"{func_name}() - {description}")
            all_exist = all_exist and exists
        
        return all_exist
        
    except ImportError as e:
        print_check(False, f"Could not import ptx_tracing: {e}")
        return False

def check_loop_and_predicate_match(base_name):
    """Check if loop predicates match available predicates"""
    print_header("STEP 5: Checking Loop-Predicate Matching")
    
    bb_file = Path(f"syntax_tree/{base_name}/{base_name}_bb.json")
    ki_file = Path(f"syntax_tree/{base_name}/{base_name}_kernel_info.json")
    
    if not bb_file.exists() or not ki_file.exists():
        print_check(False, "Required files not found")
        return False
    
    with open(bb_file, 'r') as f:
        bb_graph = json.load(f)
    
    with open(ki_file, 'r') as f:
        kernel_info = json.load(f)
    
    all_match = True
    
    for kernel_name, bbs in bb_graph.items():
        print(f"\n  Kernel: {kernel_name}")
        
        # Get available predicates
        available_preds = set()
        if kernel_name in kernel_info and "predicates" in kernel_info[kernel_name]:
            available_preds = set(kernel_info[kernel_name]["predicates"].keys())
        
        print(f"    Available predicates: {len(available_preds)}")
        if available_preds:
            print(f"      {list(available_preds)[:5]}")
        
        # Check each loop
        for bb_n, bb_info in bbs.items():
            if bb_info.get("is_loop_header", False):
                loop_pred = bb_info.get("loop_predicate")
                
                if not loop_pred:
                    print_check(False, f"BB {bb_n}: Loop has no predicate")
                    all_match = False
                    continue
                
                # Try to find matching predicate (with various formats)
                found = False
                for pred in available_preds:
                    # Check if predicate matches (with or without %)
                    pred_base = pred.split('_')[0].replace('%', '')
                    loop_pred_base = loop_pred.replace('%', '')
                    
                    if pred_base == loop_pred_base or pred.startswith(loop_pred):
                        found = True
                        print_check(True, f"BB {bb_n}: Loop predicate '{loop_pred}' matches '{pred}'")
                        break
                
                if not found:
                    print_check(False, f"BB {bb_n}: Loop predicate '{loop_pred}' NOT FOUND in predicates")
                    print(f"        💡 Hint: Check predicate name format in bb_graph vs kernel_info")
                    all_match = False
    
    return all_match

def provide_recommendations(results):
    """Provide specific recommendations based on test results"""
    print_header("RECOMMENDATIONS")
    
    if all(results.values()):
        print("🎉 All checks passed! Your setup looks good.")
        print("\nIf you're still having issues:")
        print("  1. Apply the debug patches from DEBUG_PATCHES.md")
        print("  2. Run: python3 locality_guru.py -f <file>.ptx")
        print("  3. Run: python3 locality_map_coalescing.py -f <file>")
        print("  4. Check debug output for specific errors")
        return
    
    if not results["files"]:
        print("❌ Missing required files")
        print("\n📋 Action: Run locality_guru.py first:")
        print("   python3 locality_guru.py -f <your_file>.ptx")
        return
    
    if not results["loops"]:
        print("❌ No loops detected in PTX")
        print("\n📋 Possible causes:")
        print("   1. The PTX file doesn't contain loops")
        print("   2. Loop detection code isn't recognizing the branch format")
        print("\n💡 Actions:")
        print("   1. Open the PTX file and look for '@%p bra' instructions")
        print("   2. Check if branch targets point backward (loop)")
        print("   3. Apply PATCH 1 from DEBUG_PATCHES.md for debug output")
        print("   4. Run again and check what branches are found")
    
    if not results["predicates"]:
        print("❌ No predicates parsed")
        print("\n📋 Possible causes:")
        print("   1. No 'setp' instructions in PTX")
        print("   2. Predicate parsing code not working")
        print("\n💡 Actions:")
        print("   1. Open PTX file and search for 'setp'")
        print("   2. Apply PATCH 4 from DEBUG_PATCHES.md")
        print("   3. Run locality_guru.py again")
    
    if not results["functions"]:
        print("❌ Missing required functions in ptx_tracing.py")
        print("\n📋 This is unusual - the functions should exist")
        print("\n💡 Actions:")
        print("   1. Check if you're using the correct version")
        print("   2. Verify ptx_files/ptx_tracing.py is not corrupted")
        print("   3. Re-download from the repository")
    
    if results["loops"] and results["predicates"] and not results.get("matching", True):
        print("❌ Loop predicates don't match parsed predicates")
        print("\n📋 This is a name formatting issue")
        print("\n💡 Actions:")
        print("   1. Check if predicate names use '%' prefix")
        print("   2. Verify naming consistency between bb_graph and kernel_info")
        print("   3. You may need to normalize predicate names")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diagnose.py <base_name>")
        print("Example: python3 diagnose.py vector_256_1")
        print("\nThis will check syntax_tree/vector_256_1/*.json files")
        sys.exit(1)
    
    base_name = sys.argv[1]
    
    print_header(f"LocalityGuru Loop Evaluation Diagnostic")
    print(f"Checking setup for: {base_name}")
    
    results = {}
    
    # Run all checks
    results["files"] = check_files_exist(base_name)
    
    if results["files"]:
        results["loops"] = check_loop_detection(base_name)
        results["predicates"] = check_predicates(base_name)
        results["matching"] = check_loop_and_predicate_match(base_name)
    else:
        results["loops"] = False
        results["predicates"] = False
        results["matching"] = False
    
    results["functions"] = check_ptx_tracing_functions()
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Checks passed: {passed}/{total}")
    
    for check_name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {check_name}")
    
    # Provide recommendations
    provide_recommendations(results)
    
    print("\n" + "="*80)
    print("For detailed debugging, see:")
    print("  - DEBUGGING_GUIDE.md")
    print("  - DEBUG_PATCHES.md")
    print("="*80)

if __name__ == "__main__":
    main()
