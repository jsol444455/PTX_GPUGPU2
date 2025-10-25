from email import parser
import os, sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns
import pandas as pd
from tqdm import tqdm

import argparse

param_dict = dict()

'''
tidx_list = list() #list(range(0,512))
tidy_list = list()
ctaidy_list = list()
ctaidx_list = list()
'''
global ctaid_map
ctaid_map = list()

def ADD(a,b):
    return a+ b
def SUB(a,b):
    return a-b
def MADLO(a,b):
    return (( a )&0x0000000011111111) + b #(( a )&0x0000000011111111) + b
def MUL(a,b):
    return a * b
def SHL(a,b):
    return int(a) << int(b)
def OR(a,b):
    return a | b
def AND(a,b):
    return a & b
def DIV(a,b):
    return a / b
def SELP(a,b,c):
    if c==1:
        return a
    else:
        return b

def OPERATE(a,b,func):
    if type(a) == str:
        a_list = param_dict[a]
    else:
        a_list = a

    if type(b) == str:
        b_list = param_dict[b]
    else:
        b_list = b
    # Handle case where a_list or b_list is a single integer
    if not isinstance(a_list, list):
        a_list = [a_list]
    if not isinstance(b_list, list):
        b_list = [b_list]    
    tmp_list = list()
    ctaidy_tmp_dict = list()
    for i in range(ctaidy*ctaidx):
        ctaidy_tmp_dict.append(list())
        
    if type(a) == str and a.startswith("%ctaid"):
        a_len = len(a_list)
        for a_ in a_list: #found the error!!!!!!!!!!!!!!!!!
            for b_ in b_list:
                tmp_list.append(func(a_,b_))
            if a == "%ctaid.x":
                for b_ in param_dict["%ctaid.y"]:
                    ctaidy_tmp_dict[a_+(b_*a_len)]= tmp_list
            else:
                for b_ in param_dict["%ctaid.x"]:
                    ctaidy_tmp_dict[a_+(b_*a_len)] = tmp_list
            tmp_list = list()
        return ctaidy_tmp_dict
    elif type(b) == str and  b.startswith("%ctaid"):
        b_len = len(b_list)
        for b_ in b_list: 
            for a_ in a_list:
                tmp_list.append(func(a_, b_))
            if b == "%ctaid.x":
                for a_ in param_dict["%ctaid.y"]:
                    ctaidy_tmp_dict[(a_*b_len)+b_]= tmp_list
            else:
                for a_ in param_dict["%ctaid.x"]:
                    ctaidy_tmp_dict[(a_*b_len)+b_] = tmp_list
            tmp_list = list()
        return ctaidy_tmp_dict
    else:
        if len(a_list)!=0  and type(a_list[0]) == list:
            for idx_, a_ in enumerate(a_list):
                for a_int in a_:
                    for b_ in b_list:
                        if type(b_) == list:
                            for b_int in b_:
                                tmp_list.append(func(a_int, b_int))
                        else:
                            tmp_list.append(func(a_int, b_))
                ctaidy_tmp_dict[idx_] = tmp_list
                tmp_list = list()
            return ctaidy_tmp_dict
        elif len(b_list)!=0 and type(b_list[0]) == list:
            for idx_, b_ in enumerate(b_list):
                #print(b_)
                for b_int in b_:
                    for a_ in a_list:
                        tmp_list.append(func(a_, b_int))
                ctaidy_tmp_dict[idx_] = tmp_list
                tmp_list = list()
            #print("DDD")
            #print(ctaidy_tmp_dict)
            return ctaidy_tmp_dict
        else:
            for a_ in a_list:
                for b_ in b_list:
                    tmp_list.append(func(a_,b_))
            return tmp_list
    return tmp_list

def tracing(tree,idx):
    opcode = tree[idx]["opcode"]
      # Handle both "child_num" and "child" keys
    child_num = tree[idx].get("child_num", tree[idx].get("child", 0))
    child_list = list()
    if child_num == 0:
        reg_ = tree[idx].get("reg", tree[idx].get("reg_name", ""))
        return reg_ # str return
    while(child_num>0):
        for id_, node_ in enumerate(tree):
            # Handle both "parent" and "parent_loc" keys
            node_parent = node_.get("parent", node_.get("parent_loc", -1))
            if node_parent == idx:
                child_list.append(id_)
                child_num -= 1

    child_num = tree[idx].get("child_num", tree[idx].get("child", 0))
    if child_num==2:
        if opcode.startswith("add"):
            res_ = OPERATE(tracing(tree,child_list[0]),tracing(tree, child_list[1]),ADD)
            return res_
        elif opcode.startswith("sub"):
            res_ = OPERATE(tracing(tree,child_list[0]),tracing(tree, child_list[1]),SUB)
            return res_
        elif opcode.startswith("shl"):
            res_ = OPERATE(tracing(tree, child_list[0]), tracing(tree, child_list[1]),SHL)
            return res_
        elif opcode.startswith("mul"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),MUL)
            return res_
        elif opcode.startswith("or"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),OR)
            return res_
    elif child_num==3:
        if opcode.startswith("mad"):
            res_ = OPERATE(OPERATE(tracing(tree, child_list[0]), tracing(tree, child_list[1]), MUL), tracing(tree, child_list[2]), MADLO)
            return res_
        elif opcode.startswith("selp"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),tracing(tree, child_list[2]),SELP)
            return res_
    else:
        #print(tree[child_list[0]]["reg"])
        return tracing(tree, child_list[0])

# #############################


def evaluate_predicate(predicate_tree, param_dict):
    """
    Evaluate a predicate syntax tree to determine true/false
    Returns a list of boolean values for each thread
    """
    result = tracing(predicate_tree, 0)
    
    # Convert result to boolean list
    if isinstance(result, list):
        if isinstance(result[0], list):
            # Per-TB results
            bool_results = []
            for tb_result in result:
                tb_bool = [bool(val) for val in tb_result]
                bool_results.append(tb_bool)
            return bool_results
        else:
            # Single list of results
            return [bool(val) for val in result]
    else:
        return [bool(result)]


def eval_predicate_tree(tree, idx, param_dict):
    """
    Evaluate predicate syntax tree to get boolean result
    Similar to tracing() but for predicates
    """
    opcode = tree[idx]["opcode"]
    child_num = tree[idx].get("child_num", tree[idx].get("child", 0))
    child_list = list()
    
    if child_num == 0:
        reg_ = tree[idx].get("reg", tree[idx].get("reg_name", ""))
        # Return parameter value
        if reg_ in param_dict:
            return param_dict[reg_]
        return reg_
    
    # Find children
    # Find children
    while child_num > 0:
        for id_, node_ in enumerate(tree):
            node_parent = node_.get("parent", node_.get("parent_loc", -1))
            if node_parent == idx:
                child_list.append(id_)
                child_num -= 1
    
    child_num = tree[idx].get("child_num", tree[idx].get("child", 0))
    
    # Evaluate based on opcode
    if "setp" in opcode:
        # Extract comparison type from opcode (e.g., setp.gt.s32 -> gt)
        parts = opcode.split('.')
        if len(parts) >= 2:
            comp_type = parts[1]  # gt, lt, ge, le, eq, ne
            
            # Evaluate operands
            op1 = eval_predicate_tree(tree, child_list[0], param_dict)
            op2 = eval_predicate_tree(tree, child_list[1], param_dict)
            
            # Perform comparison
            if comp_type == "gt":
                return OPERATE(op1, op2, lambda a, b: 1 if a > b else 0)
            elif comp_type == "lt":
                return OPERATE(op1, op2, lambda a, b: 1 if a < b else 0)
            elif comp_type == "ge":
                return OPERATE(op1, op2, lambda a, b: 1 if a >= b else 0)
            elif comp_type == "le":
                return OPERATE(op1, op2, lambda a, b: 1 if a <= b else 0)
            elif comp_type == "eq":
                return OPERATE(op1, op2, lambda a, b: 1 if a == b else 0)
            elif comp_type == "ne":
                return OPERATE(op1, op2, lambda a, b: 1 if a != b else 0)
    
    return 0  # Default false


def get_idle_thread_mask(kernel_info, bb_graph, kernel_name, param_dict):
    """
    Evaluate all predicates to determine which threads are idle
    Returns a set of thread indices that should be excluded
    """
    idle_threads = set()
    
    if "predicates" not in kernel_info.get(kernel_name, {}):
        return idle_threads
    
    # Evaluate each predicate
    predicate_results = {}
    for pred_name, pred_tree in kernel_info[kernel_name]["predicates"].items():
        if len(pred_tree) > 0:
            result = eval_predicate_tree(pred_tree, 0, param_dict)
            # Extract predicate register name
            pred_reg = pred_tree[0]["reg_name"]
            predicate_results[pred_reg] = result
    
    # Check each basic block for false branches
    for bb_id, bb_info in bb_graph.get(kernel_name, {}).items():
        if "branch_info" in bb_info:
            branch_info = bb_info["branch_info"]
            pred_reg = branch_info["predicate"]
            is_negated = branch_info["is_negated"]
            
            if pred_reg in predicate_results:
                pred_result = predicate_results[pred_reg]
                
                # Determine which threads go to false branch
                if isinstance(pred_result, list):
                    if isinstance(pred_result[0], list):
                        # Per-TB results
                        for tb_idx, tb_result in enumerate(pred_result):
                            for thread_idx, val in enumerate(tb_result):
                                global_tid = tb_idx * len(tb_result) + thread_idx
                                
                                # Thread is idle if:
                                # - Predicate is false and NOT negated (goes to false branch)
                                # - Predicate is true and IS negated (goes to false branch)
                                if (not val and not is_negated) or (val and is_negated):
                                    idle_threads.add(global_tid)
                    else:
                        # Single list
                        for thread_idx, val in enumerate(pred_result):
                            if (not val and not is_negated) or (val and is_negated):
                                idle_threads.add(thread_idx)
    
    return idle_threads


def filter_addresses_by_active_threads(formular, idle_threads):
    """
    Remove addresses accessed by idle threads
    """
    if not idle_threads or not formular:
        return formular
    
    if isinstance(formular[0], list):
        # Per-TB results
        filtered = []
        for tb_idx, tb_data in enumerate(formular):
            filtered_tb = []
            for thread_idx, addr in enumerate(tb_data):
                global_tid = tb_idx * len(tb_data) + thread_idx
                if global_tid not in idle_threads:
                    filtered_tb.append(addr)
            filtered.append(filtered_tb)
        return filtered
    else:
        # Single list
        return [addr for idx, addr in enumerate(formular) if idx not in idle_threads]




# ============================================================================
# NEW: Loop Evaluation Functions (Algorithm 2: EvalSTloop)
# ============================================================================
# ============================================================================
# NEW: Loop Evaluation Functions - Add these around line 100-150
# (Before your make_ctaid_map function definition)
# ============================================================================

def eval_loop_syntax_tree(loop_tree, loop_predicate_tree, param_dict, max_iterations=10000):
    """
    Algorithm 2: EvalSTloop() - Lines 1-8
    Evaluates all loop iterations and returns all addresses
    """
    addresses = []
    iteration = 0
    
    # Initialize loop iterator
    loop_vars = {}
    for node in loop_tree:
        if node.get("is_loop_variable") and node.get("parent_loc") == -1:
            loop_vars[node["reg_name"]] = 0
    
    # Line 3: while EvalST(BB_label) != 0
    while iteration < max_iterations:
        # Evaluate predicate
        predicate_result = evaluate_predicate_with_loop_vars(
            loop_predicate_tree, param_dict, loop_vars
        )
        
        if not predicate_result:
            break
        
        # Line 5: Update leaf nodes
        updated_tree = update_loop_tree_leaf_nodes(loop_tree, loop_vars)
        
        # Evaluate tree
        current_address = tracing(updated_tree, 0)
        
        # Line 6: Map[TB].insert(EvalST(reg))
        if isinstance(current_address, list):
            addresses.extend(current_address)
        else:
            addresses.append(current_address)
        
        # Line 4: Update iterator
        for var_name in list(loop_vars.keys()):
            new_value = evaluate_loop_variable_update(
                loop_tree, var_name, loop_vars, param_dict
            )
            loop_vars[var_name] = new_value
        
        iteration += 1
    
    return addresses


def update_loop_tree_leaf_nodes(loop_tree, loop_vars):
    """Algorithm 2, Line 5: Update leaf nodes with loop var values"""
    updated_tree = []
    
    for node in loop_tree:
        new_node = node.copy()
        if node.get("child", 0) == 0 and node["reg_name"] in loop_vars:
            new_node["reg_name"] = str(loop_vars[node["reg_name"]])
        updated_tree.append(new_node)
    
    return updated_tree


def evaluate_predicate_with_loop_vars(predicate_tree, param_dict, loop_vars):
    """Algorithm 2, Line 3: Evaluate loop exit condition"""
    if not predicate_tree:
        return False
    
    combined_dict = {**param_dict, **loop_vars}
    
    try:
        result = eval_predicate_tree(predicate_tree, 0, combined_dict)
        if isinstance(result, list):
            return all(result) if result else False
        return bool(result)
    except:
        return False


def evaluate_loop_variable_update(loop_tree, var_name, loop_vars, param_dict):
    """Algorithm 2, Line 4: Update loop iterator"""
    for node in loop_tree:
        if node["reg_name"] == var_name and node.get("parent_loc") == -1:
            combined_dict = {**param_dict, **loop_vars}
            
            if node["opcode"].startswith("add"):
                left_val = get_node_value(loop_tree, node.get("child0", 0), combined_dict)
                right_val = get_node_value(loop_tree, node.get("child1", 0), combined_dict)
                return left_val + right_val
            
            elif node["opcode"].startswith("sub"):
                left_val = get_node_value(loop_tree, node.get("child0", 0), combined_dict)
                right_val = get_node_value(loop_tree, node.get("child1", 0), combined_dict)
                return left_val - right_val
    
    return loop_vars.get(var_name, 0)


def get_node_value(tree, node_idx, param_dict):
    """Get value of a tree node"""
    if node_idx >= len(tree) or node_idx < 0:
        return 0
    
    node = tree[node_idx]
    reg_name = node["reg_name"]
    
    if node.get("child", 0) == 0:
        if reg_name in param_dict:
            val = param_dict[reg_name]
            if isinstance(val, list):
                return val[0] if val else 0
            return val
        try:
            return int(reg_name)
        except:
            return 0
    
    return 0
# #######################################
# ============================================================================
# NEW FUNCTION: GetTBAddressMap() - Algorithm 2 Lines 9-24
# ============================================================================

def GetTBAddressMap(syntax_tree, kernel_name, grid_dim, block_dim, 
                    bb_graph, predicates, formular, param_dict):
    """
    Implements Algorithm 2 Lines 9-24: GetTBAddressMap(ST)
    
    Args:
        syntax_tree: The syntax tree for the kernel
        kernel_name: Name of the kernel being analyzed
        grid_dim: Dictionary with keys 'x', 'y', 'z' for GridDim
        block_dim: Dictionary with keys 'x', 'y', 'z' for BlockDim  
        bb_graph: Basic block graph with predicate information
        predicates: Dictionary of predicate syntax trees
        formular: Formula string from syntax tree evaluation
        
    Returns:
        TB_Address_map: Dictionary mapping TB_id -> {"addresses": set()}
    """
    print(f"[GetTBAddressMap] Starting for kernel: {kernel_name}")
    print(f"  GridDim: ({grid_dim['x']}, {grid_dim['y']}, {grid_dim['z']})")
    print(f"  BlockDim: ({block_dim['x']}, {block_dim['y']}, {block_dim['z']})")
    
    # Initialize TB_Address_map
    TB_Address_map = {}
    
    # Line 10: ntid <- BlockDim
    ntid_x = block_dim['x']
    ntid_y = block_dim['y']
    ntid_z = block_dim['z']
    
    # Line 11: forall ctaid in GridDim do
    for ctaid_z in range(grid_dim['z']):
        for ctaid_y in range(grid_dim['y']):
            for ctaid_x in range(grid_dim['x']):
                
                # Line 12: TB <- get_tb_id(ctaid, GridDim)
                TB_id = (ctaid_z * grid_dim['y'] * grid_dim['x'] + 
                        ctaid_y * grid_dim['x'] + 
                        ctaid_x)
                
                # Initialize address set for this TB
                TB_Address_map[TB_id] = {"addresses": set()}
                
                # Build parameter dictionary for this TB
                # Start with global param_dict (has kernel parameters like array pointers)
                tb_param_dict = param_dict.copy()  # ← START WITH GLOBAL PARAMS!
                
                # Update with TB-specific values
                tb_param_dict.update({
                    "ctaid.x": ctaid_x,
                    "ctaid.y": ctaid_y,
                    "ctaid.z": ctaid_z,
                    "nctaid.x": grid_dim['x'],
                    "nctaid.y": grid_dim['y'],
                    "nctaid.z": grid_dim['z'],
                    "ntid.x": ntid_x,
                    "ntid.y": ntid_y,
                    "ntid.z": ntid_z,
                    # Add normalized versions (without dots)
                    "ctaid_x": ctaid_x,
                    "ctaid_y": ctaid_y,
                    "ctaid_z": ctaid_z,
                    "nctaid_x": grid_dim['x'],
                    "nctaid_y": grid_dim['y'],
                    "nctaid_z": grid_dim['z'],
                    "ntid_x": ntid_x,
                    "ntid_y": ntid_y,
                    "ntid_z": ntid_z
                })
         
                
                # Line 13: forall tid in BlockDim do
                for tid_z in range(ntid_z):
                    for tid_y in range(ntid_y):
                        for tid_x in range(ntid_x):
                            
                            # Build per-thread parameter dictionary
                            thread_param_dict = tb_param_dict.copy()
                            thread_param_dict.update({
                                "tid.x": tid_x,
                                "tid.y": tid_y,
                                "tid.z": tid_z,
                                 # Add normalized versions
                                "tid_x": tid_x,
                                "tid_y": tid_y,
                                "tid_z": tid_z
                            })
                            
                            # Line 14-15: Filter out idle threads in False branch BB
                            # for P not in BB_False_branch do
                            #     if ! EvalST(P) then continue
                            try:
                                if not should_include_thread(predicates, bb_graph, 
                                                            kernel_name, thread_param_dict):
                                    continue  # Skip idle thread
                            except Exception as e:
                                # If predicate evaluation fails, include thread by default
                                pass
                            
                            # Line 16-19: Evaluate formula for this thread
                            try:
                                # Check if this is a loop access (not fully implemented yet)
                                loop_addresses = None  # Placeholder for loop handling
                                
                                if loop_addresses:
                                    # Line 18: EvalSTloop(ST, Map)
                                    TB_Address_map[TB_id]["addresses"].update(loop_addresses)
                                else:
                                    # Line 17: Insert (EvalST(reg) + offset) into Map[TB]
                                    if formular:
                                        address = evaluate_formula(formular, thread_param_dict)
                                        if address is not None:
                                            TB_Address_map[TB_id]["addresses"].add(address)
                            except Exception as e:
                                # If formula evaluation fails, skip this thread
                                continue
    
    # Line 24: return Map
    print(f"[GetTBAddressMap] Complete. Processed {len(TB_Address_map)} thread blocks")
    return TB_Address_map


# ============================================================================
# Supporting Functions for GetTBAddressMap
# ========================================================================

def should_include_thread(predicates, bb_graph, kernel_name, param_dict):
    """
    Algorithm 2 Lines 14-15: Check if thread should be included.
    Returns False if thread is idle (in false branch basic block).
    
    This evaluates all predicates to determine if a thread executes
    the memory access instruction or is in a false branch.
    
    Args:
        predicates: Dictionary of predicate_name -> predicate_syntax_tree
        bb_graph: Basic block graph with branch information
        kernel_name: Name of the kernel
        param_dict: Thread-specific parameters (tid_x, ctaid_x, etc.)
    
    Returns:
        bool: True if thread should be included, False if idle
    """
    if not predicates or kernel_name not in bb_graph:
        return True  # No predicates, include all threads
    
    # Evaluate each predicate for this thread
    predicate_results = {}
    
    for pred_name, pred_tree in predicates.items():
        if not pred_tree or len(pred_tree) == 0:
            continue
            
        try:
            # Evaluate the predicate syntax tree with current thread params
            result = eval_predicate_tree_for_thread(pred_tree, param_dict)
            
            # Extract the actual predicate register name (e.g., %p0)
            # pred_name format: "p0_123" where 123 is line number
            pred_reg = pred_tree[0].get("reg_name", pred_name.split("_")[0])
            predicate_results[pred_reg] = result
            
        except Exception as e:
            # If evaluation fails, assume predicate is true (include thread)
            continue
    
    # Check each basic block for false branches
    for bb_id, bb_info in bb_graph.get(kernel_name, {}).items():
        
        # Check if this BB has branch information
        if "branch_info" not in bb_info:
            continue
        
        branch_info = bb_info["branch_info"]
        pred_reg = branch_info.get("predicate")
        is_negated = branch_info.get("is_negated", False)
        
        if not pred_reg:
            continue
        
        # Get the predicate result for this thread
        if pred_reg not in predicate_results:
            continue
        
        pred_result = predicate_results[pred_reg]
        
        # Determine if thread goes to false branch:
        # - If predicate is FALSE and NOT negated (@%p0): thread goes to false branch
        # - If predicate is TRUE and IS negated (@!%p0): thread goes to false branch
        
        thread_takes_false_branch = (not pred_result and not is_negated) or \
                                   (pred_result and is_negated)
        
        # If this BB is marked as false branch and thread goes there, exclude it
        if bb_info.get("is_false_branch", False) and thread_takes_false_branch:
            return False
        
        # If this BB has the memory access (ld.global) and thread takes false branch, exclude
        if bb_info.get("has_ld_global", False) and thread_takes_false_branch:
            return False
    
    # Thread is active and should be included
    return True





def eval_predicate_tree_for_thread(pred_tree, param_dict):
    """
    Evaluate a predicate syntax tree for a single thread.
    
    Args:
        pred_tree: Predicate syntax tree (list of nodes)
        param_dict: Thread parameters (tid.x, ctaid.x, etc.)
    
    Returns:
        bool: True if predicate evaluates to true, False otherwise
    """
    if not pred_tree or len(pred_tree) == 0:
        return True
    
    try:
        # Start evaluation from root node (index 0)
        result = eval_predicate_node(pred_tree, 0, param_dict)
        
        # Convert to boolean
        if isinstance(result, (list, tuple)):
            # If result is a list, take first element
            result = result[0] if result else 0
        
        return bool(result)
        
    except Exception as e:
        # If evaluation fails, default to True (include thread)
        return True


def eval_predicate_node(tree, node_idx, param_dict):
    """
    Recursively evaluate a predicate syntax tree node.
    
    Args:
        tree: The syntax tree
        node_idx: Current node index
        param_dict: Parameter dictionary
    
    Returns:
        Evaluation result (int/float/bool)
    """
    if node_idx >= len(tree) or node_idx < 0:
        return 0
    
    node = tree[node_idx]
    opcode = node.get("opcode", "")
    child_count = node.get("child", 0)
    
    # Leaf node: return parameter value
    if child_count == 0:
        reg_name = node.get("reg_name", "")
        
        # Normalize key for lookup (remove %, replace . with _)
        normalized_key = reg_name.replace("%", "").replace(".", "_")
        
        # Check both formats in param_dict
        if normalized_key in param_dict:
            val = param_dict[normalized_key]
            if isinstance(val, list):
                return val[0] if val else 0
            return val
        elif reg_name in param_dict:
            val = param_dict[reg_name]
            if isinstance(val, list):
                return val[0] if val else 0
            return val
        
        # Try to parse as immediate value
        try:
            return int(reg_name)
        except ValueError:
            try:
                return float(reg_name)
            except ValueError:
                return 0
    
    # Get child node indices
    child_indices = []
    for i in range(child_count):
        child_key = f"child{i}"
        if child_key in node:
            child_indices.append(node[child_key])
    
    # Evaluate children
    child_values = [eval_predicate_node(tree, idx, param_dict) for idx in child_indices]
    
    # Apply operation based on opcode
    if not opcode:
        return child_values[0] if child_values else 0
    
    # Comparison operations (setp.*)
    if "setp" in opcode.lower():
        if len(child_values) < 2:
            return 0
        
        left, right = child_values[0], child_values[1]
        
        # Extract comparison type from opcode (e.g., setp.gt.s32 -> gt)
        parts = opcode.split('.')
        if len(parts) >= 2:
            comp_type = parts[1].lower()
            
            if comp_type == "gt":
                return 1 if left > right else 0
            elif comp_type == "lt":
                return 1 if left < right else 0
            elif comp_type == "ge":
                return 1 if left >= right else 0
            elif comp_type == "le":
                return 1 if left <= right else 0
            elif comp_type == "eq":
                return 1 if left == right else 0
            elif comp_type == "ne":
                return 1 if left != right else 0
    
    # Arithmetic operations
    if opcode.startswith("add"):
        return sum(child_values)
    elif opcode.startswith("sub") and len(child_values) >= 2:
        return child_values[0] - child_values[1]
    elif opcode.startswith("mul"):
        result = 1
        for val in child_values:
            result *= val
        return result
    elif opcode.startswith("div") and len(child_values) >= 2:
        return child_values[0] / child_values[1] if child_values[1] != 0 else 0
    
    # Logical operations
    elif opcode.startswith("and"):
        return 1 if all(child_values) else 0
    elif opcode.startswith("or"):
        return 1 if any(child_values) else 0
    elif opcode.startswith("not"):
        return 1 if not child_values[0] else 0
    
    # Default: return first child
    return child_values[0] if child_values else 0





            


def get_loop_addresses(bb_graph, kernel_name, formular, param_dict, predicates):
    """
    Algorithm 2 Line 18: EvalSTloop(ST, Map)
    Returns set of addresses accessed by all loop iterations
    """
    if kernel_name not in bb_graph:
        return None
    
    # Find if this access is in a loop BB
    loop_bb = None
    for bb_n, bb_info in bb_graph[kernel_name].items():
        if bb_info.get("is_loop_header", False) and bb_info.get("has_ld_global", False):
            loop_bb = bb_n
            break
    
    if loop_bb is None:
        return None  # Not a loop access
    
    # Get loop information
    bb_info = bb_graph[kernel_name][loop_bb]
    loop_variables = bb_info.get("loop_variables", [])
    loop_predicate_name = bb_info.get("loop_predicate")
    
    if not loop_variables or not loop_predicate_name:
        return None
    
    # Evaluate loop iterations
    loop_addresses = set()
    
    # Get loop predicate tree
    if loop_predicate_name not in predicates:
        return None
    
    loop_predicate = predicates[loop_predicate_name]
    
    # Simulate loop iterations (Algorithm 2 Lines 2-8)
    loop_param_dict = param_dict.copy()
    
    # Find loop iterator variable
    if loop_variables:
        loop_var = loop_variables[0]  # Primary loop variable
        iterator_name = loop_var.get("register", "loop_iter")
        
        # Initialize loop iterator
        loop_param_dict[iterator_name] = 0
        max_iterations = 10000  # Safety limit
        iteration_count = 0
        
        # Line 3: while EvalST(BB_label) != 0 do
        while iteration_count < max_iterations:
            try:
                # Evaluate loop predicate
                pred_result = eval_predicate_tree(loop_predicate, loop_param_dict)
                
                if not pred_result:
                    break  # Exit loop
                
                # Line 6: Map[TB].insert(EvalST(reg))
                address = evaluate_formula(formular, loop_param_dict)
                loop_addresses.add(address)
                
                # Line 4: Update loop_iterator
                # Line 5: Update leaf_node reg in STloop(reg)
                update_operation = loop_var.get("operation", "add")
                
                if update_operation in ["add", "add.s32", "add.u32"]:
                    loop_param_dict[iterator_name] += 1
                elif update_operation in ["sub", "sub.s32", "sub.u32"]:
                    loop_param_dict[iterator_name] -= 1
                else:
                    # Unknown operation, increment by default
                    loop_param_dict[iterator_name] += 1
                
                iteration_count += 1
                
            except Exception as e:
                print(f"[Warning] Loop evaluation error: {e}")
                break
    
    return loop_addresses if loop_addresses else None


def eval_predicate_tree(predicate_tree, param_dict):
    """
    Evaluate a predicate syntax tree with given parameters
    Returns True/False
    """
    if not predicate_tree or len(predicate_tree) == 0:
        return True
    
    # Trace the predicate tree to get formula
    formula_str = tracing(predicate_tree, 0)
    
    # Evaluate the formula
    try:
        result = evaluate_formula(formula_str, param_dict)
        # Convert to boolean
        return bool(result) if result is not None else True
    except:
        return True  # Default to including thread if evaluation fails


def evaluate_formula(formula_str, param_dict):
    """
    Evaluate a formula string with parameter substitution
    """
    if not formula_str:
        return None
    
     # Create a normalized parameter dictionary
    # Keys: tid_x, ctaid_x, etc. (stripped % and . replaced with _)
    normalized_params = {}
    for param_name, param_value in param_dict.items():
        # Normalize key: remove %, replace . with _
        normalized_key = param_name.replace("%", "").replace(".", "_")
        normalized_params[normalized_key] = param_value
    
    # Replace parameter placeholders with actual values
    eval_str = formula_str
    
     # Method 1: Replace {param_name} with actual values
    for param_key, param_value in normalized_params.items():
        placeholder = f"{{{param_key}}}"
        
        if placeholder in eval_str:
            # Handle list values (thread/block indices)
            if isinstance(param_value, list):
                if len(param_value) == 1:
                    eval_str = eval_str.replace(placeholder, str(param_value[0]))
                else:
                    # For formulas with multiple values, we can't evaluate directly
                    # This shouldn't happen in per-thread evaluation
                    print(f"[Warning] List value {param_key}={param_value} in formula")
                    eval_str = eval_str.replace(placeholder, str(param_value[0]))
            else:
                eval_str = eval_str.replace(placeholder, str(param_value))
    
    # Check if all placeholders were replaced
    if "{" in eval_str and "}" in eval_str:
        # Extract remaining placeholders for debugging
        import re
        remaining = re.findall(r'\{([^}]+)\}', eval_str)
        if remaining:
            print(f"[Warning] Unresolved placeholders in formula: {remaining}")
            print(f"  Formula: {formula_str}")
            print(f"  After substitution: {eval_str}")
            print(f"  Available params: {list(normalized_params.keys())}")
            return None
    
    try:
        # Safely evaluate the expression
        result = eval(eval_str, {"__builtins__": {}}, {})
        return result
    except Exception as e:
        print(f"[Warning] Formula evaluation failed: {formula_str} -> {eval_str}")
        print(f"  Error: {e}")
        return None



# ##########################################################
def make_ctaid_map(formular, kernel_info=None, bb_graph=None, kernel_name=None, param_dict=None, loop_trees=None):
    """
    Enhanced version that uses GetTBAddressMap() when possible
    """
    global ctaid_map
    
    # Initialize ctaid_map
    for i in range(ctaidy * ctaidx):
        for j in range(ctaidy * ctaidx):
            ctaid_map[i][j] = 0
    
    # Use GetTBAddressMap if we have all required information
    if kernel_info and bb_graph and kernel_name:
        print(f"[make_ctaid_map] Using GetTBAddressMap for {kernel_name}")
        
        # Build grid_dim and block_dim from global variables
        grid_dim = {
            'x': ctaidx,
            'y': ctaidy,
            'z': 1  # Assume 2D grid
        }
        block_dim = {
            'x': ntidx,
            'y': ntidy,
            'z': 1  # Assume 2D blocks
        }
        
        # Extract predicates from kernel_info
        predicates = {}
        if "predicates" in kernel_info.get(kernel_name, {}):
            predicates = kernel_info[kernel_name]["predicates"]
        
        try:
            # Call Algorithm 2 implementation
            TB_Address_map = GetTBAddressMap(
                syntax_tree=None,  # Not needed for formula-based evaluation
                kernel_name=kernel_name,
                grid_dim=grid_dim,
                block_dim=block_dim,
                bb_graph=bb_graph,
                predicates=predicates,
                formular=formular,
                param_dict=param_dict  # ← ADD THIS!
            )
            
            # Convert TB_Address_map to ctaid_map format
            # Calculate locality between all TB pairs
            num_tbs = ctaidy * ctaidx
            for i in range(num_tbs):
                for j in range(i + 1, num_tbs):
                    if i in TB_Address_map and j in TB_Address_map:
                        # Set intersection to find common addresses
                        common = TB_Address_map[i]["addresses"] & TB_Address_map[j]["addresses"]
                        locality_count = len(common)
                        ctaid_map[i][j] = locality_count
                        ctaid_map[j][i] = locality_count
            
            print(f"[make_ctaid_map] Successfully computed locality using GetTBAddressMap")
            return TB_Address_map  # Return for further analysis if needed
            
        except Exception as e:
            print(f"[Warning] GetTBAddressMap failed: {e}")
            print(f"[Warning] Falling back to legacy computation")
            # Fall through to legacy method below
    
    # ====================================================================
    # LEGACY METHOD (fallback when GetTBAddressMap cannot be used)
    # ====================================================================
    print("[make_ctaid_map] Using legacy method (GetTBAddressMap not available)")
    
    if type(formular[0]) == list:
        print(f"len_0: {len(formular[0])}, len_1: {len(formular[1])}")
        for i in tqdm(range(ctaidy * ctaidx)):
            for j in range(i + 1, ctaidy * ctaidx):
                cnt = 0
                for f_i in formular[i]:
                    for f_j in formular[j]:
                        if f_i == f_j:
                            cnt += 1
                            break
                ctaid_map[j][i] = cnt
    else:
        print("*****************no ctaid*****************************")
    
    return None
# ###########################################################################
# Modify file_open() in ptx_tracing.py

def file_open(file_name):
    """
    Modified to load and use predicate information
    """
    with open(file_name, "r") as json_file:
        syntax_tree = json.load(json_file)
        
        # Load kernel info with predicates
          # Load kernel info with predicates
        # Handle both formats: name_st.json -> name_kernel_info.json
        kernel_info_file = file_name.replace("_st.json", "_kernel_info.json")
        if not kernel_info_file.endswith("_kernel_info.json"):
            kernel_info_file = file_name.replace(".json", "_kernel_info.json")
        
        kernel_info = {}
        if os.path.exists(kernel_info_file):
            with open(kernel_info_file, "r") as ki_file:
                kernel_info = json.load(ki_file)
            print(f"[Info] Loaded kernel_info from {kernel_info_file}")
        else:
            print(f"[Warning] No kernel_info file found: {kernel_info_file}")
        
        # Load BB graph
           # Load BB graph
        # Handle both formats: name_st.json -> name_bb.json
        bb_graph_file = file_name.replace("_st.json", "_bb.json")
        if not bb_graph_file.endswith("_bb.json"):
            bb_graph_file = file_name.replace(".json", "_bb.json")
        bb_graph = {}
        if os.path.exists(bb_graph_file):
            with open(bb_graph_file, "r") as bb_file:
                bb_graph = json.load(bb_file)
            print(f"[Info] Loaded bb_graph from {bb_graph_file}")
        else:
            print(f"[Warning] No bb_graph file found: {bb_graph_file}")
        
        for kernel_name, line in syntax_tree.items():
            print(f"\n{'='*60}")
            print(f"Processing kernel: {kernel_name}")
            print(f"{'='*60}")
            
            # Extract loop trees
            loop_trees = {}
            if kernel_name in kernel_info and "loop_trees" in kernel_info[kernel_name]:
                loop_trees = kernel_info[kernel_name]["loop_trees"]
                print(f"[Info] Loaded {len(loop_trees)} loop trees")
            
            # Separate predicates from memory accesses
            predicates = {}
            memory_accesses = {}
            
            for key, tree in line.items():
                if tree and len(tree) > 0:
                    if tree[0].get("is_predicate", False):
                        predicates[key] = tree
                    else:
                        memory_accesses[key] = tree
            
            print(f"[Info] Found {len(predicates)} predicates, {len(memory_accesses)} memory accesses")
            
            # Initialize kernel_map for locality calculation
            kernel_map = list()
            for i in range(ctaidy * ctaidx):
                kernel_map.append(list())
                for j in range(ctaidy * ctaidx):
                    kernel_map[i].append(0)
# ##################################################################################
# Load the formula file with pre-computed formulas
        formular_file = file_name.replace("_st.json", "_formular.json")
        if not formular_file.endswith("_formular.json"):
            formular_file = file_name.replace(".json", "_formular.json")

        formular_dict = {}
        if os.path.exists(formular_file):
            with open(formular_file, "r") as f_file:
                formular_dict = json.load(f_file)
            print(f"[Info] Loaded formulas from {formular_file}")
            
            # DEBUG: Show what keys are available
            if kernel_name in formular_dict:
                print(f"[Debug] Formula keys for {kernel_name}:")
                for fkey in list(formular_dict[kernel_name].keys())[:5]:  # First 5 keys
                    print(f"  - {fkey}")
        else:
            print(f"[Warning] No formula file found: {formular_file}")

        # Process each memory access
        for id_, (key, tree) in tqdm(enumerate(memory_accesses.items()), 
                                     desc=f"Processing {kernel_name}"):
            
            # Get pre-computed formula from formular_dict
            formular = None

            if kernel_name in formular_dict:
                # Try exact key match first
                if key in formular_dict[kernel_name]:
                    formular = formular_dict[kernel_name][key].get("final_formular", None)
                else:
                    # Try without line number suffix (e.g., %rd8_42 -> %rd8)
                    base_key = key.split('_')[0] if '_' in key else key
                    
                    # Search for any key that starts with this register name
                    for fkey in formular_dict[kernel_name].keys():
                        if fkey.startswith(base_key):
                            formular = formular_dict[kernel_name][fkey].get("final_formular", None)
                            print(f"[Info] Matched {key} -> {fkey}")
                            break

            if not formular:
                print(f"[Warning] No formula found for {key}")
                print(f"  Available keys: {list(formular_dict.get(kernel_name, {}).keys())[:5]}")
                continue  # ← THIS MUST BE INSIDE THE FOR LOOP
            
            # Call make_ctaid_map with all parameters
            TB_Address_map = make_ctaid_map(
                formular=formular,
                kernel_info=kernel_info,
                bb_graph=bb_graph,
                kernel_name=kernel_name,
                param_dict=param_dict,
                loop_trees=loop_trees
            )
            
            # Accumulate locality counts
            for i in range(ctaidy * ctaidx):
                for j in range(i + 1, ctaidy * ctaidx):
                    kernel_map[j][i] += ctaid_map[j][i]   
# ############################################################################
            # Generate and save heatmap (existing code continues...)
            np_kernel_map = np.array(kernel_map)
            dp_kernel_map = pd.DataFrame(np_kernel_map)
            sns.heatmap(dp_kernel_map, cmap="OrRd")
            
            if not os.path.isdir(f"img/{app_name}"):
                os.makedirs(f"img/{app_name}")
            
            plt.savefig(f"img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}.png")
            # After: plt.savefig(f"img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}.png")
            # Add:
            matrix_file = f"img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}_matrix.json"
            with open(matrix_file, 'w') as f:
                json.dump(kernel_map, f)
            print(f"[Info] Saved locality matrix to {matrix_file}")
            plt.clf()
            
            print(f"[Info] Saved locality heatmap to img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}.png")


# ###################################################################            
"""
def backprop_init():
    global tidx
    global tidy
    global ctaidx
    global ctaidy
    
    tidx = 16
    tidy = 16
    ctaidx = 1
    ctaidy = 64 #512

    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidy_list = list(range(0,ctaidy))
    ctaidx_list = list(range(0,ctaidx))


    param_dict["%ctaid.y"] = ctaidy_list #1024/16
    param_dict["%ctaid.x"] = ctaidx_list #1
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["_Z22bpnn_layerforward_CUDAPfS_S_S_ii_param_0"] = [0]
    param_dict["_Z22bpnn_layerforward_CUDAPfS_S_S_ii_param_2"] = [64*16+64]
    param_dict["_Z22bpnn_layerforward_CUDAPfS_S_S_ii_param_5"] = [64*16*2]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_0"] = [64*16*3]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_2"] = [64*16*4]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_5"] = [64*16*5]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_4"] = [64*16*6]
    param_dict["_Z24bpnn_adjust_weights_cudaPfiS_iS_S__param_1"] = [64*16*7]
    param_dict["4"] = [4]
    param_dict["16"] = [16]
    param_dict["1"] = [1]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    file_open("syntax_tree/rodinia/rodinia_backprop.json")
    #res = OPERATE("%ctaid.y",OPERATE("_Z22bpnn_layerforward_CUDAPfS_S_S_ii_param_5","%tid.y",ADD),SHL)


def MM2_init():
    global tidx
    global tidy
    global ctaidx
    global ctaidy
    
    tidx = 32 #32
    tidy = 8 #8
    ctaidx = 32 #8 #32
    ctaidy = 128 #32 #128
    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidx_list = list(range(0,ctaidx))
    ctaidy_list = list(range(0,ctaidy))
    param_dict["%ctaid.x"] = ctaidx_list #1024/16
    param_dict["%ctaid.y"] = ctaidy_list #1024/16
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%ntid.x"] = [tidx]
    param_dict["%ntid.y"] = [tidy]

    param_dict["_Z11mm2_kernel1iiiiffPfS_S__param_7"] = [0]
    param_dict["_Z11mm2_kernel1iiiiffPfS_S__param_8"] = [4194304]
    param_dict["_Z11mm2_kernel2iiiiffPfS_S__param_6"] =[4194304*2]
    param_dict["_Z11mm2_kernel2iiiiffPfS_S__param_7"] = [4194304*3]
    param_dict["_Z11mm2_kernel2iiiiffPfS_S__param_8"] = [4194304*4]
    param_dict['10'] = [10]
    param_dict['4'] = [4]
    param_dict['2048'] = [2048]
    param_dict['2'] = [2]
    param_dict['0'] = [0]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    file_open("syntax_tree/polybench/polybench_2MM.json")

def gemm_init():
    global tidx
    global tidy
    global ctaidx
    global ctaidy
    
    tidx = 32 #32
    tidy = 8 #8
    ctaidx = 2 #2
    ctaidy = 8 #8
    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidx_list = list(range(0,ctaidx))
    ctaidy_list = list(range(0,ctaidy))
    param_dict["%ctaid.x"] = ctaidx_list #1024/16
    param_dict["%ctaid.y"] = ctaidy_list #1024/16
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%ntid.x"] = [tidx]
    param_dict["%ntid.y"] = [tidy]

    param_dict["_Z11gemm_kerneliiiffPfS_S__param_7"] = [4194304*4]
    param_dict["_Z11gemm_kerneliiiffPfS_S__param_5"] = [4194304]
    param_dict["_Z11gemm_kerneliiiffPfS_S__param_6"] =[4194304*2]

    param_dict['0'] = [0]
    param_dict['4'] = [4]
    param_dict['1024'] = [1024]
    param_dict['9'] = [9]
    param_dict['2'] = [2]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    file_open("syntax_tree/polybench/polybench_GEMM.json")


    

def bfs_init():

    global tidx
    global tidy
    global ctaidx
    global ctaidy


    tidx = 256
    tidy = 1
    ctaidx = 256
    ctaidy = 1

    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidx_list = list(range(0,ctaidx))
    ctaidy_list = list(range(0,ctaidy))


    param_dict["%ctaid.x"] = ctaidx_list #1024/16
    param_dict["%ctaid.y"] = ctaidy_list
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["%ntid.x"] = [ctaidx]
    param_dict["%ntid.y"] = [ctaidy]
 
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_2"] = [0]
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_0"] = [64*16]
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_1"] = [64*16*2]
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_4"] = [64*16*3]
    param_dict["_Z6KernelP4NodePiPbS2_S2_S1_i_param_5"] = [64*16*4]
    param_dict["_Z7Kernel2PbS_S_S_i_param_1"] = [64*16*5]
    param_dict["4"] = [4]
    param_dict["16"] = [16]
    param_dict["1"] = [1]
    param_dict["9"] = [9]
    param_dict["8"] = [8]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    #file_open("syntax_tree/rodinia/rodinia_bfs.json")
    file_open("syntax_tree/rodinia/rodinia_bfs_256.json")

def hotspot_init():
    
    global tidx
    global tidy
    global ctaidx
    global ctaidy


    tidx = 16
    tidy = 16
    ctaidx = 4
    ctaidy = 8

    tidx_list = list(range(0,tidx)) #list(range(0,512))
    tidy_list = list(range(0,tidy))
    ctaidx_list = list(range(0,ctaidx))
    ctaidy_list = list(range(0,ctaidy))


    param_dict["%ctaid.x"] = ctaidx_list #1024/16
    param_dict["%ctaid.y"] = ctaidy_list
    param_dict["%tid.y"] = tidy_list #list(range(0,16))
    param_dict["%tid.x"] = tidx_list #list(range(0,16))
    param_dict["%ntid.x"] = [ctaidx]
    param_dict["%ntid.y"] = [ctaidy]
 
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_2"] = [64*16*5]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_0"] = [64*16]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_7"] = [64*16*2]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_6"] = [64*16*3]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_4"] = [64*16*4]
    param_dict["_Z14calculate_tempiPfS_S_iiiiffffff_param_1"] = [64*16*6]

    param_dict["4"] = [4]
    param_dict["16"] = [16]
    param_dict["1"] = [1]
    param_dict["9"] = [9]
    param_dict["8"] = [8]
    for i in range(ctaidy*ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy*ctaidx):
            ctaid_map[i].append(0)
    file_open("syntax_tree/rodinia/rodinia_hotspot.json")
"""

""" if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="select application to trace")
    parser.add_argument('-ap')
    args = parser.parse_args()
    switcher = { 
        "backprop": backprop_init, 
        "MM2": MM2_init, 
        "bfs": bfs_init,
        "hotspot": hotspot_init,
        "gemm": gemm_init
        }
    global app_name
    app_name = args.ap
    switcher.get(app_name)()
    '''
    if os.path.isdir(f"{app_name}") == False:
        os.makedirs(f"{app_name}")
        print("DDDDDDDDDDD")
    '''
    #switcher(app_name)
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTX Locality Analysis")
    parser.add_argument('-f', '--file', required=True, help="Base filename (e.g., vector_256_1_256_1)")
    parser.add_argument('-d', '--dir', default='syntax_tree/', help="Syntax tree directory")
    
    args = parser.parse_args()
    
    global app_name, tidx, tidy, ctaidx, ctaidy, ntidx, ntidy
    
    # Parse filename to get dimensions
    # Format: appname_ctaidx_ctaidy_tidx_tidy
    base_name = args.file.replace('.ptx', '')
    parts = base_name.split('_')
    
    # Extract dimensions from filename
    try:
        tidy = int(parts[-1])      # Last part
        tidx = int(parts[-2])      # Second to last
        ctaidy = int(parts[-3])    # Third to last
        ctaidx = int(parts[-4])    # Fourth to last
        app_name = '_'.join(parts[:-4])  # Everything before dimensions
    except (ValueError, IndexError):
        print(f"[Error] Could not parse dimensions from filename: {base_name}")
        print("[Info] Expected format: name_ctaidx_ctaidy_tidx_tidy")
        sys.exit(1)
    
    ntidx = tidx
    ntidy = tidy
    
    print(f"[Info] Application: {app_name}")
    print(f"[Info] Grid: ({ctaidx}, {ctaidy}), Block: ({tidx}, {tidy})")
    
    # Build paths - files are stored in syntax_tree/{base_name}/
    file_dir = os.path.join(args.dir, base_name)
    json_file = os.path.join(file_dir, f"{base_name}_st.json")  # ← FIXED: Added _st
    param_file = os.path.join(file_dir, f"{base_name}_param.json")
    
    if not os.path.exists(json_file):
        print(f"[Error] Syntax tree not found: {json_file}")
        print(f"[Info] Run: python3 locality_guru.py -d ./original/ -f {base_name}.ptx")
        sys.exit(1)
    
    # Initialize parameter dictionary
    param_dict["%ctaid.x"] = list(range(ctaidx))
    param_dict["%ctaid.y"] = list(range(ctaidy))
    param_dict["%tid.x"] = list(range(tidx))
    param_dict["%tid.y"] = list(range(tidy))
    param_dict["%ntid.x"] = [ntidx]
    param_dict["%ntid.y"] = [ntidy]
    param_dict["%nctaid.x"] = [ctaidx]
    param_dict["%nctaid.y"] = [ctaidy]
    # Add normalized versions (% removed, . replaced with _) for formula evaluation
    param_dict["tid_x"] = list(range(tidx))
    param_dict["tid_y"] = list(range(tidy))
    param_dict["ctaid_x"] = list(range(ctaidx))
    param_dict["ctaid_y"] = list(range(ctaidy))
    param_dict["ntid_x"] = [ntidx]
    param_dict["ntid_y"] = [ntidy]
    param_dict["nctaid_x"] = [ctaidx]
    param_dict["nctaid_y"] = [ctaidy]
    # Load saved parameters
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            saved_params = json.load(f)
            param_dict.update(saved_params)
        print(f"[Info] Loaded parameters from {param_file}")
    
    # Add common constants
    for i in range(100):
        param_dict[str(i)] = [i]
    
    # Initialize ctaid_map
    ctaid_map.clear()
    for i in range(ctaidy * ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy * ctaidx):
            ctaid_map[i].append(0)
    
    # Process the file
    print(f"[Info] Processing {json_file}")
    file_open(json_file)
