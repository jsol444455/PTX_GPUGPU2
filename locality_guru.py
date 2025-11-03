import json
import os, sys
import argparse
import re
from ptx_files import ptx_tracing_string
from numpy import sort
from tqdm import tqdm

formular_tree = dict()
syntax_tree = dict()
bb_graph = dict()
kernel_info = dict()
parameter_dict = dict()
inf_ = 987654321
address_ = 1024
search_type = ["ld.global"]
shared_mode = 0
op_dict = {
    "abs":ptx_tracing_string.ABS,
    "add":ptx_tracing_string.ADD,
    "sub":ptx_tracing_string.SUB,
    "mad": ptx_tracing_string.MADLO,
    "fma": ptx_tracing_string.MADLO,
    "mul":ptx_tracing_string.MUL,
    "ld": ptx_tracing_string.LD,
    "st": ptx_tracing_string.ST,
    "not": ptx_tracing_string.NOT,
    "cvt": ptx_tracing_string.CVTA,
    "cvta": ptx_tracing_string.CVTA,
    "mov": ptx_tracing_string.MOV,
    "shl": ptx_tracing_string.SHL,
    "shr": ptx_tracing_string.SHR,
    "or": ptx_tracing_string.OR,
    "bfe": ptx_tracing_string.BFE,
    "prmt": ptx_tracing_string.PRMT,
    "sqrt": ptx_tracing_string.SQRT,
    "min": ptx_tracing_string.MIN,
    "max": ptx_tracing_string.MAX,
    "neg": ptx_tracing_string.NEG,
    "and": ptx_tracing_string.AND,
    "div": ptx_tracing_string.DIV,
    "rem": ptx_tracing_string.REM,
    "rcp": ptx_tracing_string.RCP,
    "selp": ptx_tracing_string.SELP,
    "setp":ptx_tracing_string.SETP,
    "clz": ptx_tracing_string.CLZ,
    "setp.ge": ptx_tracing_string.SETP_GE,
    "setp.lt":ptx_tracing_string.SETP_LT,
    "setp.ne":ptx_tracing_string.SETP_NE,
    "setp.gt":ptx_tracing_string.SETP_GT,
    "setp.eq":ptx_tracing_string.SETP_EQ
}

def get_opcode(inst,search_flag="st"):
    initial = inst
    inst = re.sub("\n","",inst) #inst.split("\n")[0]
    inst = re.sub(r"[,;\[\]\{\}]","",inst)
    inst = inst.split()

    if len(inst)<3:
        return "-999999999", "-999999999", ["-999999999"]
    
    '''
    if tmp[0]=="setp":
        inst[0] = tmp[0]+"."+tmp[1]
    else:
        inst[0] = tmp[0]
    '''
    search_flag = search_flag.split('.')[0]
    if inst[0].startswith(search_flag):
        tmp = inst[0].split(".")
        if(search_flag=="st"):
            # print("DDD")
            t = inst[-1]
            inst[-1] = inst[-2]
            inst[-2] = t
        # print(tmp)
        try:
            if tmp[2]=='v2':
                dst = dict()
                offset = int(tmp[3][1:])
                src_offset = inst[-1].split("+")
                if len(inst[-1].split("+"))>1:
                    dst[inst[1]]=inst[-1]
                    dst[inst[2]]=src_offset[0]+"+"+str(offset+int(src_offset[1]))
                else:
                    dst[inst[1]]=inst[-1]
                    dst[inst[2]]=inst[-1]+"+"+str(offset)
                return inst[0], dst,"-999999999"
        except:
            pass  # Silently skip vector operations that can't be parsed
    # print("DDDDDDDDDDs")
    return inst[0], inst[1], inst[2:]
# ########################################################
def initialize_trees(ptx_file_name):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_
    
    with open(ptx_file_name,"r") as f:
        instructions = f.readlines()
        kernel_name = ""
        inst_len = len(instructions)
        bb_n = -1
        
        for idx_, inst in enumerate(instructions):
            if inst.strip().startswith(".visible"):
                kernel_name = re.sub(r"\(\n","",inst.split(" ")[-1])
                syntax_tree[kernel_name] = dict()
                bb_graph[kernel_name] = dict()
                kernel_info[kernel_name] = {
                    "start_id": idx_, 
                    "end_id": -1,
                    "predicates": {},
                    "loop_variables": {}
                }
                bb_n = -1
                
            elif inst.startswith("}"):
                kernel_info[kernel_name]["end_id"] = idx_
                
            elif inst.startswith("$"):
                try:
                    bb_n = int(re.sub("[:\n]","",inst.split("_")[-1]))
                except:
                    continue
                bb_graph[kernel_name][bb_n] = {
                    "from_bb": inst,
                    "visited": False,
                    "next": -1,
                    "start_line": idx_,
                    "end_line": 0,
                    "has_ld_global": False,
                    "predicates": [],         
                    "is_false_branch": False,
                    "branch_predicate": None,
                    # NEW: Loop detection fields
                    "is_loop_header": False,
                    "loop_predicate": None,
                    "loop_variables": []  # Variables updated in loop 
                }
            # #######################
            # NEW: Detect loop branches (backward branches)
            elif inst.strip().startswith("@") and "bra" in inst and bb_n != -1:
                parts = inst.strip().split()
                if len(parts) >= 2:
                    target = parts[-1].strip().replace("$L__BB", "").replace("_", "").replace(";", "")
                    try:
                        target_bb = int(target.split("_")[-1])
                        # Backward branch = loop
                        if target_bb <= bb_n:
                            bb_graph[kernel_name][target_bb]["is_loop_header"] = True
                            predicate = parts[0].replace("@", "").replace("!", "")
                            bb_graph[kernel_name][target_bb]["loop_predicate"] = predicate
                    except:
                        pass
            
            #
            # NEW: Detect loop iterator updates (EXPANDED to include mad, mul, shl)
            elif bb_n != -1 and any(inst.strip().startswith(op) for op in ["add", "sub", "mad", "mul", "shl"]):
                try:
                    # Determine which operation type this is
                    op_type = None
                    for op in ["add", "sub", "mad", "mul", "shl"]:
                        if inst.strip().startswith(op):
                            op_type = op
                            break
                    
                    if op_type is None:
                        continue
                    
                    opcode, dst, src = get_opcode(inst, op_type)
                    
                    # Check if this is a self-update (dst appears in src list)
                    if isinstance(src, list) and dst in src:
                        if bb_n in bb_graph[kernel_name]:
                            print(f"[LOOP VAR] BB {bb_n}: Detected loop variable {dst} = {opcode}({src})")
                            bb_graph[kernel_name][bb_n]["loop_variables"].append({
                                "register": dst,
                                "operation": opcode,
                                "sources": src,
                                "line": idx_
                            })
                except Exception as e:
                    print(f"[LOOP VAR] Failed to parse instruction: {inst.strip()}, error: {e}")
                    pass
            # #
                
            # NEW: Detect and build syntax trees for ALL predicate operations
            elif inst.strip().split('.')[0] in ["setp", "set"] or \
                 any(inst.strip().startswith(op) for op in ["setp.", "set."]):
                
                opcode, dst, src = get_opcode(inst, "setp")
                
                # Check if destination is a predicate register
                if dst != "-999999999" and (dst.startswith("%p") or dst.startswith("p")):
                    src_ = dst + "_" + str(idx_)
                    
                    # Create predicate syntax tree entry
                    if kernel_name not in kernel_info:
                        continue
                        
                    if "predicates" not in kernel_info[kernel_name]:
                        kernel_info[kernel_name]["predicates"] = {}
                    
                    kernel_info[kernel_name]["predicates"][src_] = list()
                    
                    pred_dict = {
                        "reg_name": dst,
                        "score_from_up": inf_,
                        "score_from_down": 0,
                        "child": 0,
                        "my_idx": -1,
                        "parent_loc": -1,
                        "parent_reg": "",
                        "BB_N": bb_n,
                        "opcode": "",
                        "child0": 0,
                        "child1": 0,
                        "child2": 0,
                        "offset": "0",
                        "line": idx_ - kernel_info[kernel_name]["start_id"],
                        "is_predicate": True,
                        "predicate_type": opcode  # Store full opcode (e.g., setp.gt.s32)
                    }
                    kernel_info[kernel_name]["predicates"][src_].append(pred_dict)
                    
                    if bb_n != -1 and bb_n in bb_graph[kernel_name]:
                        bb_graph[kernel_name][bb_n]["predicates"].append(dst)
                
            # NEW: Track branch instructions and their predicates
            elif inst.strip().startswith("@") or inst.strip().startswith("bra"):
                parts = inst.strip().split()
                
                # Extract predicate from @%p0 or @!%p1 format
                predicate = None
                is_negated = False
                branch_target = None
                
                if parts[0].startswith("@"):
                    pred_part = parts[0].replace("@", "")
                    if pred_part.startswith("!"):
                        is_negated = True
                        predicate = pred_part.replace("!", "")
                    else:
                        predicate = pred_part
                    
                    # Get branch target
                    if len(parts) >= 2:
                        branch_target = parts[-1].replace(";", "").strip()
                
                if predicate and bb_n != -1 and bb_n in bb_graph[kernel_name]:
                    bb_graph[kernel_name][bb_n]["branch_info"] = {
                        "predicate": predicate,
                        "is_negated": is_negated,
                        "target": branch_target,
                        "line": idx_
                    }
                    
                    # Try to get target BB number
                    if branch_target and branch_target.startswith("$"):
                        try:
                            target_bb = int(branch_target.split("_")[-1])
                            
                            # Mark the target BB with predicate info
                            if target_bb not in bb_graph[kernel_name]:
                                bb_graph[kernel_name][target_bb] = {
                                    "from_bb": branch_target,
                                    "visited": False,
                                    "next": -1,
                                    "start_line": -1,
                                    "end_line": -1,
                                    "has_ld_global": False,
                                    "predicates": [],
                                    "is_false_branch": is_negated,  # True branch if NOT negated
                                    "branch_predicate": predicate
                                }
                            else:
                                bb_graph[kernel_name][target_bb]["is_false_branch"] = is_negated
                                bb_graph[kernel_name][target_bb]["branch_predicate"] = predicate
                        except:
                            pass
                            
            elif inst.strip().startswith(search_type[shared_mode]):
                opcode, dst, src = get_opcode(inst, search_type[shared_mode])
                offset = "0"
                tmp = src[-1].split("+")
                if len(tmp) == 2:
                    offset = "+" + tmp[-1]
                src_ = tmp[0] + "_" + str(idx_)
                src = tmp[0]
                syntax_tree[kernel_name][src_] = list()
                
                ld_dict = {
                    "reg_name": src,
                    "score_from_up": inf_,
                    "score_from_down": 0,
                    "child": 0,
                    "my_idx": -1,
                    "parent_loc": -1,
                    "parent_reg": "",
                    "BB_N": bb_n,
                    "opcode": "",
                    "child0": 0,
                    "child1": 0,
                    "child2": 0,
                    "offset": offset,
                    "line": idx_ - kernel_info[kernel_name]["start_id"],
                    "is_predicate": False
                }
                
                if bb_n != -1:
                    bb_graph[kernel_name][bb_n]["has_ld_global"] = True
                syntax_tree[kernel_name][src_].append(ld_dict)
                
            if inst == "\n" and bb_n != -1:
                if bb_n in bb_graph[kernel_name]:
                    bb_graph[kernel_name][bb_n]["end_line"] = idx_
                    bb_n = -1
# ########################################################

# ============================================================================
# NEW: Loop Syntax Tree Functions
# ============================================================================
def build_loop_syntax_tree(loop_var_info, kernel_lines, kernel_name, bb_n):
    """Build STloop for a loop variable"""
    global syntax_tree, inf_
    
    register = loop_var_info["register"]
    operation = loop_var_info["operation"]
    sources = loop_var_info["sources"]
    
    loop_tree = []
    
    root_node = {
        "reg_name": register,
        "score_from_up": inf_,
        "score_from_down": 0,
        "child": len(sources),
        "my_idx": 0,
        "parent_loc": -1,
        "BB_N": bb_n,
        "opcode": operation,
        "child0": 1 if len(sources) > 0 else 0,
        "child1": 2 if len(sources) > 1 else 0,
        "is_loop_variable": True
    }
    loop_tree.append(root_node)
    
    for i, src in enumerate(sources):
        child_node = {
            "reg_name": src,
            "child": 0,
            "my_idx": i + 1,
            "parent_loc": 0,
            "BB_N": bb_n,
            "is_loop_variable": True
        }
        loop_tree.append(child_node)
    
    if "loop_trees" not in kernel_info[kernel_name]:
        kernel_info[kernel_name]["loop_trees"] = {}
    
    loop_tree_key = f"{register}_loop_bb{bb_n}"
    kernel_info[kernel_name]["loop_trees"][loop_tree_key] = loop_tree
    
    return loop_tree

# ##################################
def trace_loop_syntax_trees(kernel_name, kernel_lines):
    """Build loop trees for all loops in kernel"""
    global bb_graph, kernel_info
    
    if kernel_name not in bb_graph:
        return
    
    for bb_n, bb_info in bb_graph[kernel_name].items():
        if bb_info.get("is_loop_header", False):
            for loop_var in bb_info.get("loop_variables", []):
                # Build initial loop tree
                loop_tree = build_loop_syntax_tree(loop_var, kernel_lines, kernel_name, bb_n)
                
                # NOW TRACE THE DEPENDENCIES (this is what was missing!)
                completed_loop_tree = trace_loop_variable_dependencies(
                    loop_tree, 
                    kernel_lines, 
                    bb_n, 
                    kernel_name
                )
                
                # Update the stored loop tree with the completed version
                register = loop_var["register"]
                loop_tree_key = f"{register}_loop_bb{bb_n}"
                kernel_info[kernel_name]["loop_trees"][loop_tree_key] = completed_loop_tree
# #####
def trace_loop_variable_dependencies(loop_tree, kernel_lines, bb_n, kernel_name):
    """
    Trace dependencies of loop variable sources, similar to trace_syntax_tree.
    Updates the loop tree with full dependency information.
    
    This function recursively builds the dependency tree for loop variables
    by tracing backward through the PTX instructions to find where registers
    are defined.
    """
    global bb_graph
    global kernel_info
    global inf_
    
    # Get the basic block information for context
    if kernel_name not in bb_graph or bb_n not in bb_graph[kernel_name]:
        return loop_tree
    
    bb_info = bb_graph[kernel_name][bb_n]
    bb_start_line = bb_info.get("start_line", 0)
    
    # NEW: Create a single visited set for the entire tree
    global_visited = set()
    
    # Trace dependencies for each leaf node (source register)
    for node_idx, node in enumerate(loop_tree):
        # Only process leaf nodes that are not the root
        if node.get("child", 0) == 0 and node.get("my_idx", -1) > 0:
            reg_name = node.get("reg_name", "")
            
            # Skip if already resolved (immediate value or parameter)
            try:
                int(reg_name)
                continue  # It's an immediate value
            except ValueError:
                pass
            
            if not reg_name.startswith("%"):
                continue  # It's a parameter, not a register
            
            # Trace backward from loop BB to find where this register is defined
            loop_tree = _trace_loop_register_definition(
                loop_tree, 
                node_idx,
                reg_name,
                kernel_lines, 
                bb_start_line,
                kernel_name,
                global_visited,  # NEW: Pass shared visited set
                depth=0
            )
    
    return loop_tree

# #################################################################

def _trace_loop_register_definition(loop_tree, node_idx, reg_name, kernel_lines, 
                                     bb_start_line, kernel_name, visited=None, depth=0):
    """
    Helper function to trace a single register's definition backward through PTX.
    Follows the same pattern as trace_syntax_tree().
    
    Args:
        loop_tree: The loop syntax tree being built
        node_idx: Index of the current node in loop_tree
        reg_name: Name of the register to trace (e.g., "%r5")
        kernel_lines: All PTX lines for the kernel
        bb_start_line: Starting line of the basic block
        kernel_name: Name of the kernel being analyzed
    
    Returns:
        Updated loop_tree with dependencies resolved
    """
    global inf_
    
     # NEW: Initialize visited set if not provided
    if visited is None:
        visited = set()
    
    # NEW: Prevent infinite recursion - depth limit
    MAX_DEPTH = 50
    if depth > MAX_DEPTH:
        return loop_tree
    
    # NEW: Check for circular dependency - if we're already tracing this register, stop
    if reg_name in visited:
        return loop_tree
    
    # NEW: Add current register to visited set
    visited.add(reg_name)
    
    current_node = loop_tree[node_idx]
    tree_len = len(loop_tree)
    
    # Search backward from the loop BB start line
    for idx_, line_ in enumerate(reversed(kernel_lines[:bb_start_line])):
        # Parse the instruction - catch errors from complex instructions
        try:
            opcode, dst, src = get_opcode(line_, "add")  # Generic parsing
        except:
            continue  # Skip instructions that can't be parsed
        
        # Skip invalid instructions
        if not isinstance(src, list) or len(src) == 0:
            continue
        if src[0] == "-999999999":
            continue    
        # Check if this instruction writes to our target register
        # Handle both simple dst and dict-type dst (vector operations)
        target_found = False
        actual_src = src
        
        if isinstance(dst, dict):
            if reg_name in dst:
                actual_src = [dst[reg_name]]
                target_found = True
        elif dst == reg_name:
            target_found = True
        
        if not target_found:
            continue
        
        # Found the instruction that defines this register
        # Update the current node with operation information
        current_node["child"] = len(actual_src)
        current_node["opcode"] = opcode
        current_node["line"] = bb_start_line - (idx_ + 1)
        
        # Create child nodes for each source operand
        for src_id, source_reg in enumerate(actual_src):
            # Handle address offsets (e.g., "reg+4")
            offset = "0"
            tmp = source_reg.split("+")
            if len(tmp) == 2:
                offset = "+" + tmp[-1]
                source_reg = tmp[0]
            
            # Create child node
            child_node = {
                "reg_name": source_reg,
                "score_from_up": inf_,
                "score_from_down": 0,
                "child": 0,
                "my_idx": tree_len + src_id,
                "parent_loc": node_idx,
                "parent_reg": reg_name,
                "BB_N": -1,
                "opcode": "",
                "child0": 0,
                "child1": 0,
                "child2": 0,
                "offset": offset,
                "line": bb_start_line - (idx_ + 1),
                "is_loop_variable": True
            }
            
            loop_tree.append(child_node)
            
            # Update parent's child pointer
            current_node[f"child{src_id}"] = tree_len + src_id
        
        # Recursively trace each source register
        for src_id in range(len(actual_src)):
            source_reg_clean = actual_src[src_id].split("+")[0]  # Remove offset
            
            # NEW: Only recurse if this source register hasn't been visited
            if source_reg_clean not in visited:
                # NEW: Create a copy of visited for this branch to allow different paths
                visited_copy = visited.copy()
                
                loop_tree = _trace_loop_register_definition(
                    loop_tree,
                    tree_len + src_id,
                    source_reg_clean,
                    kernel_lines,
                    bb_start_line,
                    kernel_name,
                    visited_copy,  # NEW: Pass visited set
                    depth + 1      # NEW: Increment depth
                )
        
        # Found and processed the definition, stop searching
        break
    
    return loop_tree



# NEW: Add function to trace predicate syntax trees
def trace_predicate_tree(total_pred_info, kernel_lines, start_idx):
    """Build syntax tree for predicate registers"""
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_
    
    pred_info = total_pred_info[start_idx]
    pred_reg = pred_info["reg_name"]
    start_line = pred_info["line"]
    syntax_tree_len = len(total_pred_info)
    
    for idx_, line_ in enumerate(reversed(kernel_lines[:start_line])):
        # Look for instructions that define this predicate
        opcode, dst, src = get_opcode(line_, "setp")
        
        if src[0] == "-999999999":
            continue
            
        if dst == pred_reg:
            pred_info["child"] = len(src)
            pred_info["opcode"] = opcode
            
            for id, s in enumerate(src):
                offset = "0"
                tmp = s.split("+")
                if len(tmp) == 2:
                    offset = "+" + tmp[-1]
                s = tmp[0]
                
                pred_info[f"child{id}"] = syntax_tree_len + id
                pred_dict = {
                    "reg_name": s,
                    "score_from_up": inf_,
                    "score_from_down": 0,
                    "child": 0,
                    "my_idx": id,
                    "parent_loc": start_idx,
                    "parent_reg": pred_reg,
                    "BB_N": -1,
                    "opcode": "",
                    "child0": 0,
                    "child1": 0,
                    "child2": 0,
                    "offset": offset,
                    "line": start_line - (idx_ + 1),
                    "is_predicate": False
                }
                total_pred_info.append(pred_dict)
            
            for id, _ in enumerate(src):
                trace_predicate_tree(total_pred_info, kernel_lines, syntax_tree_len + id)
            break
            
    return total_pred_info
# ########################################################
                    
def trace_syntax_tree(total_ld_global_info, kernel_lines,start_idx):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_
    ld_global_info = total_ld_global_info[start_idx]
    ld_global = ld_global_info["reg_name"]
    #print(ld_global)
    #print(kernel_lines[ld_global_info["line"]])
    start_line = ld_global_info["line"]
    line_length = len(kernel_lines[start_line:])
    syntax_tree_len = len(total_ld_global_info)
    for idx_, line_ in enumerate(reversed(kernel_lines[:start_line])):
        opcode, dst, src = get_opcode(line_,search_type[shared_mode])

        if src[0] == "-999999999":
            continue
        if ld_global in dst and type(dst)==dict:
            # print("dddddddd")
            src = [dst[ld_global]]
            dst = ld_global

        if dst == ld_global:     
            ld_global_info["child"] = len(src)
            ld_global_info["opcode"] = opcode
            for id, s in enumerate(src):
                offset = "0"
                
                tmp = s.split("+")
                if len(tmp) == 2:
                    offset = "+"+tmp[-1]
                s = tmp[0]
                ld_global_info[f"child{id}"] = syntax_tree_len+id
                ld_dict = {"reg_name":s, "score_from_up":inf_, "score_from_down":0, "child":0, "my_idx":id,"parent_loc":start_idx, "parent_reg":ld_global, "BB_N":-1, "opcode":"",  "child0":0,"child1":0,"child2":0, "offset":offset,"line":start_line-(idx_+1)} #-(idx_)
                total_ld_global_info.append(ld_dict)
            
            for id, _ in enumerate(src):
                trace_syntax_tree(total_ld_global_info, kernel_lines,syntax_tree_len+id)
    return total_ld_global_info

def print_syntax_tree(syntax_tree_global, my_idx, depth):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_
    my_dict = syntax_tree_global[my_idx]
    child_len = my_dict["child"]
    full_opcode_ = my_dict["opcode"]
    opcode_ = full_opcode_.split(".")[0]
    options_ = []
    if len(full_opcode_.split("."))>1:
        options_ = full_opcode_.split(".")[1:]
    indent = "    "
    reg_name = my_dict["reg_name"]
    offset_ = my_dict["offset"]
    '''
    print(f"{indent*depth}{reg_name}")
    print(f"{indent*(depth+1)}{opcode_}")
    '''
    if child_len == 0 :
        try:
            param = int(my_dict["reg_name"])
            #parameter_dict[my_dict["reg_name"]] = param
            if offset_ != "0":
                return my_dict["reg_name"]+offset_
            return my_dict["reg_name"]
        except:
            if my_dict["reg_name"].startswith("%"):
                param = re.sub("%","",my_dict["reg_name"])
                param = re.sub(r"\.","_",param)
                #if my_dict["reg_name"] not in parameter_dict:
                #    parameter_dict[my_dict["reg_name"]] = param
                if offset_ != "0":
                    param += offset_
                return "{"+param+"}"
                #ddddd
            if my_dict["reg_name"] not in parameter_dict:
                parameter_dict[my_dict["reg_name"]] = address_
                address_ +=address_
                if offset_ != "0":
                    return "{"+my_dict["reg_name"]+"}"+offset_
            return "{"+my_dict["reg_name"]+"}"
    if child_len == 1:
        tmp1 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child0"],depth+1)
        #print("one child")
        try:
            if offset_ !="0":
                if opcode_=="clz":
                    return op_dict.get(opcode_)(tmp1,full_opcode_) + offset_
                return op_dict.get(opcode_)(tmp1) + offset_
            if opcode_=="clz":
                return op_dict.get(opcode_)(tmp1,full_opcode_)
            return op_dict.get(opcode_)(tmp1) 
        except Exception as e:
            print(e)
            print(opcode_)
            print("************************************")
            exit(1)
    elif child_len == 2:
        tmp1 =print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child0"],depth+1)
        tmp2 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child1"],depth+1)
        #print("two child")
        try:
            if offset_ != "0":
                return op_dict.get(opcode_)(tmp1,tmp2) + offset_
            return op_dict.get(opcode_)(tmp1,tmp2)
        except Exception as e:
            print(e)
            print(tmp1)
            print(tmp2)
            print(options_)
            print(opcode_)
            print("*****************************************")
            exit(1)
    elif child_len == 3:
        #print("three child")
        tmp1 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child0"],depth+1)
        tmp2 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child1"],depth+1)
        tmp3 = print_syntax_tree(syntax_tree_global,syntax_tree_global[my_idx]["child2"],depth+1)
        try:
            if offset_ != "0":
                return op_dict.get(opcode_)(tmp1,tmp2,tmp3) + offset_
            return op_dict.get(opcode_)(tmp1,tmp2, tmp3)
        except:
            print(e)
            print(opcode_)

def trace(ptx_file_name):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    
    with open(ptx_file_name, "r") as f:
        file_lines = f.readlines()
    
    for kernel_name in kernel_info:
        formular_tree[kernel_name] = dict()  # <-- ADD THIS LINE
        kernel_lines = file_lines[kernel_info[kernel_name]["start_id"]:kernel_info[kernel_name]["end_id"]]
        print("       " + kernel_name)
        
        # Build predicate syntax trees
        if "predicates" in kernel_info[kernel_name]:
            for pred_reg in kernel_info[kernel_name]["predicates"]:
                trace_predicate_tree(
                    kernel_info[kernel_name]["predicates"][pred_reg],
                    kernel_lines,
                    0
                )
     
        # NEW: Build loop syntax trees
        trace_loop_syntax_trees(kernel_name, kernel_lines)
        
        # Debug: show loop trees if they exist
        if kernel_name in kernel_info and "loop_trees" in kernel_info[kernel_name]:
            for loop_key, loop_tree in kernel_info[kernel_name]["loop_trees"].items():
                print(f"Loop tree {loop_key}:")
                for node in loop_tree:
                    if node.get("child", 0) > 0:
                        print(f"  Node: {node['reg_name']} = {node['opcode']}(...)")
                    else:
                        print(f"  Leaf: {node['reg_name']}")
        
        # Build regular memory access syntax trees (MUST NOT BE INDENTED INSIDE IF!)
        for ld_global in syntax_tree[kernel_name]:
            formular_tree[kernel_name][ld_global] = dict()
            trace_syntax_tree(
                syntax_tree[kernel_name][ld_global],
                kernel_lines,
                0
            )   
            final_formular = print_syntax_tree(
                syntax_tree[kernel_name][ld_global],
                0,
                0
            )
            print(f"[Debug] Formula for {ld_global}: {final_formular}")  # ← ADD DEBUG
            formular_tree[kernel_name][ld_global]["final_formular"] = final_formular
# #####################

def main(ptx_file_name):
    global formular_tree
    global syntax_tree
    global bb_graph
    global kernel_info
    global parameter_dict
    global address_
    global inf_

    initialize_trees(ptx_file_name)
    trace(ptx_file_name)
    
    # Save everything
    file_dir = ptx_file_name.split("/")[-1].split(".")[0]
    save_dir = f"syntax_tree/{file_dir}"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # Save predicate trees as part of kernel_info
    with open(os.path.join(save_dir, file_dir + "_kernel_info.json"), "w") as f:
        json.dump(kernel_info, f, indent=4)
    
    # Save BB graph
    with open(os.path.join(save_dir, file_dir + "_bb.json"), "w") as f:
        json.dump(bb_graph, f, indent=4)
    
    # Original saves
    with open(os.path.join(save_dir, file_dir + "_param.json"), "w") as f:
        json.dump(parameter_dict, f, indent=4)
    
    with open(os.path.join(save_dir, file_dir + "_st.json"), "w") as f:
        json.dump(syntax_tree, f, indent=4)
    
    with open(os.path.join(save_dir, file_dir + "_formular.json"), "w") as f:
        json.dump(formular_tree, f, indent=4)

# #################################################################
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="dir name",default="./original/")
    parser.add_argument("-f", help="file name",default="")
    parser.add_argument("-s", help="shared mode",default=0, type=int)

    args = parser.parse_args()
    dir_name = args.d
    file_input = args.f
    shared_mode = args.s
    shared_flag=["."]
    print(file_input)

    if dir_name!="" and file_input!="":

        main(os.path.join(dir_name,file_input))
        # exit(1)
        file_dir = file_input.split("/")[-1]
        file_dir = file_dir.split(".")[0]
        print(file_dir)
        save_dir = f"syntax_tree/{file_dir}"
        print(save_dir)
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
        with open(os.path.join(save_dir,file_dir+f"_param.json"),"w") as f:
            json.dump(parameter_dict, f, indent=4)
        with open(os.path.join(save_dir,file_dir+f"_st.json"),"w") as f:
            json.dump(syntax_tree, f, indent=4)
        with open(os.path.join(save_dir,file_dir+f"_formular.json"),"w") as f:
                json.dump(formular_tree, f, indent=4)
    else:
        file_list = os.listdir(dir_name)
        for file_name in file_list:
            
            if file_name == "particlefilter.ptx" :
                continue
            # if file_name.startswith("b+") or file_name.startswith("heart") or file_name.startswith("hotspot") or file_name.startswith("3D") or file_name.startswith("convSep") or file_name.startswith("heart"):
            #    continue
            formular_tree = dict()
            syntax_tree = dict()
            bb_graph = dict()
            kernel_info = dict()
            parameter_dict = dict()
            #address_ = 10240

            #main(os.path.join(dir_name,file_name))
            try:
                print(file_name)
                main(os.path.join(dir_name,file_name))
            except Exception as e:
                print(e)
                print(f"error in {file_name}")
                continue
            file_dir = file_name.split("/")[-1]
            file_dir = file_dir.split(".")[0]
            print(file_dir)
            #exit(1)
            save_dir = f"ptx_files/syntax_tree/{file_dir}"
            print(save_dir)
            if os.path.isdir(save_dir) == False:
                os.makedirs(save_dir)

            with open(os.path.join(save_dir,file_dir+f"_param{shared_flag[shared_mode]}json"),"w") as f:
                json.dump(parameter_dict, f, indent=4)
            with open(os.path.join(save_dir,file_dir+f"_st{shared_flag[shared_mode]}json"),"w") as f:
                json.dump(syntax_tree, f, indent=4)
            with open(os.path.join(save_dir,file_dir+f"_formular{shared_flag[shared_mode]}json"),"w") as f:
                json.dump(formular_tree, f, indent=4)
