#!/usr/bin/env python3
import os
import sys
import re

# Thêm danh sách các thư mục cần kiểm tra vào đây:
DIRS_TO_CHECK = [
    "outputs/train/fs26/v239--satudora_107cia--2s-15frames--cluster-skeleton-8--j",
    "outputs/train/fs26/v240--satudora_107cia--2s-15frames--cluster-skeleton-8--b",
    "outputs/train/fs26/v241--satudora_107cia--2s-15frames--cluster-skeleton-8--jm",
    "outputs/train/fs26/v242--satudora_107cia--2s-15frames--cluster-skeleton-8--bm",
]

def get_best_acc_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    
    # regex to match best_acc: <value>% or best_acc: <value>
    # e.g., best_acc: 48.40%
    pattern = re.compile(r'best_acc:\s*([0-9.]+)(%?)')
    best_acc = None
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    try:
                        val = float(match.group(1))
                        best_acc = val
                    except ValueError:
                        pass
    except Exception as e:
        print(f"Error reading {log_path}: {e}", file=sys.stderr)
        
    return best_acc

def find_best_acc_for_dir(main_dir):
    if not os.path.isdir(main_dir):
        print(f"Directory not found: {main_dir}", file=sys.stderr)
        return None, None
    
    max_acc = -1.0
    best_subdir = None
    
    try:
        subdirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    except Exception as e:
        print(f"Error listing subdirectories of {main_dir}: {e}", file=sys.stderr)
        return None, None
        
    for subdir in sorted(subdirs):
        log_path = os.path.join(subdir, 'log.txt')
        acc = get_best_acc_from_log(log_path)
        if acc is not None:
            if acc > max_acc:
                max_acc = acc
                best_subdir = subdir
                
    if best_subdir is None:
        return None, None
        
    return max_acc, best_subdir

def main():
    dirs = DIRS_TO_CHECK
    if not dirs:
        print("Please add directories to 'DIRS_TO_CHECK' list at the top of the file.", file=sys.stderr)
        sys.exit(1)
        
    for d in dirs:
        max_acc, best_subdir = find_best_acc_for_dir(d)
        if max_acc is not None:
            print(f"Directory: {d}")
            print(f"  Best Accuracy: {max_acc}%")
            print(f"  Path: {best_subdir}")
        else:
            print(f"Directory: {d}")
            print("  No best_acc found in any log.txt under subdirectories.")
        print("-" * 50)

if __name__ == '__main__':
    main()
