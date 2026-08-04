from PIL import Image
import numpy as np
import os
import glob

def find_colors():
    mask_files = glob.glob("archive/train/train/*/Labels/*.png")
    if not mask_files:
        print("No masks found")
        return

    all_colors = set()
    
    # Check first 5 masks to be fast but comprehensive
    for mask_path in mask_files[:5]:
        img = Image.open(mask_path).convert('RGB')
        colors = np.unique(np.array(img).reshape(-1, 3), axis=0)
        for c in colors:
            all_colors.add(tuple(c))
            
    print("Unique colors across samples:")
    for c in all_colors:
        print(c)

if __name__ == "__main__":
    find_colors()
