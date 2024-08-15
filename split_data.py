import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

# Argument parsing
parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
parser.add_argument('--image_dir', type=str, required=True, help="Path to the directory containing images of malawi-cyclone.")
parser.add_argument('--label_dir', type=str, required=True, help="Path to the directory containing labels of malawi-cyclone.")
parser.add_argument('--output_dir', type=str, required=True, help="Path to the output directory. This is the folder with xbd data.")

args = parser.parse_args()

# Assigning the paths from command-line arguments
image_dir = args.image_dir
label_dir = args.label_dir
output_dir = args.output_dir

# Create output directories
train_img_dir = os.path.join(output_dir, 'train/images')
val_img_dir = os.path.join(output_dir, 'hold/images')
test_img_dir = os.path.join(output_dir, 'test/images')

train_label_dir = os.path.join(output_dir, 'train/labels')
val_label_dir = os.path.join(output_dir, 'hold/labels')
test_label_dir = os.path.join(output_dir, 'test/labels')

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

# List all image and label files
files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
labels = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]

# Create a dictionary to group images and labels by their base filename
image_pairs = {}
for file in files:
    if 'pre_disaster' in file or 'post_disaster' in file:
        base_name = file.split('_')[0] + '_' + file.split('_')[1]  # Extract base name (e.g., hurricane-michael_00000363)
        if base_name not in image_pairs:
            image_pairs[base_name] = {'pre': None, 'post': None, 'labels': []}
        if 'pre_disaster' in file:
            image_pairs[base_name]['pre'] = file
        elif 'post_disaster' in file:
            image_pairs[base_name]['post'] = file

for label in labels:
    base_name = label.split('_')[0] + '_' + label.split('_')[1]
    if base_name in image_pairs:
        image_pairs[base_name]['labels'].append(label)

# Filter out pairs that do not have both pre and post images, and associated labels
filtered_pairs = {base_name: pair for base_name, pair in image_pairs.items() 
                  if pair['pre'] and pair['post'] and len(pair['labels']) == 2}

# Split the base names
base_names = list(filtered_pairs.keys())
train_base_names, temp_base_names = train_test_split(base_names, test_size=0.3, random_state=42)
val_base_names, test_base_names = train_test_split(temp_base_names, test_size=0.5, random_state=42)

def move_files(base_names, source_img_dir, source_label_dir, dest_img_dir, dest_label_dir):
    for base_name in base_names:
        pre_img = filtered_pairs[base_name]['pre']
        post_img = filtered_pairs[base_name]['post']
        label_files = filtered_pairs[base_name]['labels']
        
        # Move images
        shutil.copy(os.path.join(source_img_dir, pre_img), dest_img_dir)
        shutil.copy(os.path.join(source_img_dir, post_img), dest_img_dir)
        
        # Move labels
        for label_file in label_files:
            shutil.copy(os.path.join(source_label_dir, label_file), dest_label_dir)

# Move files to corresponding directories
move_files(train_base_names, image_dir, label_dir, train_img_dir, train_label_dir)
move_files(val_base_names, image_dir, label_dir, val_img_dir, val_label_dir)
move_files(test_base_names, image_dir, label_dir, test_img_dir, test_label_dir)

print("Data has been split and moved to corresponding folders.")
