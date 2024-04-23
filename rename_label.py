import os

# Directory containing the images and annotation files
dataset_dir = "cat-emotion/data/valid"
annotation_folder = os.path.join(dataset_dir, "labels")

# Define the original label and the new label
original_label = 2
new_label = 1

# Go through each annotation file and replace the original label with the new label
for annotation_file in os.listdir(annotation_folder):
    annotation_path = os.path.join(annotation_folder, annotation_file)
    updated_lines = []

    with open(annotation_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            label_id = int(parts[0])
            if label_id == original_label:
                parts[0] = str(new_label)  # Replace the label
            updated_line = " ".join(parts) + "\n"
            updated_lines.append(updated_line)

    # Write the updated lines back to the file
    with open(annotation_path, "w") as file:
        file.writelines(updated_lines)

print("Label replacement completed.")
