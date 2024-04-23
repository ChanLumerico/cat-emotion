import os

# Directory containing the images and annotation files
dataset_dir = "cat-emotion/data/valid"
image_folder = os.path.join(dataset_dir, "images")
annotation_folder = os.path.join(dataset_dir, "labels")

# Label ID to check for deletion
label_to_delete = 1  # change this to the ID of the label you want to delete

# Go through each annotation file and check for the label
for annotation_file in os.listdir(annotation_folder):
    delete_image = False
    annotation_path = os.path.join(annotation_folder, annotation_file)

    with open(annotation_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            label_id = int(parts[0])
            if label_id == label_to_delete:
                delete_image = True
                break

    # If the label was found, delete both the image and the annotation file
    if delete_image:
        # Find and delete the image file corresponding to the annotation
        image_file_name = annotation_file.replace(
            ".txt", ".jpg"
        )  # Change the extension if needed
        image_path = os.path.join(image_folder, image_file_name)

        # Delete the files
        os.remove(annotation_path)
        os.remove(image_path)
        print(f"Deleted {image_file_name} and {annotation_file}")

print("Deletion process completed.")
