import os

def create_mapping_from_ground_truth(gt_folder, output_file):
    unique_labels = set()

    # Step 1: Collect all unique labels
    for filename in os.listdir(gt_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(gt_folder, filename)
            with open(file_path, "r") as f:
                for line in f:
                    label = line.strip()
                    if label:
                        unique_labels.add(label)

    # Step 2: Sort the labels alphabetically
    sorted_labels = sorted(unique_labels)

    # Step 3: Write to output file with indices
    with open(output_file, "w") as out_f:
        for idx, label in enumerate(sorted_labels):
            out_f.write(f"{idx} {label}\n")

    print(f"Mapping of {len(sorted_labels)} sub-actions saved to {output_file}.")

# Example usage:
gt_folder_path = r"D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo\data\breakfast_dataset_our_approach\ground_truth"  # Replace with your path
output_file_path = r"D:\Artificial Intelligence\IIT Mandi Internship\MTP_final\Final_MSTCN_repo\data\breakfast_dataset_our_approach\all_sub_actions.txt"

create_mapping_from_ground_truth(gt_folder_path, output_file_path)
