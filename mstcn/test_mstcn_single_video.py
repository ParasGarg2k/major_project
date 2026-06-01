# This python script tests teh MS-TCN model for a single video
# It takes as input a single video's features in a single .npy file

import torch
import numpy as np
import argparse
from model import Trainer

def load_model(model_dir, epoch, num_stages, num_layers, num_f_maps, features_dim, num_classes, device):
    """Loads the trained MS-TCN model from the specified epoch."""
    model = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
    model.model.load_state_dict(torch.load(f"{model_dir}/epoch-{epoch}.model", map_location=device))
    model.model.to(device)
    model.model.eval()
    return model

def test_single_video(model, feature_file, actions_dict, sample_rate, device):
    """Performs inference on a single video's extracted features and upscales to match the estimated original frame count."""
    # Load the extracted features (should be shape: (num_extracted_frames, 2048))
    features = np.load(feature_file)
    print("Feature Shape Before MS-TCN:", features.shape)
    print("Feature Mean Before MS-TCN:", np.mean(features))
    print("Feature Std Dev Before MS-TCN:", np.std(features))
    print("First 5 Feature Vectors:\n", features[:5])

    # Get number of extracted frames
    # num_extracted_frames = features.shape[0]  # e.g., 93
    num_extracted_frames = features.shape[1]

    # Estimate total video frames
    total_frames = num_extracted_frames * sample_rate  # e.g., 93 * 5 = 465

    print(f"🔢 Estimated total frames in video: {total_frames}")

    # Transpose to match MS-TCN expected input format (features_dim, num_frames)
    # features = features.T  # Now shape: (2048, num_extracted_frames)

    # Convert to PyTorch tensor & add batch dimension
    features = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)  # Shape: [1, 2048, num_extracted_frames]

    # Create a mask (since we process full sequences, we use all 1s)
    mask = torch.ones(features.size(), device=device)

    with torch.no_grad():
        predictions = model.model(features, mask)
        _, predicted = torch.max(predictions[-1].data, 1)
        predicted = predicted.squeeze().cpu().numpy()  # Shape: (num_extracted_frames,)

    print("Raw Predicted Indices:", predicted[:20])  # Show first 20 predictions
    
    # Convert predictions to action labels
    idx_to_action = {v: k for k, v in actions_dict.items()}
    predicted_labels = [idx_to_action[idx] for idx in predicted]

    # **UPSAMPLING: Repeat each action label 'sample_rate' times to match original frames**
    upsampled_labels = []
    for label in predicted_labels:
        upsampled_labels.extend([label] * sample_rate)  # Repeat each label 'sample_rate' times

    # **Handle any mismatch in frame count**
    if len(upsampled_labels) > total_frames:
        upsampled_labels = upsampled_labels[:total_frames]  # Trim excess frames
    elif len(upsampled_labels) < total_frames:
        upsampled_labels.extend([upsampled_labels[-1]] * (total_frames - len(upsampled_labels)))  # Repeat last label

    return upsampled_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_file', required=True, help='Path to the .npy file with extracted features')
    parser.add_argument('--model_dir', required=True, help='Directory where the trained model is stored')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number of the trained model to load')
    # parser.add_argument('--mapping_file', required=True, help='Path to the mapping.txt file')
    parser.add_argument('--num_stages', type=int, default=4, help='Number of stages in MS-TCN')
    parser.add_argument('--num_layers', type=int, default=10, help='Number of layers in each stage')
    parser.add_argument('--num_f_maps', type=int, default=64, help='Number of feature maps')
    parser.add_argument('--features_dim', type=int, default=2048, help='Feature dimension')
    parser.add_argument('--sample_rate', type=int, default=1, help='Frame sampling rate used during feature extraction')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')

    args = parser.parse_args()

    # Load action mapping dictionary
    with open("data/pick_place/mapping.txt", 'r') as f:
        actions_dict = {line.split()[1]: int(line.split()[0]) for line in f.read().splitlines()}
    print("🔎 Actions Dictionary:", actions_dict)

    num_classes = len(actions_dict)
    model = load_model(args.model_dir, args.epoch, args.num_stages, args.num_layers, args.num_f_maps, args.features_dim, num_classes, args.device)

    # Run inference on the extracted features
    predicted_labels = test_single_video(model, args.feature_file, actions_dict, args.sample_rate, args.device)

    print("\n✅ Predicted Action Sequence Generated")

    # Save results to a file
    result_file = args.feature_file.replace('.npy', '_predicted_labels.txt')
    with open(result_file, 'w') as f:
        f.write("\n".join(predicted_labels))

    print(f"🚀 Predictions saved to {result_file}")

if __name__ == "__main__":
    main()

'''
CMD command to execute this python script:

python test_mstcn.py --feature_file <feature_file.npy> --model_dir <model_dir> --epoch <epoch_no>

Insert appropriate names/values in place of "<...>"
'''