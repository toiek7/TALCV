import torch  # Import PyTorch for loading the model
import yaml  # Import YAML for saving the configuration
import os  # Import os for file path handling

def extract_yaml(model_path, output_yaml_path):
    """Extract YAML configuration from a PyTorch model checkpoint."""
    try:
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the checkpoint with weights_only=False
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)  # Load full checkpoint
        yaml_config = checkpoint['model'].yaml  # Extract YAML configuration
        print("Original number of classes:", yaml_config['nc'])  # Print nc (should be 10)
        print("Original class names:", checkpoint['model'].names)  # Print class names

        # Create configs directory if it doesn't exist
        os.makedirs('configs', exist_ok=True)

        # Save YAML to file
        with open(output_yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, sort_keys=False)
        print(f"Extracted YAML saved to {output_yaml_path}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # Define paths
    model_path = 'modelzoo/safety.pt'  # Path to safety.pt
    output_yaml_path = 'configs/extracted_yaml.yaml'  # Path to save extracted YAML
    extract_yaml(model_path, output_yaml_path)  # Call the extraction function