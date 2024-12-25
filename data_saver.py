import os
import pickle
import matplotlib.pyplot as plt

def save_img(model_title=""):
    output_folder = os.path.join("eval_models_img_results")
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, model_title+".png")
    plt.savefig(save_path, dpi=300)
    
def save_model_to_pkl(model, filename, folder="model_saver"):
    """Save the trained model to a .pkl file in a specific folder.

    Args:
        model: Trained model object.
        folder (str): Path to the folder where the file should be saved.
        filename (str): Name of the .pkl file (e.g., 'model.pkl').
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Full path to save the file
    file_path = os.path.join(folder, filename)

    # Save the model
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {file_path}")

def load_model_from_pkl(folder, filename):
    """Load a model from a .pkl file in a specific folder.

    Args:
        folder (str): Path to the folder where the file is saved.
        filename (str): Name of the .pkl file (e.g., 'model.pkl').

    Returns:
        The loaded model object.
    """
    # Full path to load the file
    file_path = os.path.join(folder, filename)

    # Load the model
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {file_path}")
    return model
