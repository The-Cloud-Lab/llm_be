from supabase_client import supabase_client
import os

def get_user_id():
    return supabase_client.auth.current_user['id']

def save_model(model_id, model, dataset_id, user_id):
    try:
        # Construct the full directory path
        save_path = os.path.join('storage', user_id, dataset_id)
        
        # Create the directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save the model
        model.save_pretrained(os.path.join(save_path, model_id))
        print(f"Model saved successfully at: {os.path.join(save_path, model_id)}")
    except Exception as e:
        print(f"Error occurred while saving the model: {str(e)}")
