from supabase_client import supabase_client
from config import Config

def get_user_id():
    return supabase_client.auth.current_user['id']

def insert_status_model(model, dataset_id):
    model_id = supabase_client.table('models').select('id').eq('model', str(model)).execute().get('data',[])[0].get('id')
    user_id = get_user_id()
    insert_data = { 'user_id' : user_id,
                    'model_id' : model_id,
                    'dataset_id' : dataset_id,
                    'status' : False }
    supabase_client.table("trained_models").insert(insert_data).execute()

def update_status_model(model, user_id):
    model_id = supabase_client.table('models').select('id').eq('model', str(model)).execute().get('data',[])[0].get('id')
    id = supabase_client.table('trained_model').select('id').eq('model_id', str(model_id)).eq('user_id',user_id).execute().get('data',[])[0].get('id')
    upsert_data = {'id':id, 'status' : True}
    supabase_client.table('trained_models').upsert(upsert_data).execute()

def save_model(model_id, model, user_id):
    try:
        # Save the model
        model.save_pretrained(Config.save_path + model_id + user_id)
        print("Model saved successfully at:", Config.save_path)
    except Exception as e:
        print("Error occurred while saving the model:", str(e))
