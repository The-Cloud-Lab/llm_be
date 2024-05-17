import os
import json
from models import *
import importlib
import threading
from supabase_client import supabase_client
import time

class LLMWrapper:
    def __init__(self, model_id, dataset_id, hyperparameters):
        self.config_path = 'config.json'
        self.model_id = model_id
        self.models = self.load_config()
        self.dataset_id = dataset_id
        self.hyperparameters = hyperparameters
        self.model_thread = threading.Thread(target=self.run_model)
        self.model_thread.start()

    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file '{self.config_path}' not found.")

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        llms = config[0]['Config']['MultiLLM']['llms']
    
        for item in llms:
            if item['model'] == self.model_id:
                return item
        return "Error: Model ID not found in config."

    def run_model(self):     
        script_path = self.models['file']
        class_name = self.models['class_name']
        
        # Remove .py extension and replace / with .
        module_path = script_path.replace('/', '.')[:-3]
        
        spec = importlib.util.spec_from_file_location(module_path, script_path)
        # Import the module dynamically
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        user_id = supabase_client.auth.current_user['id']
        # Get the class from the module
        ModelClass = getattr(module, class_name)

        self.insert_status_model(self.model_id, self.dataset_id, user_id)

        start_time = time.time()
        # Instantiate the model and run it
        ModelClass(self.dataset_id,self.model_id,self.hyperparameters)

        end_time = time.time()

        self.update_status_model(self.model_id, self.dataset_id, user_id)

        self.log_execution_time(start_time, end_time)
        
        return

    def insert_status_model(model, dataset_id, user_id):
        model_id = supabase_client.table('models').select('id').eq('model', str(model)).execute().get('data',[])[0].get('id')
        insert_data = { 'user_id' : user_id,
                        'model_id' : model_id,
                        'dataset_id' : dataset_id,
                        'status' : False }
        supabase_client.table("trained_models").insert(insert_data).execute()

    def update_status_model(model, dataset_id, user_id):
        model_id = supabase_client.table('models').select('id').eq('model', str(model)).execute().get('data',[])[0].get('id')
        id = supabase_client.table('trained_models').select('id').eq('model_id', str(model_id)).eq('dataset_id',str(dataset_id)).eq('user_id',user_id).execute().get('data',[])[0].get('id')
        upsert_data = {'id':id, 'status' : True}
        supabase_client.table('trained_models').upsert(upsert_data).execute()
    
    # Function to log execution time to Supabase
    def log_execution_time(self,start_time, end_time):
        execution_time = (end_time - start_time)/3600
        user_id = supabase_client.auth.current_user['id']
        # Insert the execution time along with model_id and dataset_id into Supabase
        supabase_client.table('execution_times').insert({'execution_time': execution_time, 'model_id': self.model_id, 'dataset_id': self.dataset_id, 'user_id': user_id}).execute()
