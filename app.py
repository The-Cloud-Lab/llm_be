from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from util import *
from llm_wrapper import LLMWrapper
from auth_controller import token_required, AuthenticationController
from config import Config
from supabase_client import supabase_client

app = Flask(__name__)
CORS(app)
AuthenticationController(app)


@app.route('/tasks', methods=['GET'])
@token_required
def get_tasks():
    try:
        # Fetch tasks data from Supabase
        tasks = supabase_client.table('tasks').select('*').execute().get('data', [])

        # Construct response JSON
        task_list = [{'Task': task['task'], 'ID': task['task_id'], 'Description': task['description']} for task in tasks]

        return jsonify({'tasks': task_list})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/models', methods=['GET'])
@token_required
def get_models():
    try:
        filter_param = request.args.get('filter')
        search_param = request.args.get('model_id')
        author_param = request.args.get('author')

        # Check if filter_param is present and fetch the task_id
        task_id = None
        if filter_param:
            task_id = supabase_client.table('tasks').select('id').eq('task_id', filter_param).execute().get('data', [])[0].get('id')

        # Initialize API URL
        api_url = 'https://huggingface.co/api/models?'

        # Construct API URL based on parameters
        if author_param:
            api_url += f'author={author_param}&'
        if search_param:
            api_url += f'search={search_param}&'
        if filter_param:
            api_url += f'filter={filter_param}&'
        api_url += f'sort=downloads&direction=-1&limit=10'

        # Fetch models from Hugging Face API
        response = requests.get(api_url)

        if response.status_code == 200:
            models_data = response.json()
            model_data = []

            # Store data in the Models table and fetch top models
            for model in models_data:
                memory = ModelMemoryUtil.estimate_model_memory(model['modelId'])
                new_model = {
                    'task_id': task_id,
                    'model': model['modelId'],
                    'memory': memory,
                    'hits': model['downloads']
                }
                model_data.append(new_model)

            # Insert fetched models into the database
            if task_id:
               fst = supabase_client.table('models').insert(model_data).execute()

            # Convert the query results to a list of dictionaries
            return prep_data(model_data)
        else:
            return jsonify({'error': f'Request failed with status code {response.status_code}'})

    except Exception as e:
        return jsonify({'error': str(e)})


def prep_data(list):
    res_data = []
    for model in list:
        res_data.append({
            'model_id': model['model'],
            'memory': model['memory'],
            'hits': model['hits']
        })
    return jsonify(res_data)


@app.route('/datasets', methods=['GET'])
@token_required
def get_datasets():
    try:
        # Extract filter parameter from the query string
        filter_param = request.args.get('task_id')
        search_param = request.args.get('dataset_id')
        # Construct the API URL
        api_url = 'https://huggingface.co/api/datasets?'
        if search_param:
            api_url += f'search={search_param}&'
        if filter_param:
            api_url += f'filter={filter_param}&' 
        api_url += f'sort=downloads&'
        api_url += f'direction=-1&'
        api_url += f'limit=20&'
        

        # Send GET request to the Hugging Face API
        response = requests.get(api_url)
        
        # Check if the request was successful
        if response.status_code == 200:
        # Parse the JSON response
            datasets = response.json()
        # Extract only the dataset IDs from the response
            dataset_ids = [dataset['id'] for dataset in datasets]
            return jsonify({'datasets': dataset_ids})
        else:
            return jsonify({'error': 'Failed to fetch datasets from Hugging Face API'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/hyperparameters', methods=['GET'])
@token_required
def get_hyperparameters():
    try:
        # Extract the 'model_id' parameter from the request
        hyperparameters_data = []
        model = request.args.get('model_id')
        model_id = supabase_client.table('models').select('id').eq('model', str(model)).execute().get('data',[])
        user_id = supabase_client.auth.current_user['id']
        if model_id:
            # Query the Hyperparameters table for all entries with the specified model_id
            hyperparameters_data = supabase_client.table('user_hyper_params').select('*').eq('model_id',str(model_id[0].get('id'))).eq('user_id',user_id).execute().get('data',[])

        if not hyperparameters_data:
            hyperparameters_data = supabase_client.table('hyper_params').select('*').execute().get('data',[])
            # If the model_id exists, return its hyperparameters
        hyperparameters_list = [{'model_id': model,
                                'hyper_param': hyper['hyper_param'],
                                'hyper_value': hyper['hyper_value']} for hyper in hyperparameters_data]
    
        return jsonify(hyperparameters_list)

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/train_model', methods=['POST'])
@token_required
def train_model():
    try:
        # Get data from the request's JSON payload
        data = request.json

        # Extract required parameters
        dataset = data.get('dataset')
        model_id = data.get('model_id')
        training_args = data.get('training_args')

        save_user_hyper_params(model_id,training_args)

        if not dataset or not model_id:
            return jsonify({'error': 'Missing required parameters. Please provide dataset, model_id, and training_args'})

        LLMWrapper(dataset_id=dataset, model_id=model_id, hyperparameters=training_args)

        return jsonify({'success': f'Training model with dataset'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

def save_user_hyper_params(model, training_args):
    model_id = supabase_client.table('models').select('id').eq('model', str(model)).execute().get('data',[])[0].get('id')
    user_id = supabase_client.auth.current_user['id']
    hyperparameters_data = supabase_client.table('user_hyper_params').select('*').eq('model_id',str(model_id)).eq('user_id',user_id).execute().get('data',[])

    if training_args and hyperparameters_data:
        try:
            supabase_client.table('user_hyper_params').delete().eq('model_id',str(model_id)).eq('user_id',user_id).execute()
        except Exception as e:
            print(e)

    # If the model_id exists, return its hyperparameters
    hyperparameters_list = [{'model_id': model_id,
                            'hyper_param': key,
                            'hyper_value': value,
                            'user_id' : user_id} for key, value in training_args.items()]
    supabase_client.table('user_hyper_params').insert(hyperparameters_list).execute()


@app.route('/files/<path:file_path>', methods=['POST'])
@token_required
def serve_file(file_path):
    response = FileHandlingUtil.download_file(file_path)
    df = FileHandlingUtil.parse_file(response, file_path)
    if df is not None:
        if FileHandlingUtil.validate_data(df):
            return jsonify({"message": "Validation successful"})
        else:
            return 'Invalid file format. Columns for text and label not found.', 400
    else:
        return 'File not found', 404

@app.route('/model_status', methods=['GET'])
@token_required
def get_model_status():
    try:
        user_id = supabase_client.auth.current_user['id']
        # Fetch tasks data from Supabase
        status = supabase_client.table('trained_models').select('*').eq('user_id',user_id).execute().get('data', [])
        statuses = []
        # Construct response JSON
        for st in status:
            model_id = str(st['model_id'])
            model_data = supabase_client.table('models').select('model').eq('id', model_id).execute().get('data', [])
            if model_data:
                model = model_data[0]['model']
                statuses.append({'model': model, 'status': st['status'], 'dataset' : st['dataset_id']})
        return jsonify({'model_status': statuses})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=False)
