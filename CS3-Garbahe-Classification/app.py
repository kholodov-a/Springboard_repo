# =============================================================================
#                             Import Standard Packages
# =============================================================================
import os
import sys
import logging
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import shutil
import matplotlib

# Path prefix: '.' - for local run, '/app' - 
path_prefix = '.' # '/app'
# =============================================================================
#                             Import Custom Modules
# =============================================================================
sys.path.append(os.path.abspath(path_prefix + '/Notebooks'))
from convnext import *                              # Module for ConvNeXt model
from random_samples import select_random_images     # Module to select random images for testing prediction request


# =============================================================================
#                          Logger Configuration
# =============================================================================
logger = logging.getLogger(__name__)
logger.propagate = False
if __name__ == '__main__':
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    # Set the logging level based on configuration.
    logger.setLevel(logging.DEBUG if app_config.get('DEBUG', False) else logging.INFO)
else:
    if not logger.hasHandlers():
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.INFO)


root_logger = logging.getLogger()
# Remove duplicate handlers
if len(root_logger.handlers) > 1:
    root_logger.handlers = root_logger.handlers[:1]


# =============================================================================
#                       Device Detection (CPU, GPU, MPS, etc.)
# =============================================================================
device = get_device()


# =============================================================================
#                               Helper functions
# =============================================================================

def load_images_from_folder(folder: str):
    '''
    Loads images from a specified folder and returns a tensor of image data along with their filenames.

    Parameters:
        folder (str): The path to the folder containing the images.

    Returns:
        tuple: A tuple containing:
            - image_tensors (torch.Tensor): A tensor with the processed images.
            - image_filenames (list of str): A list of filenames corresponding to the images.
    '''

    # Load images from the specified folder
    image_tensors = []
    image_filenames = []
    transform = get_transforms(train=False)

    # Create a list of files in the folder specified
    try:
        # Ignore hidden files
        files = [f for f in os.listdir(folder) if not f.startswith('.')]
    except Exception as e:
        logger.error(f'Error accessing folder {folder}: {e}')
        return torch.empty(0), []

    # Processing each file 
    for filename in files:
        img_path = os.path.join(folder, filename)
        try:
            # Open the image, convert to RGB, transform and accumulate in the list
            with Image.open(img_path) as image:
                image = image.convert('RGB')
                image_tensor = transform(image)
                image_tensors.append(image_tensor)
                image_filenames.append(filename)
        except Exception as e:
            logger.error(f'Could not load {filename}: {e}')

    # Convert the list of tensors into a tensor
    if image_tensors:
        image_tensors = torch.stack(image_tensors)
    else:
        image_tensors = torch.empty(0)

    return image_tensors, image_filenames

def is_folder_exist(folder: str) -> bool:
    '''
    Checks if the provided path exists and is a directory.

    Parameters:
        folder (str): The path to check.

    Returns:
        bool: True if the folder exists and is a directory; otherwise, False.
    '''
    # Return True if the folder exists and is a directory    
    return os.path.exists(folder) and os.path.isdir(folder)


def get_static_images(images_folder: str, image_filenames: list):
    '''
    Copies images from the specified source folder to a corresponding folder within the Flask app's static directory.
    This is used for rendering images on HTML pages.

    Parameters:
        images_folder (str): The source folder containing the images.
        image_filenames (list): A list of image filenames to copy.

    Returns:
        None
    '''

    # Ensure the static directory for images exists
    target_folder = os.path.basename(images_folder)
    static_folder = os.path.join(app.static_folder, target_folder)
    
    # Delete the existing static filder and create new one
    if is_folder_exist(static_folder):
        shutil.rmtree(static_folder)
    os.makedirs(static_folder, exist_ok = True)

    # Copy images to the static folder
    for filename in image_filenames:
        src_path = os.path.join(images_folder, filename)
        dst_path = os.path.join(static_folder, filename)
        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)


# =============================================================================
#                              Flask Application Setup
# =============================================================================
app = Flask(__name__)

# Obtain classes
data = ImagesData('./Data/raw')
classes = data.classes
del data

# Initiate the model
model = build_model(weights_path = app_config['CNX_LOAD'])
model.to(device)


# =============================================================================
#                              Flask Routes
# =============================================================================
@app.route('/')
def index_view():
    '''Render the homepage.'''
    return render_template('index.html')


@app.route('/info')
def info():
    '''
    Display the current configuration settings.

    Returns:
        Rendered HTML or JSON containing the current configuration.
    '''    
    if 'text/html' in request.headers.get('Accept', ''):
        # Default to HTML response
        return render_template('info.html', config = app_config, title = 'Current Configuration')
    else:
        return jsonify(app_config)


@app.route('/change', methods=['GET', 'POST'])
def change_param():
    '''
    Change a parameter of the application configuration.

    Methods:
        GET or POST

    Parameters (as query arguments):
        param (str): Name of the configuration parameter.
        value (str): New value for the parameter.

    Returns:
        JSON or rendered HTML with the updated configuration, or an error message if the parameter is invalid.

    Note:
        - Changes are applied in memory only and will be reset after the application restarts.
        - Use the /save_config route to persist the current parameters as default.
    '''   
    global app_config

    param = request.args.get('param')
    value = request.args.get('value')

    # Check if the parameter exist 
    try:
        if param not in app_config:
            raise ValueError(f"Parameter '{param}' does not exist.")

        # Convert the string value to the type of the existing config entry.
        app_config[param] = type(app_config[param])(value)

    except Exception as e:
        # Render an error page for all exceptions
        if 'text/html' in request.headers.get('Accept', ''):
            return render_template('error.html', error_message = str(e), title = 'Error'), 400
        else:
            return jsonify({'error': str(e)}), 400

    # Return the updated config in the requested format
    if 'text/html' in request.headers.get('Accept', ''):
        return render_template('info.html', config=app_config, 
                               title='New Configuration', 
                               subtitle=f'The new value of {param} is {value}')
    else:
        return jsonify(app_config)


@app.route('/save_config', methods=['GET', 'POST'])
def save():
    '''
    Save the current configuration as the default.

    Methods:
        GET or POST

    Parameters (as query arguments):
        confirm (str): Must be set to 'True' to save the configuration.

    Returns:
        JSON or rendered HTML with the saved configuration, or an error message if confirmation is not provided.
    '''
    confirmation = request.args.get('confirm', False)

    # Chech if the confirmation parameter is True
    if confirmation:
        # Save the config paremeters to json file
        save_config(app_config)

        # Return saved config
        if 'text/html' in request.headers.get('Accept', ''):
            return render_template('info.html', config = app_config, 
                                title = 'The Configuration Has Been Saved',
                                subtitle = f'Configuration file: {DEFAULT_CONFIG_PATH}')
        else:
            return jsonify(app_config)
    else:
        error_message = "You must set the 'confirm' parameter to 'True' to save the config as default."
        if 'text/html' in request.headers.get('Accept', ''):
            return render_template('error.html', error_message = error_message, title = 'Error'), 400
        else:
            return jsonify({'error': error_message}), 400


@app.route('/restore_config', methods=['GET', 'POST'])
def restore():
    '''
    Restore the application configuration from a saved file.

    Methods:
        GET or POST

    Parameters (as query arguments):
        confirm (str): Must be set to 'True' to proceed with restoring the configuration.
        path (str, optional): Path to the configuration file (default is DEFAULT_CONFIG_PATH).

    Returns:
        JSON or rendered HTML with the restored configuration, or an error message if confirmation is not provided.

    Note:
        If the specified configuration file does not exist or is inaccessible, default configuration values
        will be used (which may differ from those saved in app_config.json).
    '''

    global app_config

    confirmation = request.args.get('confirm', False)
    config_path = request.args.get('path', DEFAULT_CONFIG_PATH)

    # Chech if the confirmation parameter is True
    if confirmation:
        # Load the current fonfiguration
        app_config = load_config(config_path)
               
        # Retunr/Display the loaded config
        if 'text/html' in request.headers.get('Accept', ''):
            return render_template('info.html', config = app_config, 
                                title = 'The Configuration Has Been Restored',
                                subtitle = f'From config file: {config_path}')
        else:
            return jsonify(app_config)
    else:
        error_message = "You must set the 'confirm' parameter to 'True' to save the config as default."
        if 'text/html' in request.headers.get('Accept', ''):
            return render_template('error.html', error_message = error_message, title = 'Error'), 400
        else:
            return jsonify({'error': error_message}), 400
        

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    Predict image classes based on the provided images.

    Methods:
        GET or POST

    Parameters (as query arguments):
        folder (str, optional): Folder containing images for prediction.
                                If not specified, the default folder from app_config is used.

    Returns:
        JSON or rendered HTML with a list of filenames and their predicted classes,
        or an error message if the folder does not exist.
    '''

    # Use the folder provided by the user; if not specified, use the default folder from app_config
    folder = request.args.get('folder', app_config['IMAGES_TO_PREDICT'])

    # Raise an error if the specified folder does not exist
    if not is_folder_exist(folder):
        error_message = f'The folder {folder} does not exist'
        if 'text/html' in request.headers.get('Accept', ''):
            return render_template('error.html', error_message = error_message, title = 'Error'), 400
        else:
            return jsonify({'error': error_message}), 400

    # Load images and log the filenames
    image_tensors, image_filenames = load_images_from_folder(folder)
    logger.debug(f'Loaded filenames: {image_filenames}')

    # Check if any images were loaded. For a nonempty tensor, size(0) > 0.
    if image_tensors.dim() > 0 and image_tensors.size(0) > 0:
        model.eval()
        image_tensors = image_tensors.to(device)
        
        # Predict probabilities of images classes and save it in outputs
        outputs = model(image_tensors)
        
        # Convert probabilities to classes labels
        _, predicted_indices = torch.max(outputs, 1)
        predicted_indices = predicted_indices.cpu().numpy().tolist()
        predictions = [classes[i] for i in predicted_indices]

        logger.debug(f'Predictions: {predictions}')

        # Prepare results
        results = [{'filename': filename, 'prediction': prediction}
                   for filename, prediction in zip(image_filenames, predictions)]

        # Check if the request is from a browser or an API client
        if 'text/html' in request.headers.get('Accept', ''):
            # Ensure the static directory for images exists
            get_static_images(folder, image_filenames)

            return render_template('images.html', 
                                   results = results, 
                                   folder = os.path.basename(folder), 
                                   title = 'Prediction Results')
        else:
            return jsonify(results)
    else:
        return jsonify({'error': f'There are no images in the folder: {folder}'}), 400


@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    '''
    Retrain the model with new data.

    Methods:
        GET or POST

    Parameters (as query arguments):
        folder (str, optional): Folder with new data for training. Required for fine-tuning.
        from_scratch (str, optional): 'True' to retrain from scratch (default: 'False').
        verbose (str, optional): 'True' for detailed training logs (default: 'False').

    Returns:
        JSON or rendered HTML with a message indicating successful deployment of the new model,
        along with training graphs and a confusion matrix, or an error message.
    '''

    folder = request.args.get('folder', None)
    from_scratch = (request.args.get('from_scratch', 'False') == 'True')
    verbose = (request.args.get('verbose', 'True') == 'True')


    # Check if the client accepts HTML or JSON
    accept_header = request.headers.get('Accept', '')

    # Check if the folder is specified and exists    
    if folder:
        if not is_folder_exist(folder):
            error_message = f'The folder {folder} does not exist'
            if 'text/html' in accept_header:
                return render_template('error.html', error_message = error_message), 400
            else:
                return jsonify({'error': error_message}), 400
        
    # Fine-tuning the model (i.e., not retraining from scratch) is only possible if a data folder is specified
    if not from_scratch and folder is None:
        error_message = 'The folder is needed to fine-tune the model'
        if 'text/html' in accept_header:
            return render_template('error.html', error_message = error_message), 400
        else:
            return jsonify({'error': error_message}), 400


    # Toggle the 'DEBUG' mode in accordance with 'verbose' parameter
    tmp_debug = app_config['DEBUG']
    app_config['DEBUG'] = verbose
    switch_logging()

    # Train the model
    epoch_stat, test_stat, debug_data = train_model(
        source_dir = './Data/raw',
        new_data_dir = folder,
        from_scratch = from_scratch
    )

    # Plotting only when there is validation data and there are enougth epochs (3 and more) 
    matplotlib.use('Agg')
    plt.ioff()
    if (app_config['VALID_SPLIT_RATIO'] > 0) and (app_config['NUM_EPOCHS'] > 2):
        plot_metrics(epoch_stat['train_accuracy'], epoch_stat['valid_accuracy'], 
                     'Accuracy', 'Training and Validation Accuracy', show = False)
        plot_metrics(epoch_stat['train_loss'], epoch_stat['valid_loss'], 
                     'Loss', 'Training and Validation Loss', show = False)
    
    # Plot the confusion matrix only if test data is available and there is at least one trial.
    if (app_config['TEST_SPLIT_RATIO'] > 0) and (app_config['NUMBER_OF_TRIALS'] > 0):        
        plot_confusion_matrix(test_stat['y_true'], test_stat['y_pred'], 
                                test_stat['classes'], None, show = False)

    # Ensure the directory exists
    images_dir = './static/Graphs'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    # Copy graphs to the static folder for rendering the results HTML page
    shutil.copy('./Images/Loss.jpg', f'{images_dir}/Loss.jpg')
    shutil.copy('./Images/Accuracy.jpg', f'{images_dir}/Accuracy.jpg')
    shutil.copy('./Images/conf_matrix.jpg', f'{images_dir}/conf_matrix.jpg')

    # Rebuild and update the deployed model.
    global model
    model = build_model(weights_path = app_config['CNX_LOAD'])
    model.to(device)

    # Toggle the 'DEBUG' mode back
    app_config['DEBUG'] = tmp_debug
    switch_logging()

    # Returnign or displaying results
    if 'text/html' in request.headers.get('Accept', ''):
        return render_template('retrain.html', subtitle = app_config['CNX_LOAD'])
    else:
        return jsonify({'message': f"New model has been deployed - {app_config['CNX_LOAD']}"})


@app.route('/refresh', methods=['GET', 'POST'])
def refresh():
    '''
    Fetch additional images from the original dataset for testing.

    Methods:
        GET or POST

    Parameters (as query arguments):
        num (int, optional): Number of images to fetch (default is 10).

    Returns:
        JSON or rendered HTML containing a list of filenames for the fetched images,
        or an error message if the input is invalid.

    Note:
        - This function is intended for demonstration purposes; images are sourced from the training dataset.
        - All files in the target folder will be deleted before copying new images.
    '''
    
    # Chech the value of the num parameter. If it is not integer, then use default value 10
    num_arg = request.args.get('num')
    try:
        num = int(num_arg) if num_arg is not None else 10
    except ValueError:
        return jsonify({'error': f'The number of samples is wrong - {num_arg}'}), 400

    # Check if the folder for sample images exist
    if is_folder_exist(app_config['IMAGES_TO_PREDICT']):
        shutil.rmtree(app_config['IMAGES_TO_PREDICT'])
    os.makedirs(app_config['IMAGES_TO_PREDICT'])

    # Cope random sample images to the configured folder and return the list of the sample files
    file_names = select_random_images(app_config['DATA_DIR'], app_config['IMAGES_TO_PREDICT'], num)
    
    # Check if the request is from a browser or an API client
    if 'text/html' in request.headers.get('Accept', ''):
        # Ensure the static directory for images exists
        get_static_images(app_config['IMAGES_TO_PREDICT'], file_names)
        results = [{'filename': filename, 'prediction': ''}
                   for filename in file_names]

        return render_template('images.html', 
                                results = results, 
                                folder = os.path.basename(app_config['IMAGES_TO_PREDICT']), 
                                title = 'Randomly Selected Images')
    else:
        return jsonify(file_names)


# =============================================================================
#                          Run the Application
# =============================================================================
if __name__ == '__main__':
    app.run(debug = False, port = 8000, host = '0.0.0.0')