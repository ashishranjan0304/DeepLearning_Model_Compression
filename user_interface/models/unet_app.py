from flask import Blueprint, render_template, request, Response, send_from_directory
import subprocess
import os
import logging

unet_blueprint = Blueprint('unet', __name__)
logging.basicConfig(level=logging.DEBUG)

def generate_output(command):
    logging.debug(f"Running command: {command}")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    logging.debug(f"Command output: {process.stdout}")
    logging.debug(f"Command error: {process.stderr}")
    logging.debug(f"Command return code: {process.returncode}")
    return process.stdout, process.stderr, process.returncode

@unet_blueprint.route('/stream_output')
def stream_output():
    model = request.args.get('model')
    algorithm = request.args.get('algorithm')

    common_params = {
        'dataset': request.args.get('dataset'),
        'output_dir': request.args.get('output_dir'),
        'epochs': request.args.get('epochs'),
        'batch_size': request.args.get('batch_size'),
        'resume': request.args.get('resume'),
        'lr': request.args.get('lr'),
        'nproc_per_node': request.args.get('nproc_per_node')
    }

    common_params = {k: v for k, v in common_params.items() if v}

    if algorithm in ['L1', 'L2']:
        algo_params = {
            'rate_norm': request.args.get('rate_norm'),
            'dist_type': algorithm.lower()
        }
        algo_params = {k: v for k, v in algo_params.items() if v}
        command = f"python -m torch.distributed.launch --nproc_per_node={common_params['nproc_per_node']} --use_env ../UNET/pruning_unet_distributed_metric.py ../Data/CARVANA " + \
                  f"--dataset {common_params['dataset']} --arch UNet --batch_size {common_params['batch_size']} " + \
                  f"--output_dir {common_params['output_dir']} --rate_norm {algo_params['rate_norm']} --dist_type {algo_params['dist_type']} " + \
                  f"--lr {common_params['lr']} --epochs {common_params['epochs']}"
        if 'resume' in common_params and common_params['resume']:
            command += f" --resume {common_params['resume']}"
    elif algorithm == 'FPGM':
        algo_params = {
            'rate_dist': request.args.get('rate_dist'),
            'dist_type': 'fpgm'
        }
        algo_params = {k: v for k, v in algo_params.items() if v}
        command = f"python -m torch.distributed.launch --nproc_per_node={common_params['nproc_per_node']} --use_env ../UNET/pruning_unet_distributed_metric.py ../Data/CARVANA " + \
                  f"--dataset {common_params['dataset']} --arch UNet --batch_size {common_params['batch_size']} " + \
                  f"--output_dir {common_params['output_dir']} --rate_dist {algo_params['rate_dist']} " + \
                  f"--lr {common_params['lr']} --epochs {common_params['epochs']}"
        if 'resume' in common_params and common_params['resume']:
            command += f" --resume {common_params['resume']}"
    else:
        return Response(f"Unrecognized algorithm {algorithm}", status=400)

    logging.debug(f"Constructed command: {command}")

    stdout, stderr, returncode = generate_output(command)

    response = f"Process completed with return code {returncode}\n\n"
    if stdout:
        response += f"Standard Output:\n{stdout}\n\n"
    if stderr:
        response += f"Standard Error:\n{stderr}\n\n"

    return Response(response, mimetype='text/plain')

@unet_blueprint.route('/', methods=['GET', 'POST'])
def index():
    return render_template('unet.html')

@unet_blueprint.route('/images/<path:output_dir>/<filename>')
def get_image(output_dir, filename):
    path = os.path.join(unet_blueprint.root_path, '..', output_dir)
    logging.debug(f"Serving image from path: {path}, filename: {filename}")
    return send_from_directory(path, filename)
