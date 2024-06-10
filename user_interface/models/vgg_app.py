from flask import Blueprint, render_template, request, Response, send_from_directory
import subprocess
import os
import logging

vgg_blueprint = Blueprint('vgg', __name__)
logging.basicConfig(level=logging.DEBUG)

def generate_output(command):
    logging.debug(f"Running command: {command}")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    logging.debug(f"Command output: {process.stdout}")
    logging.debug(f"Command error: {process.stderr}")
    logging.debug(f"Command return code: {process.returncode}")
    return process.stdout, process.stderr, process.returncode

@vgg_blueprint.route('/stream_output')
def stream_output():
    model = request.args.get('model')
    algorithm = request.args.get('algorithm')

    common_params = {
        'dataset': request.args.get('dataset'),
        'output_dir': request.args.get('output_dir'),
    }

    common_params = {k: v for k, v in common_params.items() if v is not None}

    if algorithm in ['L1', 'L2']:
        algo_params = {
            'rate_norm': request.args.get('rate_norm'),
            'depth': request.args.get('depth'),
            'batch-size': request.args.get('batch_size'),
            'epochs': request.args.get('epochs'),
        }
        algo_params = {k: v for k, v in algo_params.items() if v is not None}
        command = f"CUDA_VISIBLE_DEVICES=1 python ../FPGM_pruning/pruning_cifar_vgg.py ../Data --dataset {common_params['dataset']} --arch vgg " + \
                  ' '.join([f"--{k}={v}" for k, v in algo_params.items()]) + \
                  f" --rate_norm={algo_params['rate_norm']} --output_dir={common_params['output_dir']}"
    elif algorithm == 'FPGM':
        algo_params = {
            'rate_dist': request.args.get('rate_dist'),
            'depth': request.args.get('depth'),
            'batch-size': request.args.get('batch_size'),
            'epochs': request.args.get('epochs'),
        }
        algo_params = {k: v for k, v in algo_params.items() if v is not None}
        command = f"CUDA_VISIBLE_DEVICES=1 python ../FPGM_pruning/pruning_cifar_vgg.py ../Data --dataset {common_params['dataset']} --arch vgg " + \
                  ' '.join([f"--{k}={v}" for k, v in algo_params.items()]) + \
                  f" --rate_dist={algo_params['rate_dist']} --output_dir={common_params['output_dir']}"
    elif algorithm in ['SNIP', 'Synflow', 'GRASP', 'our_algo']:
        algo_params = {
            'compression': request.args.get('compression'),
            'pruner': request.args.get('pruner'),
            'batch_size': request.args.get('batch_size'),
            'epochs': request.args.get('epochs'),
        }
        if request.args.get('prune'):
            algo_params['prune'] = ''
        algo_params = {k: v for k, v in algo_params.items() if v is not None}
        command = f"python ../snip_snyflow_grasp_pruning/main.py --dataset {common_params['dataset']} --model vgg16 " + \
                  ' '.join([f"--{k}={v}" for k, v in algo_params.items()]) + \
                  f" --pruner={algo_params['pruner']} --output_dir={common_params['output_dir']}"
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

@vgg_blueprint.route('/', methods=['GET', 'POST'])
def index():
    return render_template('vgg.html')

@vgg_blueprint.route('/images/<path:output_dir>/<filename>')
def get_image(output_dir, filename):
    path = os.path.join(vgg_blueprint.root_path, '..', output_dir)
    logging.debug(f"Serving image from path: {path}, filename: {filename}")
    return send_from_directory(path, filename)
