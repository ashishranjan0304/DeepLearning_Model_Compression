from flask import Blueprint, render_template, request, Response, send_from_directory
import subprocess
import os
import logging

resnet_blueprint = Blueprint('resnet', __name__)
logging.basicConfig(level=logging.DEBUG)

def generate_output(command):
    logging.debug(f"Running command: {command}")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    logging.debug(f"Command output: {process.stdout}")
    logging.debug(f"Command error: {process.stderr}")
    logging.debug(f"Command return code: {process.returncode}")
    return process.stdout, process.stderr, process.returncode

@resnet_blueprint.route('/stream_output')
def stream_output():
    model = request.args.get('model')
    algorithm = request.args.get('algorithm')
    arch = request.args.get('arch')

    common_params = {
        'dataset': request.args.get('dataset'),
        'output_dir': request.args.get('output_dir'),
        'epochs': request.args.get('epochs'),
        'batch_size': request.args.get('batch_size'),
    }

    common_params = {k: v for k, v in common_params.items() if v is not None}

    if algorithm in ['L1', 'L2']:
        algo_params = {
            'rate_norm': request.args.get('rate_norm'),
            'layer_begin': request.args.get('layer_begin'),
            'layer_end': request.args.get('layer_end'),
            'layer_inter': request.args.get('layer_inter'),
            'ngpu': request.args.get('ngpu'),
            'learning_rate': request.args.get('learning_rate'),
            'arch': arch,
            'dist_type': algorithm.lower()
        }
        if request.args.get('resume'):
            algo_params['resume'] = request.args.get('resume')
    elif algorithm == 'FPGM':
        algo_params = {
            'rate_dist': request.args.get('rate_dist'),
            'layer_begin': request.args.get('layer_begin'),
            'layer_end': request.args.get('layer_end'),
            'layer_inter': request.args.get('layer_inter'),
            'ngpu': request.args.get('ngpu'),
            'learning_rate': request.args.get('learning_rate'),
            'arch': arch,
        }
        if request.args.get('resume'):
            algo_params['resume'] = request.args.get('resume')
    elif algorithm in ['SNIP', 'Synflow', 'GRASP', 'our_algo']:
        algo_params = {
            'compression': request.args.get('compression'),
            'pruner': request.args.get('pruner'),
            'gpu': request.args.get('gpu'),
            'model': arch,
            'experiment': request.args.get('experiment'),
            'prune_epochs': request.args.get('prune_epochs')
        }
        if request.args.get('pretrained') == 'True':
            algo_params['pretrained'] = 'True'
            if request.args.get('pretrained_path'):
                algo_params['pretrained_path'] = request.args.get('pretrained_path')
    else:
        return Response(f"Unrecognized algorithm {algorithm}", status=400)

    algo_params = {k: v for k, v in algo_params.items() if v is not None}

    params = {**common_params, **algo_params}

    logging.debug(f"Parameters: {params}")

    if model == 'resnet':
        if algorithm == 'FPGM':
            command = f"python ../FPGM_pruning/pruning_resnet_cifar.py ../Data/ --dataset {params['dataset']} --arch {params['arch']} " + \
                      ' '.join([f"--{k}={v}" if v != '' else f"--{k}" for k, v in params.items()]) + \
                      f" --rate_dist={params['rate_dist']} "
        elif algorithm in ['L1', 'L2']:
            command = f"python ../FPGM_pruning/pruning_resnet_cifar.py ../Data/ --dataset {params['dataset']} --arch {params['arch']} " + \
                      ' '.join([f"--{k}={v}" if v != '' else f"--{k}" for k, v in params.items()]) + \
                      f" --rate_norm={params['rate_norm']} --dist_type {algorithm.lower()}"
        else:
            command = f"python ../snip_snyflow_grasp_pruning/main.py --dataset {params['dataset']}  " + \
                      ' '.join([f"--{k}={v}" if v != '' else f"--{k}" for k, v in params.items() if k != 'pruner']) + \
                      f" --pruner={params['pruner']} --gpu {params['gpu']}"
    else:
        return Response(f"Unrecognized model {model}", status=400)

    logging.debug(f"Constructed command: {command}")

    stdout, stderr, returncode = generate_output(command)

    response = f"Process completed with return code {returncode}\n\n"
    if stdout:
        response += f"Standard Output:\n{stdout}\n\n"
    if stderr:
        response += f"Standard Error:\n{stderr}\n\n"

    return Response(response, mimetype='text/plain')

@resnet_blueprint.route('/', methods=['GET', 'POST'])
def index():
    return render_template('resnet.html')

@resnet_blueprint.route('/images/<path:output_dir>/<filename>')
def get_image(output_dir, filename):
    path = os.path.join(resnet_blueprint.root_path, '..', output_dir)
    logging.debug(f"Serving image from path: {path}, filename: {filename}")
    return send_from_directory(path, filename)
