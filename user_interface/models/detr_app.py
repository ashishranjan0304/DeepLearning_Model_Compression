from flask import Blueprint, render_template, request, Response, send_from_directory
import subprocess
import os
import logging

detr_blueprint = Blueprint('detr', __name__)
logging.basicConfig(level=logging.DEBUG)

def generate_output(command):
    logging.debug(f"Running command: {command}")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    logging.debug(f"Command output: {process.stdout}")
    logging.debug(f"Command error: {process.stderr}")
    logging.debug(f"Command return code: {process.returncode}")
    return process.stdout, process.stderr, process.returncode

@detr_blueprint.route('/stream_output')
def stream_output():
    model = request.args.get('model')
    algorithm = request.args.get('algorithm')
    nproc_per_node = request.args.get('nproc_per_node')

    common_params = {
        'coco_path': request.args.get('coco_path'),
        'epochs': request.args.get('epochs'),
        'lr': request.args.get('lr'),
        'batch_size': request.args.get('batch_size'),
        'num_workers': request.args.get('num_workers'),
        'output_dir': request.args.get('output_dir'),
        'resume': request.args.get('resume'),
    }

    common_params = {k: v for k, v in common_params.items() if v is not None}

    if algorithm in ['L1', 'L2']:
        algo_params = {
            'rate_norm': request.args.get('rate_norm'),
            'dist_type': algorithm.lower(),
            'layer_begin': request.args.get('layer_begin'),
            'layer_end': request.args.get('layer_end')
        }
    elif algorithm == 'FPGM':
        algo_params = {
            'rate_dist': request.args.get('rate_dist'),
            'layer_begin': request.args.get('layer_begin'),
            'layer_end': request.args.get('layer_end')
        }
    elif algorithm in ['SNIP', 'Synflow', 'our_algo']:
        algo_params = {
            'compression': request.args.get('compression'),
            'pruner': request.args.get('pruner')
        }
        if request.args.get('prune'):
            algo_params['prune'] = ''  # No value needed for --prune
    else:
        return Response(f"Unrecognized algorithm {algorithm}", status=400)

    algo_params = {k: v for k, v in algo_params.items() if v is not None}

    params = {**common_params, **algo_params}

    script_name = '../DETR_pruning/prune_detr_FPGM_L1_L2.py' if algorithm in ['FPGM', 'L1', 'L2'] else '../DETR_pruning/detr_prune_snip_synflow.py'

    command = f"python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --use_env {script_name} " + ' '.join([f"--{k}={v}" if v != '' else f"--{k}" for k, v in params.items()])

    logging.debug(f"Constructed command: {command}")

    stdout, stderr, returncode = generate_output(command)

    response = f"Process completed with return code {returncode}\n\n"
    if stdout:
        response += f"Standard Output:\n{stdout}\n\n"
    if stderr:
        response += f"Standard Error:\n{stderr}\n\n"

    return Response(response, mimetype='text/plain')

@detr_blueprint.route('/', methods=['GET', 'POST'])
def index():
    return render_template('detr.html')

@detr_blueprint.route('/images/<path:output_dir>/<filename>')
def get_image(output_dir, filename):
    path = os.path.join(detr_blueprint.root_path, '..', output_dir)
    logging.debug(f"Serving image from path: {path}, filename: {filename}")
    return send_from_directory(path, filename)
