from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

def build_command(script_name, params, advanced_params):
    command = f"python {script_name}"
    for param, value in params.items():
        if value:
            command += f" --{param.replace('_', '-')} {value}"
    for param, value in advanced_params.items():
        if value and not value.startswith('--'):
            command += f" --{param.replace('_', '-')} {value}"
        elif value:
            command += f" {value}"
    return command

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    if request.method == 'POST':
        algorithm = request.form.get('algorithm')
        model = request.form.get('model')

        params = {}
        advanced_params = {}

        if algorithm == 'FPGM' and model == 'DETR':
            script_name = 'fpgm_pruning_detr.py'
            params = {
                'lr': request.form.get('lr'),
                'batch_size': request.form.get('batch_size'),
                'resume': request.form.get('resume'),
                'coco_path': request.form.get('coco_path'),
                'no_aux_loss': '--no_aux_loss' if request.form.get('no_aux_loss') else '',
                'eval': '--eval' if request.form.get('eval') else '',
                'epochs': request.form.get('epochs'),
                'num_workers': request.form.get('num_workers'),
                'output_dir': request.form.get('output_dir') if not request.form.get('eval') else '',
                'rate_norm': request.form.get('rate_norm'),
                'rate_dist': request.form.get('rate_dist'),
                'layer_begin': request.form.get('layer_begin'),
                'layer_end': request.form.get('layer_end'),
                'layer_inter': request.form.get('layer_inter')
            }
            advanced_params = {
                'lr_backbone': request.form.get('lr_backbone'),
                'weight_decay': request.form.get('weight_decay'),
                'clip_max_norm': request.form.get('clip_max_norm'),
                'nclasses': request.form.get('nclasses'),
                'frozen_weights': request.form.get('frozen_weights'),
                'backbone': request.form.get('backbone'),
                'dilation': '--dilation' if request.form.get('dilation') else '',
                'position_embedding': request.form.get('position_embedding'),
                'enc_layers': request.form.get('enc_layers'),
                'dec_layers': request.form.get('dec_layers'),
                'dim_feedforward': request.form.get('dim_feedforward'),
                'hidden_dim': request.form.get('hidden_dim'),
                'dropout': request.form.get('dropout'),
                'nheads': request.form.get('nheads'),
                'num_queries': request.form.get('num_queries'),
                'pre_norm': '--pre_norm' if request.form.get('pre_norm') else '',
                'masks': '--masks' if request.form.get('masks') else '',
                'set_cost_class': request.form.get('set_cost_class'),
                'set_cost_bbox': request.form.get('set_cost_bbox'),
                'set_cost_giou': request.form.get('set_cost_giou'),
                'mask_loss_coef': request.form.get('mask_loss_coef'),
                'dice_loss_coef': request.form.get('dice_loss_coef'),
                'bbox_loss_coef': request.form.get('bbox_loss_coef'),
                'giou_loss_coef': request.form.get('giou_loss_coef'),
                'eos_coef': request.form.get('eos_coef'),
                'dataset_file': request.form.get('dataset_file'),
                'coco_panoptic_path': request.form.get('coco_panoptic_path'),
                'remove_difficult': '--remove_difficult' if request.form.get('remove_difficult') else '',
                'device': request.form.get('device'),
                'seed': request.form.get('seed'),
                'start_epoch': request.form.get('start_epoch'),
                'world_size': request.form.get('world_size'),
                'dist_url': request.form.get('dist_url'),
                'dist_type': request.form.get('dist_type'),
                'epoch_prune': request.form.get('epoch_prune'),
                'use_cuda': '--use_cuda' if request.form.get('use_cuda') else '',
                'learning_rate': request.form.get('learning_rate'),
                'print_freq': request.form.get('print_freq')
            }

        elif algorithm == 'FPGM' and model == 'ResNet':
            script_name = 'fpgm_pruning_resnet.py'
            params = {
                'dataset': request.form.get('dataset'),
                'arch': request.form.get('arch'),
                'save_path': request.form.get('save_path'),
                'epochs': request.form.get('epochs'),
                'schedule': request.form.get('schedule'),
                'gammas': request.form.get('gammas'),
                'learning_rate': request.form.get('learning_rate'),
                'decay': request.form.get('decay'),
                'batch_size': request.form.get('batch_size'),
                'rate_norm': request.form.get('rate_norm'),
                'rate_dist': request.form.get('rate_dist'),
                'layer_begin': request.form.get('layer_begin'),
                'layer_end': request.form.get('layer_end'),
                'layer_inter': request.form.get('layer_inter'),
                'epoch_prune': request.form.get('epoch_prune')
            }

        # Add more conditions for other combinations of algorithms and models

        command = build_command(script_name, params, advanced_params)
        output = subprocess.getoutput(command)
    
    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
