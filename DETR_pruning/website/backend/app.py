from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/run-command', methods=['POST'])
def run_command():
    data = request.json
    nproc_per_node = data.get('nproc_per_node')
    batch_size = data.get('batch_size')
    no_aux_loss = '--no_aux_loss' if data.get('no_aux_loss') else ''
    eval_mode = '--eval' if data.get('eval') else ''
    resume_path = data.get('resume_path')
    coco_path = data.get('coco_path')
    
    # Construct the command
    command = (
        f"python -m torch.distributed.launch --nproc_per_node={nproc_per_node} --use_env main.py "
        f"--batch_size {batch_size} {no_aux_loss} {eval_mode} "
        f"--resume {resume_path} --coco_path {coco_path}"
    )
    
    # Execute the command
    os.system(command)
    
    return jsonify({'status': 'Command executed successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
