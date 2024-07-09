from flask import Flask, render_template, request, Response, send_from_directory
from models.detr_app import detr_blueprint
from models.vgg_app import vgg_blueprint
from models.resnet_app import resnet_blueprint
from models.unet_app import unet_blueprint 

app = Flask(__name__)

# Register blueprints for each model
app.register_blueprint(detr_blueprint, url_prefix='/detr')
app.register_blueprint(vgg_blueprint, url_prefix='/vgg')
app.register_blueprint(resnet_blueprint, url_prefix='/resnet')
app.register_blueprint(unet_blueprint, url_prefix='/unet')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

