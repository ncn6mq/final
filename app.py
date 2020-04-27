from flask import Flask, send_file, request

from models.cyclegan import *

app = Flask(__name__)

baseFile = "base.jpg"

@app.route('/', methods=['GET'])
def test():
    return('Hello World')

@app.route('/cyclegan/result', methods=['GET'])
def send_GAN():
    return send_file('result.jpg', attachment_filename='result.jpg')


@app.route('/cyclegan/base', methods = ['GET', 'POST'])
def upload_base():
    if request.method == 'POST':
      f = request.files['file']
      f.save('base.jpg')
      return 'file uploaded successfully'

@app.route('/cyclegan/texture', methods = ['GET', 'POST'])
def upload_texture():
    if request.method == 'POST':
      f = request.files['file']
      f.save('texture.jpg')
      produceResult('base.jpg', 'texture.jpg')
      return 'file uploaded successfully'

# @app.route('/bert/getbert', methods = ['GET', 'POST'])
# def getbert():
#     r

if __name__ == '__main__':
    app.run()
