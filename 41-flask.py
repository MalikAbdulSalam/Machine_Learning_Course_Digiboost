from flask import Flask




app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, Flask!'

@app.route('/first_page')
def first():
    return 'Hello, this is my first page'

@app.route('/second_page')
def secnond():
    return 'Hello, this is my sencond page'

if __name__ == '__main__':
    app.run(debug=True)





