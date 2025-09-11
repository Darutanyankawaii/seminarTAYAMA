from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Hello World! Flask is working!</h1>'

if __name__ == '__main__':
    print("Starting test Flask app...")
    app.run(debug=True, host='127.0.0.1', port=8080)
