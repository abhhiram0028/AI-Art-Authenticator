from flask import Flask, request, render_template
import base64
import model  # Import your image classification code

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return "No image uploaded", 400

    
    image_file = request.files['image']

    result = model.test(image_file,"AI")
    print(result)
    return result
    #return render_template('result.html', predicted_label=result, image_url='data:image/jpeg;base64,' + base64.b64encode(image_data).decode())

if __name__ == '__main__':
    app.run(debug=True)
