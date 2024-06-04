import matplotlib
import matplotlib.backends.backend_agg as agg
matplotlib.use('Agg')
from flask import request, jsonify
from flask import Flask, render_template, send_file
from flask_cors import CORS
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image
from circleNesting import Circle
from rectangleNesting import Rectangle
import circleNesting as C
import rectangleNesting as R

app = Flask(__name__.split('.')[0])
CORS(app, support_credentials=True)

@app.route('/hello', methods=['Get'])
def hello():
    return "Hey there"

@app.route('/nest', methods=['POST'])
def nest():
    data = request.json
    print(data)

    img_buffer = io.BytesIO()

    if data['type'] == 'rectangle':
        rectList = [Rectangle(rect['width'], rect['height'], rect['t_left'], rect['t_top'], rect['t_right'], rect['t_bottom']) for rect in data['dimensions']]
        nestList = [R.Algorithm(data['frames']['width'], data['frames']['height'], 0, 0)]
        nestList.sort(key=lambda x: x.spaceLeft, reverse=True)

        images = []
        for nest in nestList:
            if not rectList:
                break

            nest.setRectangles(rectList)
            nest.run()

            nestList = nest.get_unnested_rectangle()
            
            ax = nest.plot()

            canvas = agg.FigureCanvasAgg(ax.figure)
            canvas.draw()

            width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()
    
            image_data = canvas.tostring_rgb()
    
            image = Image.frombytes('RGB', (int(width), int(height)), image_data)
            images.append(image)
        
        images[0].save(img_buffer, format='png', save_all=True, append_images=images[1:])
        print(images)

    elif data['type'] == 'circle':
        circList = [Circle(circ['radius'], circ['thickness']) for circ in data['dimensions']]
        frameList = [C.Frame(data['frames']['width'], data['frames']['height'])]
        frameList.sort(key=lambda x: x.get_area(), reverse=True)

        images = []
        for frame in frameList:
            if not circList:
                break

            nest, not_nest = C.Algorithm.nest_circles(circList, frame)
            circList = not_nest

            ax = frame.plot()

            canvas = agg.FigureCanvasAgg(ax.figure)
            canvas.draw()

            width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()
    
            image_data = canvas.tostring_rgb()
    
            image = Image.frombytes('RGB', (int(width), int(height)), image_data)
            images.append(image)

        images[0].save(img_buffer, format='png', save_all=True, append_images=images[1:])

        #print(images)

    img_buffer.seek(0)
    #print(img_buffer.getvalue())
    print(img_buffer)

    #return send_file(img_buffer, mimetype='image/png')
    response_data = {'image': base64.b64encode(img_buffer.read()).decode('utf-8')}

    return jsonify(response_data)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)