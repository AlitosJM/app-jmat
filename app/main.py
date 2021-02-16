from flask import Flask, render_template, redirect, url_for, request, Response
from sklearn.linear_model import LinearRegression
from threading import Thread, currentThread
from .post import Post
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import queue
import time
import asyncio
import os


# On gitBash
# $ export FLASK_APP=main.py,
# $ export FLASK_RUN_PORT=8000
# $ Flask run

app = Flask(__name__)

post_objects = []
# post_objects.append(Post(0, "¡Hola mundo!", "😄", Post.classmethod()))
post_objects.append(Post(0, "¡Hola mundo!", "😄", Post.CONST_NUM0))
post_objects.append(Post(1, "Regresión Lineal", "📉", "Hi there..."))
post_objects.append(Post(3, "Amigo robot", "🤖", "Robot"))

que = queue.Queue()

# Upload folder
UPLOAD_FOLDER = "app/static/files"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

logFile = logging.FileHandler(UPLOAD_FOLDER.replace('files','') + 'LogFile.log', mode='a')
# logFile = logging.FileHandler(UPLOAD_FOLDER+'/LogFile.log', mode='a')
logFile.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
logFile.setFormatter(formatter)
logging.getLogger('').addHandler(logFile)
LOG = logging.getLogger('logFile')

# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def csv_delete(path: str = ''):
    table_list = []
    for filename in os.listdir(path):
        if filename.endswith('.csv'):
            # print(filename)
            table_list.append(filename)
            # table_list.append(pd.read_csv(app.config['UPLOAD_FOLDER'] + '/' + filename, sep="|"))
            # new_table_list.append(filename.split(".")[0])

    if len(table_list) > 2:
        for filename in table_list:
            os.remove(app.config['UPLOAD_FOLDER'] + '/' + filename)


def single_linear_regression(path: str = '', x_to_predict: str = '') -> dict:
    time.sleep(0.5)
    data = pd.read_csv(app.config['UPLOAD_FOLDER'] + '/' + path)
    time.sleep(0.5)

    col = data.columns.values.tolist()
    time.sleep(0.5)

    x = pd.DataFrame(data, columns=[col[0]])
    y = pd.DataFrame(data, columns=[col[1]])
    time.sleep(0.5)

    min_max = [min(x.to_numpy()), max(x.to_numpy()), min(y.to_numpy()), max(y.to_numpy())]
    time.sleep(0.5)

    regression = LinearRegression()
    time.sleep(0.5)

    regression.fit(x, y)
    time.sleep(0.5)

    final_feature = np.array([[int(x_to_predict)]])
    time.sleep(0.5)

    y_ = regression.predict(x)
    time.sleep(0.5)
    prediction = regression.predict(final_feature)
    time.sleep(0.5)

    outputs = {'X': x, 'y': y, 'y_': y_, 'prediction': prediction[0], 'col': col, 'min_max': min_max}
    return outputs


@app.route('/')
def home():
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            csv_delete(app.config['UPLOAD_FOLDER'])
        else:
            os.makedirs(app.config['UPLOAD_FOLDER'])
            raise Exception("No directory")
    except Exception as e:
        LOG.exception("Exception occurred exc: ")
        LOG.error(f"Exception occurred err:{str(e)}")
    finally:
        return render_template("intro.html")


async def progress():
    x = 0
    while x <= 100:
        str0 = "data:" + str(x) + "\n\n"
        x = x + 10
        await asyncio.sleep(0.5)
    return await Response(progress(), mimetype='text/event-stream')


@app.route('/home/')
def get_all_posts():
    try:
        # print(post_objects)
        csv_delete(app.config['UPLOAD_FOLDER'])
    except Exception as e:
        LOG.exception("Exception occurred exc: ")
        LOG.error(f"Exception occurred err:{str(e)}")
    finally:
        return render_template("index.html", all_posts=post_objects)


@app.route("/post/")
@app.route("/post/<int:index>")
def show_post(index: int = None):
    try:
        requested_post = None
        if index is not None:
            for blog_post in post_objects:
                if blog_post.id == index:
                    requested_post = blog_post
            if index != 3:
                return render_template("layout.html", post=requested_post)
            else:
                return render_template("robot.html", post=requested_post)
    except Exception as e:
        LOG.exception("Exception occurred exc: ")
        LOG.error(f"Exception occurred err:{str(e)}")
    return redirect(url_for('get_all_posts'))


@app.route('/save/', methods=['GET', 'POST'])
def save():
    try:
        uploaded_file = request.files['file']
        if request.method == 'POST' and uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)

            return render_template("forecasting.html", file=uploaded_file.filename)
    except Exception as e:
        LOG.exception("Exception occurred exc: ")
        LOG.error(f"Exception occurred err:{str(e)}")
    return redirect(url_for('get_all_posts'))


@app.route("/predict/<string:file_path>", methods=['GET', 'POST'])
def linear_regression(file_path: str = ''):
    try:
        if file_path == 'start':
            return Response(progress(), mimetype='text/event-stream')

        if request.method == 'POST':
            int_features = [int(x) for x in request.form.values()]
            x_to_predict = request.form["input1"]

            xnew = int(x_to_predict)

            thread0 = Thread(target=lambda q, path, forecasting: q.put(single_linear_regression(path, forecasting)),
                             args=(que, file_path, x_to_predict))

            thread0.start()
            redirect(url_for('get_all_posts'))
            time.sleep(1.0)
            thread0.join()
            outputs = que.get()

            prediction_text = 'Para '+x_to_predict+', la salida es '+str(outputs['prediction'][0])
            # print("-> " + prediction_text)

            fig = plt.figure(figsize=(10, 6))

            # Adds subplot on position 1
            ax = fig.add_subplot(121)
            # Adds subplot on position 2
            ax2 = fig.add_subplot(122)

            ax.scatter(outputs['X'], outputs['y'], alpha=0.3)
            ax.plot(outputs['X'], outputs['y_'], color="red", linewidth=4)

            ax.set_title(file_path[:-4])
            ax.set_xlabel(outputs['col'][0])
            ax.set_ylabel(outputs['col'][1])
            ax.set_xlim(0, outputs['min_max'][1])
            ax.set_ylim(0, outputs['min_max'][3])

            ax2.plot(outputs['X'], outputs['y_'], color="red", linewidth=4)
            ax2.scatter(xnew, outputs['prediction'][0], marker='d', s=400, color="green")
            ax2.set_title(file_path[:-4])
            ax2.set_xlabel(outputs['col'][0])
            ax2.set_ylabel(outputs['col'][1])
            ax2.set_xlim(0, outputs['min_max'][1])
            ax2.set_ylim(0, outputs['min_max'][3])

            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
            plt.savefig(image_path)
            # image_path = "./"+image_path
            image_path = image_path.replace('app', '')
            return render_template("forecasting.html", file=file_path, image=image_path, prediction_text=prediction_text)
    except Exception as e:
        LOG.exception("Exception occurred exc: ")
        LOG.error(f"Exception occurred err:{str(e)}")


# if __name__ == "__main__":
#     app.run()
