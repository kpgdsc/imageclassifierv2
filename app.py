import os,  time
from flask import Flask, render_template, request, redirect, url_for, session, flash
import shutil
from forms import ClassifyForm
import re
from imgclassifywrapper import  Imgclassifywrapper
from  predict   import predictint, imageprepare



app = Flask(__name__)
app.config['SECRET_KEY'] = "Your_secret_string"


UPLOAD_FOLDER = os.path.basename('static')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
NUMBER_IMAGE_FILE = 'static/output'
NUMBER_IMAGE_FILE_FINAL = None


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/image')
def image():
    return render_template('selectimage.html')



@app.route('/upload', methods=['POST'])
def upload_file():

    if request.files == None :
        return render_template('selectimage.html')


    file = request.files['image']
    print('type : ' + str(type(file)))


    if file.filename == None or len(file.filename) ==0 :
        return render_template('selectimage.html')

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if  allowed_file(file.filename) == False:
        flash('Only PNG and JPG filea are allowed!')
        return render_template('selectimage.html')

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    print('file.filename : ' + file.filename + ' full_filename : ' + full_filename)
    #archive_file(file.filename)

    # add your custom code to check that the uploaded file is a valid image
    file.save(full_filename)
    file.close()
    session['image_file'] = full_filename
    return render_template('classify.html', image_filename=full_filename)

'''
    os.chmod(full_filename, 0o777)
    st = os.stat(full_filename)
    os.chmod(full_filename, st.st_mode | stat.S_IWOTH)

    st = os.stat('uploads')
    os.chmod('uploads', st.st_mode | stat.S_IWOTH)
    #os.chmod('upload', stat.st_mode | stat.S_IWOTH)

    print('\n os.access(full_filename, os.X_OK) ')
    print ( os.access(full_filename, os.X_OK) ) # Check for read access
'''
    #full_filename = 'static/car.jpg'



@app.route('/classify', methods=['GET', 'POST'])
def classify():

    form = ClassifyForm()
    if form.validate_on_submit():
        return redirect(url_for('interpret'))

    return render_template('index.html')

@app.route('/interpret', methods=['GET', 'POST'])
def interpret():
    img_cls_wrapper = Imgclassifywrapper()

    image = session['image_file']
    airesult = 'Testing'
    airesult = img_cls_wrapper.run_inference_on_image(image)
    return render_template('ai.html', image_filename=session['image_file'], airesult= airesult)

@app.route('/interpret2', methods=['GET', 'POST'])
def interpret2():
    img_cls_wrapper = Imgclassifywrapper()

    image = session['image_file']
    airesult = 'Testing'
    airesult = img_cls_wrapper.run_inference_on_image(image)
    return airesult

@app.route('/prg', methods=['GET'])
def prg():
    img_cls_wrapper = Imgclassifywrapper()

    image = session['image_file']
    airesult = "testing"
    return render_template('aiwithprg.html', image_filename=session['image_file'], airesult= airesult)


@app.route('/number', methods=['GET'])
def number():

    return render_template('numdraw.html')




@app.route('/canvas', methods=['POST'])
def save_canvas():
    global NUMBER_IMAGE_FILE_FINAL
    dict = request.form
    #for key in dict:
    #    print 'form key :  '  + str(key) + " : "+ dict[key]
    print('\n\n saving canvas.......')
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')

    #imgData = request.values['imageData'];     # 'imageData' imageData


    #imgData = request.values['imageData'];

    #file = request.files['imageData']
    fname = NUMBER_IMAGE_FILE
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M-%s', t)


    fname = fname + str(timestamp) + '.png'
    NUMBER_IMAGE_FILE_FINAL = fname


    if os.path.exists(fname):
        os.remove(fname)

    f = open(fname,"wb")
    f.write(image_data)
    f.close()


    print('\n\n END saving canvas.......')

@app.route('/num_prediction', methods=['Get'])
def num_prediction():
    image_filename = NUMBER_IMAGE_FILE_FINAL

    imvalue = imageprepare(image_filename)
    print('0')
    predint_global = predictint(imvalue)


    airesult = predint_global[0]

    return render_template('number.html', image_filename=image_filename, airesult=airesult)

def archive_file(source_filename):
    # adding exception handling

    file_name = os.path.basename(source_filename)
    source_path = os.path.join('uploads', file_name)

    target = os.path.join('uploads', 'backup')
    target = os.path.join(target, file_name)


    try:
        shutil.copy(source_path, target)
        os.remove(source_path)
    except IOError as e:
        print("Unable to copy file. %s" % e)
    except:
        print("Unexpected error:", sys.exc_info())

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__' :
    app.run(debug=True)

