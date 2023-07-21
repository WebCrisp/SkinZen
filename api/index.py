from flask import Flask
from flask import send_from_directory
from flask import request
import json
import get_patches
import model
from PIL import Image
import numpy as np
import os
import shutil
import openai
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)


@app.route('/temp/<path:path>')
def send_report(path):
    return send_from_directory('temp', path)


def convert_image(file_path):
    img = Image.open(file_path)
    img_re = img.resize((128, 128))
    numpy_arr = np.asarray(img_re)
    return numpy_arr


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        def extract_patches(imageURL):
            patch_path = 'patches'
            dimension_dict = dict()
            face_dict = dict()
            image_dim = []
            try:
                dim, face, img = get_patches.extract_patches(
                    imageURL, dimension_dict, face_dict, image_dim, patch_path)
                print("extract patches pass")
                return dim, face, img
            except:
                print("extract patches fail")
                return None, None, None

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image = request.files['file']
        imagePath = os.path.join(BASE_DIR, f"temp/{image.filename}")
        image.save(imagePath)

        imageUrl = f"http://localhost:5000/temp/{image.filename}"

        os.mkdir(os.path.join(BASE_DIR, 'patches'))

        dim, face, img = extract_patches(imageUrl)

        if dim is None and face is None and img is None:
            return json.dumps({"msg": "fail"})

        model_path = os.path.join(
            BASE_DIR, 'models/Acne_Classifyer_N_Resnet.h5')

        resnet_model = model.load_trained_model(model_path)
        imageFiles = [os.path.join(BASE_DIR, 'patches', f)
                      for f in os.listdir(os.path.join(BASE_DIR, 'patches'))]

        images = []
        landmarks = []
        for f in imageFiles:
            if f.endswith('.jpg'):
                landmark = f.split('/')[-1].split('_')[-1]
                landmarks.append(landmark)
                images.append(convert_image(f))

        images = np.array(images)

        prediction = resnet_model.predict(images)
        prediction = prediction.tolist()
        shutil.rmtree(os.path.join(BASE_DIR, 'patches'))
        return json.dumps({"msg": "success", "prediction": prediction, "landmarks": landmarks})


@app.route('/recommendation', methods=['POST'])
def recommendation():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.get_json()

        if json.get('msg') == "success":
            prediction = json.get('prediction')
            prediction = np.array(prediction)
            prediction = prediction.argmax(axis=1)

            landmarks = json.get('landmarks')

            output = {landmarks[i].split(
                '.')[0]: f"level{prediction[i]}" for i in range(len(landmarks))}
            output['age'] = json.get('age')
            output['gender'] = json.get('gender')
            output['skin_type'] = json.get('skin_type')
            output['allergies'] = json.get('allergies')
            output['diet'] = json.get('diet')
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "you are a medical AI assistant that assists patients with acne on face with getting rid of it, okay? don't give me tips just nod"},

                    {"role": "user", "content": "what do you know about acne severity levels like level 0, level 1 and level 2 ?"},

                    {"role": "assistant", "content": """Acne severity levels are used to classify the severity of acne, with level 0 being the mildest and level 4 being the most severe. However, the commonly used system usually consists of four levels, which are described as follows:
                    Level 0: This level is used to describe skin with no signs of acne. The skin is generally clear with no visible pimples or blemishes.
                    Level 1: This level is used to describe skin with minor acne. The skin may have some blackheads and whiteheads, but they are not widespread and there are no visible pimples.
                    Level 2: This level is used to describe skin with moderate acne. The skin may have more blackheads and whiteheads, as well as a few visible pimples.
                    Level 3: This level is used to describe skin with severe acne. The skin may have many blackheads, whiteheads, and visible pimples, as well as some nodules and cysts.
                    It is worth noting that acne severity can vary greatly from person to person, and there may be other factors to consider when assessing the severity of acne, such as the presence of scarring or inflammation. Therefore, it is important to consult with a dermatologist or healthcare professional for an accurate diagnosis and personalized treatment plan."""},

                    {"role": "user", "content": "okay let me teach u something new"},

                    {"role": "assistant", "content": "Sure, I'm always open to learning something new! What would you like to teach me?"},

                    {"role": "user", "content": """so level 0 is This level is used to describe skin with no signs of acne. The skin is generally clear with no visible pimples or blemishes.
                    level 1 This level is used to describe skin with minor acne. The skin may have some blackheads and whiteheads, but they are not widespread and there are no visible pimples.
                    level 2 This level is used to describe skin with moderate acne to severe acne. The skin may have few to many blackheads and whiteheads, as well as a few or many visible pimples, as well as some nodules and cysts. there does not exist any other level these 3 are the only levels you know about okay"""},

                    {"role": "assistant", "content": """I apologize for the error in my previous response. You are correct that the commonly used acne severity classification system consists of three levels, not four. Here is the correct information:
                    Level 0: This level is used to describe skin with no signs of acne. The skin is generally clear with no visible pimples or blemishes.
                    Level 1: This level is used to describe skin with mild acne. The skin may have some blackheads and whiteheads, but they are not widespread and there are no visible pimples.
                    Level 2: This level is used to describe skin with moderate to severe acne. The skin may have many blackheads and whiteheads, as well as a few or many visible pimples, as well as some nodules and cysts.
                    Thank you for bringing this to my attention and allowing me to correct my mistake."""},

                    {"role": "user", "content": "okay now i will tell you some instructions and the responses with boudries to which you will adhere"},

                    {"role": "assistant", "content": "Sure, please go ahead and give me the instructions and boundaries to adhere to."},

                    {"role": "user", "content": """so we have divided a face into patches these are namely fh(forehead), lc(left cheek), rc(right cheek), ch(chin). Each time my system will analyze the face image extract the patches from it and will send them to you along with some other info of the user which are Age, Gender(Male, Female, other), Skin-Type(Oily, Dry, Normal), Allergies(salicylic acid, benzoyl peroxide, Sun-exposure, Itching, Swelling, Redness), Diet(Veg, Non-veg) these are a total of 5 meta-data values about the person or individual for whom you will generate the Diet, Daily Skin-care Routine,  it is not necessary that each time you will be provided with all the 4 mentioned patches above(but you will always be provided with the 5 meta-data values of the individual)  some time the message may contain only 1 or 2 patches. I will also send the acne severity level associated with that patch like {'fh':'level0', 'lc':'level2', 'rc':'level2', 'chin':'level0', 'age':20, 'gender':'male', 'skin':'oily', 'allergies':['itching'], 'diet':'veg'} this json data means that the patch forehead has level0 acne and left cheek has level2 acne and he is of age 20 and he has oily skin and have itching allergy and his diet is veg. Now that you now what level of acne does the parts of a particular face has you will generate a week long custom made remedy, diet, good habits, bad habits and daily skincare routine for the acne treatment. I can ask you to generate a general acne routine and remedy but i am providing you with this classified details so that the output you will generate will be more optimized and personalized. okay that's it if you understand completely what i just instructed to you then generate a sample response for me for the above example data. also next time i will not be sending such long detailed message just the json message that i sent above so you will not get confused and do the task as i told you. also the response should be in a raw mark up language format the language should be raw html with <br> tag at places where line break is necessary so i can copy it as it is and use it some where else. Also i need you to generate answers in way that they don't sound like they are being generated by a chat bot, leave out any sentences like "Sure, here's a customized routine for the given example:" so you need to sound like the answers are being generated by a deep learning NLP model not by a chat bot. You don't need to greet to me in the response just straight out tell the answer don't greet to me like 'here is the routine for the example' or anything like that."""},

                    {"role": "assistant", "content": """
<h3>Skincare Routine:</h3>
<ul>
  <li>Wash your face with a gentle cleanser twice a day.</li>
  <li>Use an alcohol-free toner after cleansing to balance your skin.</li>
  <li>Apply a moisturizer with SPF in the morning.</li>
  <li>Apply a retinoid cream at night to help unclog pores.</li>
  <li>Use a clay mask once a week to draw out impurities.</li>
</ul>
<h3>Diet:</h3>
<p>Since the individual has oily skin, it's recommended to follow a low-glycemic index diet which includes more fruits, vegetables, whole grains, and lean proteins. Also, since they have a vegetarian diet, they can consume more nuts, seeds, and legumes as sources of protein. It's also important to drink plenty of water and avoid sugary and processed foods as they can cause inflammation in the body and lead to acne breakouts.</p>
<h3>Good Habits:</h3>
<ul>
  <li>Keep your hands off your face to prevent bacteria transfer.</li>
  <li>Change your pillowcases regularly to avoid buildup of oils and bacteria.</li>
  <li>Avoid touching or picking at acne as it can lead to scarring.</li>
  <li>Wash your makeup brushes regularly to avoid buildup of bacteria.</li>
  <li>Wear sunscreen every day to protect your skin from harmful UV rays.</li>
</ul>
<h3>Bad Habits:</h3>
<ul>
  <li>Avoid smoking as it can cause premature aging and damage to the skin.</li>
  <li>Avoid excessive alcohol consumption as it can dehydrate the skin.</li>
  <li>Avoid using harsh scrubs or exfoliants as they can irritate the skin.</li>
  <li>Avoid using products with salicylic acid or benzoyl peroxide if the individual has an allergy to them.</li>
  <li>Avoid consuming processed and sugary foods as they can cause inflammation and lead to acne breakouts.</li>
</ul>
<h3>Additional Recommendations:</h3>
<p>Since the individual has an itching allergy, it's recommended to avoid products with fragrance and other potential irritants. Also, since they have oily skin, it's recommended to use non-comedogenic and oil-free products to avoid clogging pores.</p>"""},

                    {"role": "user", "content": "{'fh':'level2', 'lc':'level1', 'rc':'level0', 'chin':'level2', 'age':29, 'gender':'female', 'skin':'dry', 'allergies':['itching', 'swelling', 'sun-exposure', 'redness'], 'diet':'non-veg'"},

                    {"role": "assistant", "content": """
                    <h3>Week-long Acne Treatment Plan</h3>
<p>Age: 29 | Gender: Female | Skin Type: Dry | Allergies: Itching,Swelling,Sun-exposure,Redness | Diet: Non-Veg</p>
<h4>Daily Skincare Routine:</h4>
<ul>
  <li>Wash your face twice a day using a gentle, non-comedogenic cleanser.</li>
  <li>Apply a moisturizer with SPF 30 or higher to protect your skin from sun damage.</li>
  <li>Use a toner containing salicylic acid to help unclog pores and reduce acne.</li>
  <li>Apply a spot treatment containing benzoyl peroxide to acne-prone areas.</li>
</ul>
<h4>Diet Recommendations:</h4>
<ul>
  <li>Include foods rich in vitamin A, such as sweet potatoes, carrots, and leafy greens, which can help reduce inflammation and promote healthy skin.</li>
  <li>Avoid dairy products and high-glycemic foods, such as white bread and sugary snacks, which may worsen acne.</li>
  <li>Incorporate foods rich in omega-3 fatty acids, such as salmon and walnuts, which can help reduce inflammation and improve skin health.</li>
</ul>
<h4>Good Habits:</h4>
<ul>
  <li>Get enough sleep each night to help reduce stress, which can trigger acne.</li>
  <li>Drink plenty of water to stay hydrated and flush out toxins.</li>
  <li>Exercise regularly to help reduce stress and improve overall health.</li>
</ul>
<h4>Bad Habits to Avoid:</h4>
<ul>
  <li>Avoid touching your face, as this can transfer oil and bacteria to the skin and worsen acne.</li>
  <li>Avoid using harsh scrubs or exfoliants, which can irritate the skin and worsen acne.</li>
  <li>Avoid picking or squeezing pimples, as this can lead to scarring and further breakouts.</li>
</ul>
<h4>Acne Treatment:</h4>
<p>Based on the severity of acne on different patches of the face:</p>
<ul>
  <li>For level 0 acne on fh, chin: continue with daily skincare routine and good habits.</li>
  <li>For level 1 acne on lc: use an over-the-counter retinoid cream in the evening to help unclog pores and reduce acne.</li>
  <li>For level 2 acne on fh, rc, chin: consult a dermatologist for prescription-strength topical or oral medications.</li>
</ul>"""},
                    {"role": "user", "content": f"{output}"},
                ]
            )

        return completion


if __name__ == '__main__':
    app.run()
