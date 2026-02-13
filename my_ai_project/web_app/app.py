import os  # مكتبة OS: للتعامل مع مسارات الملفات والمجلدات
from flask import Flask, request, jsonify, send_from_directory, render_template  # مكتبة Flask: لإعداد خادم الويب
from flask_cors import CORS  # مكتبة Flask-CORS: لحل مشاكل الأمان بين المتصفح والخادم
import tensorflow as tf  # مكتبة TensorFlow: لتحميل واستخدام نموذج الذكاء الاصطناعي
import numpy as np  # مكتبة NumPy: للتعامل مع الأرقام والمصفوفات (خاصة أرقام الصور)
import cv2  # مكتبة OpenCV: لمعالجة الصور (مثل تغيير حجمها)
import pandas as pd  # مكتبة Pandas: لقراءة ملفات CSV
from sklearn.preprocessing import MinMaxScaler  # أداة لتحجيم القيم (لأعمار)
import pickle  # مكتبة Pickle: لحفظ وتحميل كائنات بايثون (غير مستخدمة بشكل مباشر هنا لكنها مكتبة شائعة لتخزين الـ scalers)

# طباعة المسار الحالي الذي يعمل منه التطبيق ومسار المجلد الثابت
print(f"DEBUG: Current working directory: {os.getcwd()}")  # يطبع مجلد العمل الحالي للبرنامج
print(f"DEBUG: Static folder path: {os.path.abspath('static')}")  # يطبع المسار المطلق لمجلد 'static' (حيث توجد ملفات الويب)

app = Flask(__name__, static_folder='static')  # ينشئ تطبيق Flask ويحدد أن مجلد 'static' سيحتوي على الملفات الثابتة (HTML, CSS, JS)
CORS(app)  # تمكين CORS لتجنب مشاكل الأمان بين الواجهة الأمامية والخلفية (يسمح للمتصفح بالاتصال من أي مكان)

# --- تحميل النموذج ومُحجِّم العمر (مرة واحدة عند بدء تشغيل الخادم) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # يحصل على المسار المطلق للمجلد الحالي (web_app)
PROJECT_ROOT = os.path.join(BASE_DIR, '..')  # يعود مجلد واحد للخلف للوصول إلى جذر المشروع (my_ai_project)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')  # يحدد مسار مجلد النماذج المدربة
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')  # يحدد مسار مجلد البيانات المعالجة

# اسم النموذج الجديد الذي تم تدريبه
MODEL_NAME = 'best_transfer_learning_vgg16_model.keras'  # اسم ملف النموذج الذي تم تدريبه في الكود السابق
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)  # المسار الكامل لملف النموذج
TRAIN_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv')  # مسار ملف CSV لبيانات التدريب (لإعادة تهيئة الـ scaler)

try:  # يحاول تحميل النموذج والـ scaler
    model = tf.keras.models.load_model(MODEL_PATH)  # يُحمّل نموذج الذكاء الاصطناعي المدرب
    print(f"Backend: Model '{MODEL_NAME}' loaded successfully from {MODEL_PATH}!")  # رسالة تأكيد بتحميل النموذج

    train_df = pd.read_csv(TRAIN_CSV_PATH)  # يقرأ بيانات التدريب من ملف CSV (لنحتاجها لتهيئة الـ scaler)
    age_scaler = MinMaxScaler(feature_range=(0, 1))  # ينشئ مُحجِّم (scaler) جديد لقيم العمر
    age_scaler.fit(train_df[['age']])  # يُعلِّم الـ scaler على أعمار بيانات التدريب (نفس الخطوة التي تمت أثناء التدريب)
    print("Backend: Age scaler fitted successfully!")  # رسالة تأكيد بتهيئة الـ scaler

except Exception as e:  # إذا حدث أي خطأ أثناء التحميل
    print(f"Backend: Error loading model or scaler: {e}")  # يطبع رسالة الخطأ
    print(f"Backend: Expected model path: {MODEL_PATH}")  # يطبع المسار المتوقع للمساعدة في التصحيح
    exit()  # ينهي تشغيل التطبيق إذا فشل تحميل النموذج أو الـ scaler (لا يمكن للتطبيق أن يعمل بدونها)

IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128  # يحدد الأبعاد التي يتوقعها النموذج للصور (يجب أن تكون نفس الأبعاد المستخدمة في التدريب)

# --- نقطة نهاية لتقديم ملف index.html ---
@app.route('/')  # يُحدد أن هذه الدالة ستُشغل عندما يزور المستخدم المسار الرئيسي للموقع (مثلاً: localhost:5000/)
def serve_index():
    # بما أن static_folder='static'، فإن Flask يبحث عن index.html داخل مجلد 'static'
    return send_from_directory(app.static_folder, 'index.html')  # يرسل ملف index.html من مجلد 'static' إلى المتصفح

# --- نقطة نهاية لتقديم الملفات الثابتة (مثل CSS و JS) ---
# هذه الدالة ضرورية لكي يتمكن المتصفح من الوصول لـ style.css و أي ملفات أخرى في مجلد static
@app.route('/<path:filename>')  # يُحدد أن هذه الدالة ستُشغل لأي ملفات ثابتة أخرى في مجلد 'static' (مثل CSS أو JavaScript)
def static_files(filename):
    return send_from_directory(app.static_folder, filename)  # يرسل الملف المطلوب من مجلد 'static'

# --- نقطة نهاية للتنبؤ ---
@app.route('/predict', methods=['POST'])  # يُحدد أن هذه الدالة ستُشغل عندما يتم إرسال طلب 'POST' إلى المسار '/predict'
def predict_age_gender():
    if 'image' not in request.files:  # يتحقق مما إذا كانت الصورة موجودة في الطلب المرسل
        return jsonify({'error': 'No image file provided'}), 400  # يُرجع خطأ إذا لم يتم توفير صورة

    file = request.files['image']  # يحصل على ملف الصورة من الطلب
    if file.filename == '':  # يتحقق مما إذا كان اسم الملف فارغًا
        return jsonify({'error': 'No selected image file'}), 400  # يُرجع خطأ إذا لم يتم اختيار ملف

    try:  # يحاول معالجة الصورة وإجراء التنبؤ
        # قراءة الصورة
        image_np = np.frombuffer(file.read(), np.uint8)  # يقرأ بيانات الصورة الخام كـ NumPy array
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)  # يُحوّل بيانات الصورة إلى مصفوفة صور OpenCV (ملونة)

        if image is None:  # يتحقق مما إذا تم فك ترميز الصورة بنجاح
            return jsonify({'error': 'Could not decode image'}), 400  # يُرجع خطأ إذا لم يتمكن من فك ترميز الصورة

        # معالجة الصورة بنفس طريقة التدريب
        image_resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))  # يُغيّر حجم الصورة لتناسب حجم مدخلات النموذج
        image_normalized = image_resized / 255.0  # يقوم بتطبيع قيم البكسل (من 0-255 إلى 0-1)
        image_batch = np.expand_dims(image_normalized, axis=0)  # يضيف بعدًا إضافيًا للصورة لتصبح جاهزة لمدخلات النموذج (دفعة واحدة)

        # إجراء التنبؤ
        predictions = model.predict(image_batch)  # يُجري النموذج التنبؤ على الصورة المعالجة

        predicted_age_scaled = predictions[0][0][0]  # يستخرج قيمة العمر المُتوقعة (المُحجّمة) من مخرجات النموذج
        predicted_gender_raw = predictions[1][0][0]  # يستخرج قيمة الجنس المُتوقعة (الخام) من مخرجات النموذج

        # عكس تحجيم العمر
        predicted_age_original = age_scaler.inverse_transform(np.array([[predicted_age_scaled]]))[0][0]  # يُعيد تحجيم العمر المتوقع إلى قيمته الأصلية

        # تفسير الجنس
        gender_label = "Female" if predicted_gender_raw > 0.5 else "Male"  # يُحدد تسمية الجنس بناءً على التوقع (أكثر من 0.5 = أنثى، وإلا = ذكر)
        gender_confidence = predicted_gender_raw if gender_label == "Female" else (1 - predicted_gender_raw)  # يحسب "ثقة" النموذج في توقع الجنس

        return jsonify({  # يُرجع النتائج إلى المتصفح بصيغة JSON
            'age': int(predicted_age_original),  # العمر المتوقع (كرقم صحيح)
            'gender': gender_label,  # تسمية الجنس ("Male" أو "Female")
            'gender_confidence': float(gender_confidence)  # الثقة في توقع الجنس (كرقم عشري)
        })

    except Exception as e:  # إذا حدث أي خطأ أثناء عملية التنبؤ
        return jsonify({'error': str(e)}), 500  # يُرجع رسالة خطأ مع رمز حالة 500 (خطأ داخلي في الخادم)

if __name__ == '__main__':  # هذا الشرط يعني أن الكود التالي سيُشغل فقط عندما يتم تشغيل هذا الملف مباشرة
    print("Starting Flask server...")  # يطبع رسالة بداية تشغيل الخادم
    app.run(debug=True, port=5000)  # يُشغّل خادم Flask. `debug=True` يُفعّل وضع التصحيح (مفيد أثناء التطوير)، `port=5000` يعني أنه سيعمل على المنفذ 5000