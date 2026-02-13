import cv2  # استيراد مكتبة OpenCV لمعالجة الصور وعمليات رؤية الكمبيوتر (مثل كشف الوجه)
import os  # استيراد مكتبة OS للتعامل مع مسارات الملفات والمجلدات في نظام التشغيل
import pandas as pd  # استيراد مكتبة Pandas للتعامل مع البيانات الجدولية (مثل ملفات CSV)
import numpy as np  # استيراد مكتبة NumPy للعمليات العددية والمصفوفات بكفاءة عالية
from tqdm import tqdm  # استيراد أداة tqdm لإظهار شريط التقدم أثناء الحلقات (لتحسين تجربة المستخدم)
from sklearn.model_selection import train_test_split  # استيراد أداة لتقسيم البيانات إلى مجموعات تدريب واختبار وتحقق

# --- 1. تعريف المسارات ---
# المسار الحالي لملف data_preprocessing.py (داخل src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # يحصل على المسار المطلق للمجلد الحالي الذي يحتوي هذا الملف
# العودة مجلد واحد للوراء للوصول لجذر المشروع (my_ai_project)
PROJECT_ROOT = os.path.join(BASE_DIR, '..')  # يحدد المسار إلى جذر المشروع (المجلد الأب للمجلد الحالي)

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')  # يحدد المسار إلى مجلد البيانات الخام (الصور الأصلية)
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')  # يحدد المسار إلى مجلد البيانات المعالجة (الصور التي سيتم قصها وتعديلها)
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_labels.csv')  # يحدد مسار ملف CSV الذي سيحتوي على تسميات البيانات المعالجة

# التأكد من وجود مجلد البيانات المعالجة
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # ينشئ مجلد 'processed' داخل 'data' إذا لم يكن موجودًا بالفعل

# --- 2. تحميل كاشف الوجه (Haar Cascade) ---
# ستحتاجون إلى تحميل ملف 'haarcascade_frontalface_default.xml'
# يجب أن يكون هذا الملف في نفس مجلد 'src' أو في مسار يمكن الوصول إليه
HAARCASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')  # يحدد المسار إلى ملف كاشف الوجه XML
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)  # يُحمّل كاشف الوجه باستخدام OpenCV

if face_cascade.empty():  # يتحقق مما إذا تم تحميل كاشف الوجه بنجاح
    print(f"Error: Haar Cascade XML file not found at {HAARCASCADE_PATH}")  # رسالة خطأ إذا لم يتم العثور على الملف
    print("Please ensure 'haarcascade_frontalface_default.xml' is correctly placed in the 'src' folder.")  # تعليمات للمستخدم
    exit()  # ينهي تشغيل البرنامج إذا لم يتم تحميل الكاشف

# --- 3. تهيئة قائمة لتخزين البيانات المعالجة ---
processed_data_list = []  # يُنشئ قائمة فارغة لتخزين معلومات الصور المعالجة (مسار، عمر، جنس)

# --- 4. معالجة الصور ---
print(f"Starting image preprocessing from: {RAW_DATA_DIR}")  # يطبع رسالة بداية المعالجة
image_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]  # يحصل على قائمة بجميع ملفات الصور في المجلد الخام

for filename in tqdm(image_files, desc="Processing Images"):  # يتكرر على كل ملف صورة مع شريط تقدم
    img_path = os.path.join(RAW_DATA_DIR, filename)  # ينشئ المسار الكامل للصورة الخام
    img = cv2.imread(img_path)  # يقرأ الصورة من المسار المحدد

    if img is None:  # يتحقق مما إذا تم تحميل الصورة بنجاح
        # print(f"Warning: Could not load image {filename}. Skipping.")  # رسالة تحذيرية إذا فشل التحميل (معلقة)
        continue  # يتخطى الصورة إذا لم يتمكن من تحميلها

    # تحويل الصورة إلى تدرج رمادي لكشف الوجه
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # يحول الصورة الملونة إلى تدرج رمادي (لأن كاشف الوجه يعمل بشكل أفضل عليه)
    # كشف الوجوه في الصورة
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # يستخدم كاشف الوجه للبحث عن الوجوه في الصورة الرمادية

    if len(faces) == 0:  # يتحقق مما إذا تم العثور على أي وجوه
        # print(f"No face detected in {filename}. Skipping.")  # رسالة إذا لم يتم الكشف عن وجه (معلقة)
        continue  # يتخطى الصورة إذا لم يتم العثور على وجوه

    # نأخذ أول وجه يتم الكشف عنه فقط (غالباً ما يكون هو الوجه الرئيسي)
    (x, y, w, h) = faces[0]  # يستخرج إحداثيات ومقاسات أول وجه تم الكشف عنه (x, y للزاوية العلوية اليسرى، w للعرض، h للارتفاع)
    face_img = img[y:y+h, x:x+w]  # يقص الوجه من الصورة الأصلية باستخدام الإحداثيات

    # تغيير حجم الوجه وتطبيعه
    IMG_SIZE = (128, 128)  # يحدد الحجم الموحد الذي ستكون عليه جميع صور الوجوه المعالجة
    face_resized = cv2.resize(face_img, IMG_SIZE)  # يغير حجم الوجه المقصوص إلى الحجم الموحد
    face_normalized = face_resized / 255.0  # يقوم بتطبيع قيم البكسل (يقسمها على 255) لتكون بين 0 و 1 (مهم للشبكات العصبية)

    # --- 5. استخراج العمر والجنس (خاص بـ UTKFace) ---
    # تنسيق اسم الملف في UTKFace: [age]_[gender]_[race]_[date&time].jpg
    # مثال: 23_0_0_20170113000000000.jpg (عمر 23، جنس 0 = ذكر، عرق 0 = أبيض)
    parts = filename.split('_')  # يقسم اسم الملف باستخدام الشرطة السفلية '_' كفاصل
    if len(parts) >= 2:  # يتحقق مما إذا كان هناك ما يكفي من الأجزاء لاستخراج العمر والجنس
        try:
            age = int(parts[0])  # يحاول تحويل الجزء الأول (العمر) إلى عدد صحيح
            gender = int(parts[1])  # يحاول تحويل الجزء الثاني (الجنس) إلى عدد صحيح (0 للذكر، 1 للأنثى)
        except ValueError:  # يلتقط أي خطأ إذا لم يتمكن من تحويل الأجزاء إلى أرقام
            # print(f"Could not parse age/gender from filename {filename}. Skipping.")  # رسالة خطأ (معلقة)
            continue  # يتخطى الملفات ذات الأسماء غير الصالحة
    else:
        # print(f"Filename {filename} does not match expected format. Skipping.")  # رسالة خطأ (معلقة)
        continue  # يتخطى الملفات التي لا تتوافق مع التنسيق المتوقع

    # --- 6. حفظ الصورة المعالجة وتصنيفاتها ---
    # نستخدم اسم ملف فريد للمعالج لكي لا يتكرر في مجلد processed
    processed_filename = f"processed_{filename}"  # ينشئ اسمًا جديدًا للصورة المعالجة (إضافة "processed_")
    processed_filepath = os.path.join(PROCESSED_DATA_DIR, processed_filename)  # ينشئ المسار الكامل للصورة المعالجة
    # حفظ الصورة المعالجة. يجب تحويلها مرة أخرى إلى 0-255 و uint8 قبل الحفظ.
    cv2.imwrite(processed_filepath, (face_normalized * 255).astype(np.uint8))  # يحفظ الصورة المعالجة بعد إعادة تحويلها إلى نطاق 0-255

    processed_data_list.append({  # يضيف معلومات الصورة المعالجة إلى القائمة
        'image_path': processed_filepath,  # مسار الصورة المعالجة
        'age': age,  # العمر المستخرج
        'gender': gender  # الجنس المستخرج
    })

# --- 7. حفظ التصنيفات في ملف CSV ---
df = pd.DataFrame(processed_data_list)  # يُحوّل القائمة إلى DataFrame (جدول بيانات)
df.to_csv(PROCESSED_CSV_PATH, index=False)  # يحفظ الـ DataFrame في ملف CSV، بدون حفظ فهرس الصفوف
print(f"\nPreprocessing complete! Processed {len(df)} images.")  # يطبع رسالة عند اكتمال المعالجة وعدد الصور المعالجة
print(f"Labels saved to: {PROCESSED_CSV_PATH}")  # يطبع مسار ملف CSV المحفوظ

# --- 8. تقسيم البيانات (تدريب، تحقق، اختبار) ---
# تعيين بذور عشوائية لضمان تكرار النتائج في كل مرة يتم تشغيل الكود
np.random.seed(42)  # يضبط بذرة الأرقام العشوائية لضمان أن تكون النتائج قابلة للتكرار

# تقسيم البيانات إلى تدريب + تحقق (80%) واختبار (20%)
# stratify=df['gender'] يضمن أن نسبة الذكور والإناث متساوية تقريباً في كل مجموعة
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['gender'])  # يقسم البيانات إلى 80% للتدريب والتحقق، و20% للاختبار، مع الحفاظ على توزيع الجنس

# تقسيم مجموعة التدريب + التحقق إلى تدريب (80%) وتحقق (20%) من تلك المجموعة
# (0.25 * 0.80 = 0.20 من البيانات الأصلية، لتكون نسبة التحقق 20%)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['gender'])  # يقسم مجموعة التدريب والتحقق إلى 80% تدريب و20% تحقق، مع الحفاظ على توزيع الجنس

print(f"\nData Split Summary:")  # يطبع ملخص لتقسيم البيانات
print(f"Training samples: {len(train_df)}")  # يطبع عدد عينات التدريب
print(f"Validation samples: {len(val_df)}")  # يطبع عدد عينات التحقق
print(f"Test samples: {len(test_df)}")  # يطبع عدد عينات الاختبار

# حفظ مسارات مجموعات البيانات المقسمة إلى ملفات CSV منفصلة
train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv'), index=False)  # يحفظ بيانات التدريب في ملف CSV منفصل
val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val_labels.csv'), index=False)  # يحفظ بيانات التحقق في ملف CSV منفصل
test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv'), index=False)  # يحفظ بيانات الاختبار في ملف CSV منفصل
print("Train, Validation, Test splits saved to CSV files.")  # رسالة تأكيد بحفظ الملفات