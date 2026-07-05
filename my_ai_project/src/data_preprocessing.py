import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm # لإظهار شريط التقدم
from sklearn.model_selection import train_test_split

# --- 1. تعريف المسارات ---
# المسار الحالي لملف data_preprocessing.py (داخل src)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# العودة مجلد واحد للوراء للوصول لجذر المشروع (my_ai_project)
PROJECT_ROOT = os.path.join(BASE_DIR, '..')

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'processed_labels.csv')

# التأكد من وجود مجلد البيانات المعالجة
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- 2. تحميل كاشف الوجه (Haar Cascade) ---
# ستحتاجون إلى تحميل ملف 'haarcascade_frontalface_default.xml'
# يجب أن يكون هذا الملف في نفس مجلد 'src' أو في مسار يمكن الوصول إليه
HAARCASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml') # المسار داخل مجلد src
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)

if face_cascade.empty():
    print(f"Error: Haar Cascade XML file not found at {HAARCASCADE_PATH}")
    print("Please ensure 'haarcascade_frontalface_default.xml' is correctly placed in the 'src' folder.")
    exit() # الخروج إذا لم يتم تحميل الكاشف

# --- 3. تهيئة قائمة لتخزين البيانات المعالجة ---
processed_data_list = []

# --- 4. معالجة الصور ---
print(f"Starting image preprocessing from: {RAW_DATA_DIR}")
image_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

for filename in tqdm(image_files, desc="Processing Images"):
    img_path = os.path.join(RAW_DATA_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        # print(f"Warning: Could not load image {filename}. Skipping.")
        continue # تخطي الصور التي لا يمكن تحميلها

    # تحويل الصورة إلى تدرج رمادي لكشف الوجه
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # كشف الوجوه في الصورة
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # print(f"No face detected in {filename}. Skipping.")
        continue # تخطي الصور التي لا تحتوي على وجوه

    # نأخذ أول وجه يتم الكشف عنه فقط (غالباً ما يكون هو الوجه الرئيسي)
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w] # قص الوجه من الصورة الأصلية

    # تغيير حجم الوجه وتطبيعه
    IMG_SIZE = (128, 128) # حجم الصورة الموحد للمدخلات النموذج
    face_resized = cv2.resize(face_img, IMG_SIZE)
    face_normalized = face_resized / 255.0 # تطبيع قيم البكسل بين 0 و 1

    # --- 5. استخراج العمر والجنس (خاص بـ UTKFace) ---
    # تنسيق اسم الملف في UTKFace: [age]_[gender]_[race]_[date&time].jpg
    # مثال: 23_0_0_20170113000000000.jpg (عمر 23، جنس 0 = ذكر، عرق 0 = أبيض)
    parts = filename.split('_')
    if len(parts) >= 2:
        try:
            age = int(parts[0])
            gender = int(parts[1]) # 0 for male, 1 for female
        except ValueError:
            # print(f"Could not parse age/gender from filename {filename}. Skipping.")
            continue # تخطي الملفات ذات الأسماء غير الصالحة
    else:
        # print(f"Filename {filename} does not match expected format. Skipping.")
        continue # تخطي الملفات التي لا تتوافق مع التنسيق

    # --- 6. حفظ الصورة المعالجة وتصنيفاتها ---
    # نستخدم اسم ملف فريد للمعالج لكي لا يتكرر في مجلد processed
    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(PROCESSED_DATA_DIR, processed_filename)
    # حفظ الصورة المعالجة. يجب تحويلها مرة أخرى إلى 0-255 و uint8 قبل الحفظ.
    cv2.imwrite(processed_filepath, (face_normalized * 255).astype(np.uint8))

    processed_data_list.append({
        'image_path': processed_filepath,
        'age': age,
        'gender': gender # 0 for male, 1 for female
    })

# --- 7. حفظ التصنيفات في ملف CSV ---
df = pd.DataFrame(processed_data_list)
df.to_csv(PROCESSED_CSV_PATH, index=False)
print(f"\nPreprocessing complete! Processed {len(df)} images.")
print(f"Labels saved to: {PROCESSED_CSV_PATH}")

# --- 8. تقسيم البيانات (تدريب، تحقق، اختبار) ---
# تعيين بذور عشوائية لضمان تكرار النتائج في كل مرة يتم تشغيل الكود
np.random.seed(42)

# تقسيم البيانات إلى تدريب + تحقق (80%) واختبار (20%)
# stratify=df['gender'] يضمن أن نسبة الذكور والإناث متساوية تقريباً في كل مجموعة
train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['gender'])

# تقسيم مجموعة التدريب + التحقق إلى تدريب (80%) وتحقق (20%) من تلك المجموعة
# (0.25 * 0.80 = 0.20 من البيانات الأصلية، لتكون نسبة التحقق 20%)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df['gender'])

print(f"\nData Split Summary:")
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# حفظ مسارات مجموعات البيانات المقسمة إلى ملفات CSV منفصلة
train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv'), index=False)
val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'val_labels.csv'), index=False)
test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv'), index=False)
print("Train, Validation, Test splits saved to CSV files.")