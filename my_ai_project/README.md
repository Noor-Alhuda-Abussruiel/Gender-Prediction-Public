# Gender & Age Prediction System (End-to-End AI Project)

##Project Overview
An integrated AI system designed to predict gender and age from images. The project covers the entire pipeline from raw data preprocessing using OpenCV, model training, and finally deployment through a Web Application. It utilizes the UTKFace dataset and employs Haar Cascades for real-time face detection.

# Key Components
Automated face detection, cropping, and normalization (128x128).

Smart data splitting (Train/Val/Test) with gender balance.

المعالجة: كشف آلي للوجوه، قصها، وتوحيد المقاسات مع تقسيم البيانات بشكل متوازن.

Web Application (web_app/):

User-friendly interface to upload images and display results instantly.

تطبيق الويب: واجهة سهلة تسمح للمستخدم برفع الصور وعرض النتائج فوراً.

Model Management (models/):

Pre-trained models ready for inference (Keras/TensorFlow).

# Tech Stack

Libraries: OpenCV, TensorFlow/Keras, Pandas, NumPy, Scikit-learn.

Web: HTML5, CSS3, Flask (or your specific framework).

Tools: tqdm (Progress bars), Git (Version control).



#  Gender & Age Prediction from Images (Preprocessing Pipeline)

## هذا المشروع هو الجزء الأول من نظام ذكاء اصطناعي للتنبؤ بالجنس والعمر باستخدام تقنيات رؤية الكمبيوتر (Computer Vision). 

##  المميزات
* **كشف الوجوه الآلي:** استخدام Haar Cascades لتحديد الوجوه وقصها.
* **تنظيف البيانات:** معالجة الصور المعتمدة على مجموعة بيانات UTKFace.
* **هيكلة البيانات:** تقسيم آلي للبيانات إلى (Training, Validation, Test) مع الحفاظ على التوازن (Stratification).

##  التقنيات المستخدمة
- **Python**
- **OpenCV**: لمعالجة الصور.
- **Pandas & NumPy**: لإدارة البيانات الجدولية والمصفوفات.
- **Scikit-learn**: لتقسيم مجموعات البيانات باحترافية.

## طريقة الاستخدام
1. ضع الصور الخام في مجلد `data/raw/` بتنسيق UTKFace (`age_gender_race_date.jpg`).
2. قم بتشغيل الكود المعالج:
   ```bash
   python src/data_preprocessing.py