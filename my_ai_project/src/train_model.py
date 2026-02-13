import pandas as pd  # استيراد مكتبة Pandas للتعامل مع البيانات الجدولية (مثل ملفات CSV)
import numpy as np  # استيراد مكتبة NumPy للعمليات العددية والمصفوفات بكفاءة عالية
import os  # استيراد مكتبة OS للتعامل مع مسارات الملفات والمجلدات في نظام التشغيل
import cv2  # استيراد مكتبة OpenCV لمعالجة الصور وعمليات رؤية الكمبيوتر
import tensorflow as tf  # استيراد مكتبة TensorFlow الأساسية للتعلم العميق والذكاء الاصطناعي

# استيراد أدوات محددة من TensorFlow Keras لبناء النماذج ومعالجة البيانات
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # لإنشاء دفعات من الصور مع تضخيم البيانات
from tensorflow.keras.models import Sequential, Model  # لبناء نماذج الشبكات العصبية (Sequential للبسيط، Model للمعقد)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input  # أنواع مختلفة من طبقات الشبكات العصبية
from tensorflow.keras.optimizers import Adam  # مُحسِّن (Optimizer) لتحسين أوزان النموذج أثناء التدريب
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # وظائف مساعدة لعملية التدريب (Callbacks)
import matplotlib.pyplot as plt  # استيراد مكتبة Matplotlib لرسم الرسوم البيانية وتصوير البيانات
from sklearn.preprocessing import MinMaxScaler  # أداة لتحجيم البيانات (جعلها في نطاق معين، مثل 0-1)
from sklearn.metrics import accuracy_score, mean_absolute_error  # مقاييس لتقييم أداء النموذج (الدقة ومتوسط الخطأ المطلق)

# استيراد النموذج الأساسي لـ Transfer Learning
from tensorflow.keras.applications import VGG16  # استيراد نموذج VGG16 المُدرب مسبقًا للاستفادة من التعلم بالنقل


# --- 1. تعريف المسارات (مُعدلة لتناسب هيكل المشروع MY_AI_PROJECT/src) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # يحصل على المسار المطلق للمجلد الحالي (src)
PROJECT_ROOT = os.path.join(BASE_DIR, os.pardir)  # يعود مجلد واحد للخلف للوصول إلى جذر المشروع (MY_AI_PROJECT)

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')  # يحدد مسار مجلد البيانات المعالجة
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')  # يحدد مسار مجلد حفظ النماذج المدربة

# التأكد من وجود مجلد النماذج
os.makedirs(MODELS_DIR, exist_ok=True)  # ينشئ مجلد 'models' إذا لم يكن موجودًا بالفعل

TRAIN_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv')  # مسار ملف CSV لبيانات التدريب
VAL_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'val_labels.csv')  # مسار ملف CSV لبيانات التحقق
TEST_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv')  # مسار ملف CSV لبيانات الاختبار

# --- 2. تحميل البيانات ---
train_df = pd.read_csv(TRAIN_CSV_PATH)  # يقرأ بيانات التدريب من ملف CSV إلى DataFrame
val_df = pd.read_csv(VAL_CSV_PATH)  # يقرأ بيانات التحقق من ملف CSV
test_df = pd.read_csv(TEST_CSV_PATH)  # يقرأ بيانات الاختبار من ملف CSV

print("\nGender distribution in Test Data:")  # يطبع عنوانًا لتوزيع الجنس
print(test_df['gender'].value_counts())  # يعرض عدد الذكور والإناث في بيانات الاختبار
print("\n")  # يطبع سطرًا فارغًا

# تحويل مسارات الصور لتكون مطلقة ومناسبة لنظام التشغيل الحالي
def make_absolute_paths(df):  # تعريف دالة لتحويل المسارات النسبية إلى مطلقة
    # تفترض أن 'image_path' في الـ CSV هي مسارات نسبية من 'PROCESSED_DATA_DIR'
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(PROCESSED_DATA_DIR, os.path.basename(x)))  # يحول مسار كل صورة إلى مسار مطلق
    return df  # يعيد الـ DataFrame بالمسارات المطلقة

train_df = make_absolute_paths(train_df)  # يطبق الدالة على DataFrame التدريب
val_df = make_absolute_paths(val_df)  # يطبق الدالة على DataFrame التحقق
test_df = make_absolute_paths(test_df)  # يطبق الدالة على DataFrame الاختبار

print(f"Loaded {len(train_df)} training samples.")  # يطبع عدد عينات التدريب التي تم تحميلها
print(f"Loaded {len(val_df)} validation samples.")  # يطبع عدد عينات التحقق التي تم تحميلها
print(f"Loaded {len(test_df)} test samples.")  # يطبع عدد عينات الاختبار التي تم تحميلها

# --- تحجيم قيم العمر (Age Scaling) ---
age_scaler = MinMaxScaler(feature_range=(0, 1))  # ينشئ مُحجِّم (scaler) لتحويل القيم بين 0 و 1
train_df['age_scaled'] = age_scaler.fit_transform(train_df[['age']])  # يُعلِّم المُحجِّم على أعمار التدريب ثم يُحوِّلها
val_df['age_scaled'] = age_scaler.transform(val_df[['age']])  # يُحوِّل أعمار التحقق بناءً على ما تعلمه المُحجِّم
test_df['age_scaled'] = age_scaler.transform(test_df[['age']])  # يُحوِّل أعمار الاختبار بناءً على ما تعلمه المُحجِّم

# --- 3. تهيئة مُنشئ البيانات (ImageDataGenerator) ---
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128  # تعريف الأبعاد الموحدة للصور المدخلة للنموذج
BATCH_SIZE = 32  # تعريف حجم الدفعة (عدد الصور التي تعالج دفعة واحدة)

train_datagen = ImageDataGenerator(  # يُنشئ مُولد بيانات لصور التدريب
    rescale=1./255,  # يُعيد تحجيم قيم البكسل من 0-255 إلى 0-1 (ضروري للشبكات العصبية)
    rotation_range=10,  # يسمح بتدوير الصور عشوائيًا حتى 10 درجات
    width_shift_range=0.1,  # يسمح بتحريك الصور أفقياً بنسبة 10%
    height_shift_range=0.1,  # يسمح بتحريك الصور عمودياً بنسبة 10%
    shear_range=0.1,  # يسمح بإجراء تحولات القص (Shear transformations) بنسبة 10%
    zoom_range=0.1,  # يسمح بالتكبير أو التصغير العشوائي للصور بنسبة 10%
    horizontal_flip=True,  # يسمح بقلب الصور أفقياً عشوائياً (مهم للوجوه)
    fill_mode='nearest'  # يحدد كيف يتم ملء البكسلات الجديدة التي قد تظهر بعد التحويلات
)

val_test_datagen = ImageDataGenerator(rescale=1./255)  # يُنشئ مُولد بيانات لصور التحقق والاختبار (بدون تضخيم)

# تعريف output_signature لتحديد أنواع وأشكال البيانات المتوقعة من المولد (مهم لـ tf.data.Dataset)
output_signature = (  # يحدد شكل ونوع الصورة
    tf.TensorSpec(shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=tf.float32),
    {  # يحدد شكل ونوع مخرجات العمر والجنس
        'age_output': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        'gender_output': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    }
)

def generator_wrapper(keras_generator):  # دالة مُساعدة لتكييف مخرجات ImageDataGenerator مع TensorFlow Dataset
    for X_batch, y_batch_from_generator in keras_generator:  # تتكرر على الدفعات من مُولد Keras
        age_labels = y_batch_from_generator[0].astype(np.float32).reshape(-1, 1)  # تستخرج تسميات العمر وتضبط شكلها
        gender_labels = y_batch_from_generator[1].astype(np.float32).reshape(-1, 1)  # تستخرج تسميات الجنس وتضبط شكلها

        yield X_batch, {  # تُعيد الدفعة من الصور والتسميات كقاموس (لتناسب المخرجات المتعددة للنموذج)
            'age_output': age_labels,
            'gender_output': gender_labels
        }

train_dataset = tf.data.Dataset.from_generator(  # يُنشئ كائن TensorFlow Dataset لبيانات التدريب
    lambda: generator_wrapper(train_datagen.flow_from_dataframe(  # يستخدم مُولد Keras المغلّف
        dataframe=train_df,  # يستخدم DataFrame الخاص بالتدريب
        directory=PROCESSED_DATA_DIR,  # المجلد الأساسي للصور
        x_col='image_path',  # العمود الذي يحتوي على مسار الصورة
        y_col=['age_scaled', 'gender'],  # الأعمدة التي تحتوي على تسميات العمر والجنس
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),  # الحجم المستهدف للصور
        batch_size=BATCH_SIZE,  # حجم الدفعة
        class_mode='multi_output',  # يُحدد أن النموذج له مخرجات متعددة
        shuffle=True,  # يُخلط الصور في كل دورة تدريب
        seed=42  # يُحدد بذرة عشوائية لضمان تكرار النتائج
    )),
    output_signature=output_signature  # يُحدد التوقيع المتوقع للمخرجات
).prefetch(tf.data.AUTOTUNE)  # يُحسِّن أداء تحميل البيانات عن طريق تحميل الدفعات التالية مُسبقًا

val_dataset = tf.data.Dataset.from_generator(  # يُنشئ كائن TensorFlow Dataset لبيانات التحقق (نفس فكرة التدريب لكن بدون خلط)
    lambda: generator_wrapper(val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=PROCESSED_DATA_DIR,
        x_col='image_path',
        y_col=['age_scaled', 'gender'],
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='multi_output',
        shuffle=False,  # لا يتم خلط بيانات التحقق
        seed=42
    )),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(  # يُنشئ كائن TensorFlow Dataset لبيانات الاختبار (نفس فكرة التحقق)
    lambda: generator_wrapper(val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=PROCESSED_DATA_DIR,
        x_col='image_path',
        y_col=['age_scaled', 'gender'],
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='multi_output',
        shuffle=False,  # لا يتم خلط بيانات الاختبار
        seed=42
    )),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)


# --- 4. دالة لبناء النموذج الأساسي (الذي بنيتيه من الصفر) ---
def build_original_cnn_model(input_shape):  # تعريف دالة لبناء نموذج CNN من الصفر
    print("\nBuilding ORIGINAL CNN Model...")  # رسالة لإعلام المستخدم بنوع النموذج
    input_tensor = Input(shape=input_shape)  # طبقة الإدخال، تحدد شكل الصورة
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)  # طبقة تلافيفية أولى (32 فلتر، حجم 3x3)
    x = MaxPooling2D((2, 2))(x)  # طبقة تجميع لتقليل الأبعاد
    x = Dropout(0.3)(x)  # طبقة "تسرب" لتقليل الحفظ الزائد (overfitting)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # طبقة تلافيفية ثانية (64 فلتر)
    x = MaxPooling2D((2, 2))(x)  # طبقة تجميع
    x = Dropout(0.3)(x)  # طبقة تسرب

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)  # طبقة تلافيفية ثالثة (128 فلتر)
    x = MaxPooling2D((2, 2))(x)  # طبقة تجميع
    x = Dropout(0.3)(x)  # طبقة تسرب

    x = Flatten()(x)  # طبقة "تسطيح" تحول مصفوفة الأبعاد إلى متجه أحادي
    x = Dense(256, activation='relu')(x)  # طبقة كثيفة (Fully Connected) بـ 256 وحدة عصبية
    x = Dropout(0.6)(x)  # طبقة تسرب

    age_output = Dense(1, activation='relu', name='age_output')(x)  # طبقة مخرجات العمر (وحدة واحدة، دالة تفعيل Relu)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)  # طبقة مخرجات الجنس (وحدة واحدة، دالة تفعيل Sigmoid)

    model = Model(inputs=input_tensor, outputs=[age_output, gender_output])  # يُنشئ النموذج بالمدخلات والمخرجات المحددة
    return model  # يُعيد النموذج المُنشأ

# --- 5. دالة لبناء النموذج المحسن (باستخدام Transfer Learning VGG16) ---
def build_transfer_learning_vgg16_model(input_shape):  # تعريف دالة لبناء نموذج باستخدام VGG16 للتعلم بالنقل
    print("\nBuilding TRANSFER LEARNING (VGG16) Model...")  # رسالة لإعلام المستخدم بنوع النموذج
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)  # يحمّل نموذج VGG16 بدون الطبقات العليا، مع أوزان Imagenet

    # تجميد طبقات النموذج الأساسي حتى لا تتغير أثناء التدريب
    for layer in base_model.layers:  # يمر على جميع طبقات VGG16
        layer.trainable = False  # يمنع تدريب أو تغيير أوزان هذه الطبقات

    # إضافة طبقات رأس جديدة فوق النموذج الأساسي
    x = base_model.output  # يأخذ مخرجات النموذج الأساسي (VGG16)
    x = Flatten()(x)  # طبقة تسطيح
    x = Dense(256, activation='relu')(x)  # طبقة كثيفة جديدة
    x = Dropout(0.5)(x)  # طبقة تسرب

    # مخارج العمر والجنس
    age_output = Dense(1, activation='relu', name='age_output')(x)  # طبقة مخرجات العمر
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)  # طبقة مخرجات الجنس

    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])  # يُنشئ النموذج النهائي
    return model  # يُعيد النموذج المُنشأ

# ---------------------------------------------------------------------
# --- اختيار النموذج للتدريب (الجزء الذي تقومين بتعديله يدوياً) ---
# ---------------------------------------------------------------------

# قم بإلغاء تعليق النموذج الذي تريدين تدريبه:
# model_to_train = build_original_cnn_model((IMAGE_WIDTH, IMAGE_HEIGHT, 3))  # اختيار نموذج CNN الأصلي
# MODEL_SAVE_NAME = 'best_original_cnn_model.keras'  # اسم ملف حفظ النموذج الأصلي
# INITIAL_LEARNING_RATE = 0.001  # معدل التعلم الأولي لنموذج CNN الأصلي
# EPOCHS_TO_TRAIN = 30  # عدد الدورات للنموذج الأصلي

model_to_train = build_transfer_learning_vgg16_model((IMAGE_WIDTH, IMAGE_HEIGHT, 3))  # اختيار نموذج VGG16 (الافتراضي)
MODEL_SAVE_NAME = 'best_transfer_learning_vgg16_model.keras'  # اسم ملف حفظ نموذج VGG16
INITIAL_LEARNING_RATE = 0.0001  # معدل التعلم الأولي لنموذج VGG16
EPOCHS_TO_TRAIN = 50  # عدد الدورات لنموذج VGG16


# --- 6. تجميع النموذج (Compile the Model) ---
model_to_train.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),  # يحدد المُحسِّن ومعدل التعلم
                       loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},  # يحدد دالة الخسارة لكل مخرج (mse للعمر، binary_crossentropy للجنس)
                       loss_weights={'age_output': 1.0, 'gender_output': 2.0},  # يحدد أوزان الخسارة (يعطي الجنس أولوية مضاعفة)
                       metrics={'age_output': ['mae'], 'gender_output': ['accuracy']})  # يحدد المقاييس للمتابعة أثناء التدريب (MAE للعمر، الدقة للجنس)

model_to_train.summary()  # يعرض ملخصًا لهيكل النموذج وعدد المعاملات

# --- 7. معاودة الاتصال (Callbacks) ---
checkpoint = ModelCheckpoint(  # يُنشئ نقطة فحص لحفظ أفضل نموذج
    os.path.join(MODELS_DIR, MODEL_SAVE_NAME),  # مسار حفظ النموذج
    monitor='val_loss',  # يراقب خسارة التحقق
    save_best_only=True,  # يحفظ النموذج الأفضل فقط
    mode='min',  # يعني أن "الأفضل" هو الأقل في قيمة المراقبة
    verbose=1  # يعرض رسائل عند الحفظ
)

early_stopping = EarlyStopping(  # يُنشئ نقطة توقف مبكر
    monitor='val_loss',  # يراقب خسارة التحقق
    patience=20,  # يتوقف إذا لم تتحسن الخسارة لـ 20 دورة متتالية
    restore_best_weights=True,  # يستعيد أفضل أوزان للنموذج عند التوقف
    verbose=1  # يعرض رسائل عند التوقف
)

reduce_lr = ReduceLROnPlateau(  # يُنشئ نقطة لتقليل معدل التعلم
    monitor='val_loss',  # يراقب خسارة التحقق
    factor=0.2,  # يقلل معدل التعلم بنسبة 80%
    patience=10,  # ينتظر 10 دورات قبل التخفيض
    min_lr=0.00001,  # الحد الأدنى لمعدل التعلم
    verbose=1  # يعرض رسائل عند التخفيض
)

callbacks = [checkpoint, early_stopping, reduce_lr]  # يجمع جميع الـ Callbacks في قائمة

# --- 8. تدريب النموذج (Train the Model) ---
STEPS_PER_EPOCH_TRAIN = len(train_df) // BATCH_SIZE  # يحسب عدد الخطوات لكل دورة تدريبية
STEPS_PER_EPOCH_VAL = len(val_df) // BATCH_SIZE  # يحسب عدد الخطوات لكل دورة تحقق

print(f"\nStarting model training for: {MODEL_SAVE_NAME}...")  # رسالة لبدء التدريب
history = model_to_train.fit(  # يبدأ عملية تدريب النموذج
    train_dataset,  # بيانات التدريب
    steps_per_epoch=STEPS_PER_EPOCH_TRAIN,  # عدد الخطوات في كل دورة تدريب
    validation_data=val_dataset,  # بيانات التحقق
    validation_steps=STEPS_PER_EPOCH_VAL,  # عدد الخطوات في كل دورة تحقق
    epochs=EPOCHS_TO_TRAIN,  # العدد الأقصى للدورات التدريبية
    callbacks=callbacks  # قائمة الـ Callbacks التي تم تعريفها
)

print(f"\nModel training complete for: {MODEL_SAVE_NAME}.")  # رسالة عند اكتمال التدريب

# --- 9. تقييم النموذج على بيانات الاختبار (Evaluate on Test Data) ---
print("\nEvaluating model on test data...")  # رسالة لبدء التقييم

test_predictions = model_to_train.predict(test_dataset, steps=len(test_df) // BATCH_SIZE)  # يُجري التنبؤات على بيانات الاختبار

predicted_ages_scaled = test_predictions[0]  # يستخرج تنبؤات العمر المُحجّمة
predicted_genders_raw = test_predictions[1]  # يستخرج تنبؤات الجنس الخام

num_evaluated_samples = (len(test_df) // BATCH_SIZE) * BATCH_SIZE  # يحسب العدد الفعلي للعينات التي تم تقييمها
true_ages_scaled = test_df['age_scaled'].iloc[:num_evaluated_samples].values.reshape(-1, 1)  # يستخرج الأعمار الحقيقية المُحجّمة
true_genders = test_df['gender'].iloc[:num_evaluated_samples].values.reshape(-1, 1)  # يستخرج الأجناس الحقيقية

predicted_ages = age_scaler.inverse_transform(predicted_ages_scaled)  # يعيد تحجيم الأعمار المتوقعة إلى نطاقها الأصلي
true_ages = age_scaler.inverse_transform(true_ages_scaled)  # يعيد تحجيم الأعمار الحقيقية إلى نطاقها الأصلي

final_age_mae = mean_absolute_error(true_ages, predicted_ages)  # يحسب متوسط الخطأ المطلق للعمر النهائي

predicted_genders_binary = (predicted_genders_raw > 0.5).astype(int)  # يُحوّل تنبؤات الجنس إلى 0 أو 1 (أكثر من 0.5 تعتبر 1)
manual_gender_accuracy = accuracy_score(true_genders, predicted_genders_binary)  # يحسب دقة الجنس يدويًا

evaluation_results = model_to_train.evaluate(test_dataset, steps=len(test_df) // BATCH_SIZE, verbose=0)  # يُقيّم النموذج بشكل رسمي على بيانات الاختبار

print(f"\nTest Results for {MODEL_SAVE_NAME} (After Scaling and Inverse-Transforming Age):")  # يطبع عنوان النتائج
print(f"Overall Test Loss: {evaluation_results[0]:.4f}")  # يطبع إجمالي الخسارة على بيانات الاختبار
print(f"Age Mean Absolute Error (MAE - Original Scale): {final_age_mae:.4f}")  # يطبع MAE للعمر على مقياسه الأصلي
print(f"Gender Accuracy (from model.evaluate): {evaluation_results[2]:.4f}")  # يطبع دقة الجنس من تقييم النموذج
print(f"Gender Accuracy (Manual Calculation): {manual_gender_accuracy:.4f}")  # يطبع دقة الجنس المحسوبة يدويًا

# --- 10. رسم بياني لسجل التدريب (Plot Training History) ---
plt.figure(figsize=(12, 6))  # ينشئ شكل رسم بياني بحجم محدد

plt.subplot(1, 2, 1)  # ينشئ رسمًا بيانيًا فرعيًا (صف واحد، عمودين، في الموضع الأول)
plt.plot(history.history['loss'], label='Train Loss')  # يرسم خط خسارة التدريب
plt.plot(history.history['val_loss'], label='Validation Loss')  # يرسم خط خسارة التحقق
plt.title(f'Model Loss ({MODEL_SAVE_NAME.replace(".keras", "")})')  # يضع عنوانًا للرسم البياني للخسارة
plt.xlabel('Epoch')  # يضع تسمية للمحور الأفقي
plt.ylabel('Loss')  # يضع تسمية للمحور العمودي
plt.legend()  # يعرض وسيلة الإيضاح

plt.subplot(1, 2, 2)  # ينشئ رسمًا بيانيًا فرعيًا آخر (في الموضع الثاني)
plt.plot(history.history['gender_output_accuracy'], label='Train Gender Accuracy')  # يرسم خط دقة تدريب الجنس
plt.plot(history.history['val_gender_output_accuracy'], label='Validation Gender Accuracy')  # يرسم خط دقة تحقق الجنس
plt.title(f'Gender Accuracy ({MODEL_SAVE_NAME.replace(".keras", "")})')  # يضع عنوانًا للرسم البياني للدقة
plt.xlabel('Epoch')  # يضع تسمية للمحور الأفقي
plt.ylabel('Accuracy')  # يضع تسمية للمحور العمودي
plt.legend()  # يعرض وسيلة الإيضاح

plt.tight_layout()  # يضبط تخطيط الرسوم البيانية لتجنب التداخل
plt.show()  # يعرض الرسوم البيانية

plt.figure(figsize=(6, 6))  # ينشئ شكل رسم بياني جديد
plt.plot(history.history['age_output_mae'], label='Train Age MAE')  # يرسم خط MAE تدريب العمر
plt.plot(history.history['val_age_output_mae'], label='Validation Age MAE')  # يرسم خط MAE تحقق العمر
plt.title(f'Age Mean Absolute Error (MAE) ({MODEL_SAVE_NAME.replace(".keras", "")})')  # يضع عنوانًا للرسم البياني لـ MAE العمر
plt.xlabel('Epoch')  # يضع تسمية للمحور الأفقي
plt.ylabel('MAE')  # يضع تسمية للمحور العمودي
plt.legend()  # يعرض وسيلة الإيضاح
plt.tight_layout()  # يضبط تخطيط الرسم البياني
plt.show()  # يعرض الرسم البياني

print(f"\nBest model saved to: {os.path.join(MODELS_DIR, MODEL_SAVE_NAME)}")  # يطبع رسالة بمكان حفظ أفضل نموذج