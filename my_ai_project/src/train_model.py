import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# استيراد النموذج الأساسي لـ Transfer Learning
from tensorflow.keras.applications import VGG16


# --- 1. تعريف المسارات (مُعدلة لتناسب هيكل المشروع MY_AI_PROJECT/src) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # هذا هو مسار المجلد src
PROJECT_ROOT = os.path.join(BASE_DIR, os.pardir) # نعود خطوة واحدة للأعلى للوصول إلى MY_AI_PROJECT

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models') # مجلد لحفظ النماذج المدربة

# التأكد من وجود مجلد النماذج
os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_labels.csv')
VAL_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'val_labels.csv')
TEST_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_labels.csv')

# --- 2. تحميل البيانات ---
train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = pd.read_csv(VAL_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

print("\nGender distribution in Test Data:")
print(test_df['gender'].value_counts())
print("\n")

# تحويل مسارات الصور لتكون مطلقة ومناسبة لنظام التشغيل الحالي
def make_absolute_paths(df):
    # تفترض أن 'image_path' في الـ CSV هي مسارات نسبية من 'PROCESSED_DATA_DIR'
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(PROCESSED_DATA_DIR, os.path.basename(x)))
    return df

train_df = make_absolute_paths(train_df)
val_df = make_absolute_paths(val_df)
test_df = make_absolute_paths(test_df)

print(f"Loaded {len(train_df)} training samples.")
print(f"Loaded {len(val_df)} validation samples.")
print(f"Loaded {len(test_df)} test samples.")

# --- تحجيم قيم العمر (Age Scaling) ---
age_scaler = MinMaxScaler(feature_range=(0, 1))
train_df['age_scaled'] = age_scaler.fit_transform(train_df[['age']])
val_df['age_scaled'] = age_scaler.transform(val_df[['age']])
test_df['age_scaled'] = age_scaler.transform(test_df[['age']])

# --- 3. تهيئة مُنشئ البيانات (ImageDataGenerator) ---
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255, # هذا ضروري للـ ImageDataGenerator
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255) # لا تضخيم لبيانات التحقق والاختبار

# تعريف output_signature لتحديد أنواع وأشكال البيانات المتوقعة من المولد
output_signature = (
    tf.TensorSpec(shape=(None, IMAGE_WIDTH, IMAGE_HEIGHT, 3), dtype=tf.float32),
    {
        'age_output': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        'gender_output': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    }
)

def generator_wrapper(keras_generator):
    for X_batch, y_batch_from_generator in keras_generator:
        age_labels = y_batch_from_generator[0].astype(np.float32).reshape(-1, 1)
        gender_labels = y_batch_from_generator[1].astype(np.float32).reshape(-1, 1)

        yield X_batch, {
            'age_output': age_labels,
            'gender_output': gender_labels
        }

train_dataset = tf.data.Dataset.from_generator(
    lambda: generator_wrapper(train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=PROCESSED_DATA_DIR,
        x_col='image_path',
        y_col=['age_scaled', 'gender'],
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='multi_output',
        shuffle=True,
        seed=42
    )),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: generator_wrapper(val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=PROCESSED_DATA_DIR,
        x_col='image_path',
        y_col=['age_scaled', 'gender'],
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='multi_output',
        shuffle=False,
        seed=42
    )),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: generator_wrapper(val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=PROCESSED_DATA_DIR,
        x_col='image_path',
        y_col=['age_scaled', 'gender'],
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='multi_output',
        shuffle=False,
        seed=42
    )),
    output_signature=output_signature
).prefetch(tf.data.AUTOTUNE)


# --- 4. دالة لبناء النموذج الأساسي (الذي بنيتيه من الصفر) ---
def build_original_cnn_model(input_shape):
    print("\nBuilding ORIGINAL CNN Model...")
    input_tensor = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x)

    age_output = Dense(1, activation='relu', name='age_output')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

    model = Model(inputs=input_tensor, outputs=[age_output, gender_output])
    return model

# --- 5. دالة لبناء النموذج المحسن (باستخدام Transfer Learning VGG16) ---
def build_transfer_learning_vgg16_model(input_shape):
    print("\nBuilding TRANSFER LEARNING (VGG16) Model...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # تجميد طبقات النموذج الأساسي حتى لا تتغير أثناء التدريب
    for layer in base_model.layers:
        layer.trainable = False

    # إضافة طبقات رأس جديدة فوق النموذج الأساسي
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # مخارج العمر والجنس
    age_output = Dense(1, activation='relu', name='age_output')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])
    return model

# ---------------------------------------------------------------------
# --- اختيار النموذج للتدريب (الجزء الذي تقومين بتعديله يدوياً) ---
# ---------------------------------------------------------------------

# قم بإلغاء تعليق النموذج الذي تريدين تدريبه:
# model_to_train = build_original_cnn_model((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
# MODEL_SAVE_NAME = 'best_original_cnn_model.keras'
# INITIAL_LEARNING_RATE = 0.001
# EPOCHS_TO_TRAIN = 30

model_to_train = build_transfer_learning_vgg16_model((IMAGE_WIDTH, IMAGE_HEIGHT, 3))
MODEL_SAVE_NAME = 'best_transfer_learning_vgg16_model.keras'
INITIAL_LEARNING_RATE = 0.0001
EPOCHS_TO_TRAIN = 50


# --- 6. تجميع النموذج (Compile the Model) ---
model_to_train.compile(optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
                      loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
                      loss_weights={'age_output': 1.0, 'gender_output': 2.0},
                      metrics={'age_output': ['mae'], 'gender_output': ['accuracy']})

model_to_train.summary()

# --- 7. معاودة الاتصال (Callbacks) ---
checkpoint = ModelCheckpoint(
    os.path.join(MODELS_DIR, MODEL_SAVE_NAME),
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# --- 8. تدريب النموذج (Train the Model) ---
STEPS_PER_EPOCH_TRAIN = len(train_df) // BATCH_SIZE
STEPS_PER_EPOCH_VAL = len(val_df) // BATCH_SIZE

print(f"\nStarting model training for: {MODEL_SAVE_NAME}...")
history = model_to_train.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
    validation_data=val_dataset,
    validation_steps=STEPS_PER_EPOCH_VAL,
    epochs=EPOCHS_TO_TRAIN,
    callbacks=callbacks
)

print(f"\nModel training complete for: {MODEL_SAVE_NAME}.")

# --- 9. تقييم النموذج على بيانات الاختبار (Evaluate on Test Data) ---
print("\nEvaluating model on test data...")

test_predictions = model_to_train.predict(test_dataset, steps=len(test_df) // BATCH_SIZE)

predicted_ages_scaled = test_predictions[0]
predicted_genders_raw = test_predictions[1]

num_evaluated_samples = (len(test_df) // BATCH_SIZE) * BATCH_SIZE
true_ages_scaled = test_df['age_scaled'].iloc[:num_evaluated_samples].values.reshape(-1, 1)
true_genders = test_df['gender'].iloc[:num_evaluated_samples].values.reshape(-1, 1)

predicted_ages = age_scaler.inverse_transform(predicted_ages_scaled)
true_ages = age_scaler.inverse_transform(true_ages_scaled)

final_age_mae = mean_absolute_error(true_ages, predicted_ages)

predicted_genders_binary = (predicted_genders_raw > 0.5).astype(int)
manual_gender_accuracy = accuracy_score(true_genders, predicted_genders_binary)

evaluation_results = model_to_train.evaluate(test_dataset, steps=len(test_df) // BATCH_SIZE, verbose=0)

print(f"\nTest Results for {MODEL_SAVE_NAME} (After Scaling and Inverse-Transforming Age):")
print(f"Overall Test Loss: {evaluation_results[0]:.4f}")
print(f"Age Mean Absolute Error (MAE - Original Scale): {final_age_mae:.4f}")
print(f"Gender Accuracy (from model.evaluate): {evaluation_results[2]:.4f}")
print(f"Gender Accuracy (Manual Calculation): {manual_gender_accuracy:.4f}")

# --- 10. رسم بياني لسجل التدريب (Plot Training History) ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Model Loss ({MODEL_SAVE_NAME.replace(".keras", "")})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['gender_output_accuracy'], label='Train Gender Accuracy')
plt.plot(history.history['val_gender_output_accuracy'], label='Validation Gender Accuracy')
plt.title(f'Gender Accuracy ({MODEL_SAVE_NAME.replace(".keras", "")})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(history.history['age_output_mae'], label='Train Age MAE')
plt.plot(history.history['val_age_output_mae'], label='Validation Age MAE')
plt.title(f'Age Mean Absolute Error (MAE) ({MODEL_SAVE_NAME.replace(".keras", "")})')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nBest model saved to: {os.path.join(MODELS_DIR, MODEL_SAVE_NAME)}")