import cv2
import numpy as np
import tensorflow as tf
import json
import os

# مسیر فایل‌های مدل و کلاس‌ها 
MODEL_PATH = r"C:\git\AI_Project\driver_drowsiness_model.h5"
CLASS_LABELS_PATH = r"C:\git\AI_Project\class_labels.json"
HAAR_CASCADE_PATH = r"C:\git\AI_Project\haarcascade_frontalface_default.xml"

# بررسی وجود فایل‌ها
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"فایل {file_path} یافت نشد.")
        return False
    return True

# بررسی فایل‌های ضروری
if not (check_file_exists(MODEL_PATH) and check_file_exists(CLASS_LABELS_PATH) and check_file_exists(HAAR_CASCADE_PATH)):
    print("یکی از فایل‌های ضروری یافت نشد. لطفاً مسیرها را بررسی کنید.")
    exit()

# بارگذاری مدل آموزش‌داده‌شده
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("مدل با موفقیت بارگذاری شد.")
except Exception as e:
    print(f"خطا در بارگذاری مدل: {e}")
    exit()

# بارگذاری کلاس‌ها (برچسب‌ها)
try:
    with open(CLASS_LABELS_PATH, 'r') as f:
        class_indices = json.load(f)
    # معکوس کردن نقشه کلاس‌ها برای تبدیل عدد به نام کلاس
    class_labels = {v: k for k, v in class_indices.items()}
    print("کلاس‌ها با موفقیت بارگذاری شدند:", class_labels)
except Exception as e:
    print(f"خطا در بارگذاری کلاس‌ها: {e}")
    exit()

# بارگذاری Haar Cascade برای شناسایی چهره
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# بررسی اینکه آیا Cascade به درستی بارگذاری شده است
if face_cascade.empty():
    print(f"خطا در بارگذاری Cascade از فایل {HAAR_CASCADE_PATH}.")
    exit()

# ------ تغییر مهم: استفاده از استریم موبایل به جای دوربین لپ‌تاپ ------
# آدرس استریم (مثلاً از طریق اپلیکیشن IP Webcam)
# لطفاً IP و Port را مطابق با خروجی اپلیکیشن روی موبایل جایگزین کنید
mobile_stream_url = "http://192.168.195.133:8080/video"  # مثال، باید به IP موبایل شما تغییر کند
# ------ کد مربوط به اتصال به دوربین لپ‌تاپ (کامنت شده) ------
# برای استفاده از دوربین لپ‌تاپ به جای موبایل، آدرس VideoCapture را به 0 تغییر دهید:
#mobile_stream_url = 0
# -------------------------------------------------------------

cap = cv2.VideoCapture(mobile_stream_url)



if not cap.isOpened():
    print("عدم دسترسی به استریم دوربین موبایل. آدرس را بررسی کنید.")
    exit()

# تنظیم سایز فریم (اختیاری)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("شروع دریافت فریم از دوربین موبایل... برای خروج کلید 'q' را فشار دهید.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("فریم دریافت نشد یا استریم قطع شده است.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # شناسایی چهره‌ها
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # استخراج منطقه چهره از تصویر
        face = frame[y:y+h, x:x+w]

        # پیش‌پردازش تصویر برای مدل
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_normalized = face_resized / 255.0
        face_array = np.expand_dims(face_normalized, axis=0)  # اضافه کردن بعد دسته

        # پیش‌بینی با مدل
        prediction = model.predict(face_array)
        p_drowsy, p_non_drowsy = prediction[0]  # دو احتمال

        # منطق نمایش برچسب بر اساس احتمال‌ها
        threshold = 0.8
        if p_drowsy > threshold and p_drowsy >= p_non_drowsy:
            # اگر احتمال خواب‌آلودگی بالای 80٪ باشد
            label = f"Drowsy ({p_drowsy*100:.1f}%)"
            color = (0, 0, 255)  # قرمز
        elif p_non_drowsy > threshold and p_non_drowsy >= p_drowsy:
            # اگر احتمال هوشیاری بالای 80٪ باشد
            label = f"Non Drowsy ({p_non_drowsy*100:.1f}%)"
            color = (0, 255, 0)  # سبز
        else:
            # در غیر این صورت (هیچ‌کدام بالای 80٪ نیست)
            label = f"Normal (Drowsy: {p_drowsy*100:.1f}%, Non Drowsy: {p_non_drowsy*100:.1f}%)"
            color = (0, 255, 255)  # زرد

        # درج نوشته بر روی فریم در بالای مستطیل چهره
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

        # رسم مستطیل دور چهره
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # نمایش فریم
    cv2.imshow('Driver Drowsiness Detection', frame)

    # خروج با فشردن کلید 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزادسازی منابع
cap.release()
cv2.destroyAllWindows()
