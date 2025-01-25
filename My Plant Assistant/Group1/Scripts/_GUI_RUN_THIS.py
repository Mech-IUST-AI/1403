import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

from fruit_offline_test import test_saved_image as test_fruit_offline
from flower_offline_test import test_saved_image as test_flower_offline
from disease_offline_test import predict_and_display as test_disease_offline
from fruit_online_test import live_prediction_with_price as test_fruit_online
from flower_online_test import live_prediction as test_flower_online
from disease_online_test import run_online_plant_disease_detection as test_disease_online


persian_font_path = "IRANSans.ttf"
logo_path = "logo.png"

# Helper function to center a window on the screen
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")

def browse_image():
    file_path = filedialog.askopenfilename(title="تصویر خود را انتخاب کنید", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    return file_path

def get_ip_webcam_url():
    ip_window = tk.Toplevel()
    ip_window.title("ورود آی‌پی دستگاه")
    center_window(ip_window, 600, 200)

    tk.Label(ip_window, text=("آی‌پی دستگاه خود را وارد کنید\n نمونه: http://192.168.XXX.XXX:8080/shot.jpg"), font=("Arial", 14)).pack(pady=20)

    ip_entry = tk.Entry(ip_window, font=("Arial", 14), width=50)
    ip_entry.pack(pady=10)

    ip_url = tk.StringVar()

    def submit_ip():
        ip_url.set(ip_entry.get())
        ip_window.destroy()

    tk.Button(ip_window, text="انتخاب", font=("Arial", 14), command=submit_ip).pack(pady=10)

    ip_window.grab_set()
    ip_window.wait_window()
    return ip_url.get()

def on_fruit_selected():
    def online_testing():
        ip = get_ip_webcam_url()
        if not ip:
            raise FileNotFoundError("No IP Webcam URL provided.")
        test_fruit_online(url=ip, font_path=persian_font_path)

    def offline_testing():
        image_path = browse_image()
        if not image_path:
            raise FileNotFoundError
        test_fruit_offline(image_path=image_path)

    testing_window = tk.Toplevel()
    testing_window.title(":یک مورد را انتخاب کنید")
    center_window(testing_window, 500, 200)

    tk.Button(testing_window, text="تست آنلاین", command=online_testing, font=("Arial", 14)).pack(pady=20)
    tk.Button(testing_window, text="تست آفلاین", command=offline_testing, font=("Arial", 14)).pack(pady=20)

def on_flower_selected():
    def online_testing():
        ip = get_ip_webcam_url()
        if not ip:
            raise FileNotFoundError("No IP Webcam URL provided.")
        test_flower_online(url=ip, font_path=persian_font_path)

    def offline_testing():
        image_path = browse_image()
        if not image_path:
            raise FileNotFoundError
        test_flower_offline(image_path=image_path)

    testing_window = tk.Toplevel()
    testing_window.title(":یک مورد را انتخاب کنید")
    center_window(testing_window, 500, 200)

    tk.Button(testing_window, text="تست آنلاین", command=online_testing, font=("Arial", 14)).pack(pady=20)
    tk.Button(testing_window, text="تست آفلاین", command=offline_testing, font=("Arial", 14)).pack(pady=20)

def on_disease_selected():
    def online_testing():
        ip = get_ip_webcam_url()
        if not ip:
            raise FileNotFoundError("No IP Webcam URL provided.")
        test_disease_online(ip_webcam_url=ip)

    def offline_testing():
        image_path = browse_image()
        if not image_path:
            raise FileNotFoundError
        test_disease_offline(image_path=image_path)

    testing_window = tk.Toplevel()
    testing_window.title(":یک مورد را انتخاب کنید")
    center_window(testing_window, 500, 200)

    tk.Button(testing_window, text="تست آنلاین", command=online_testing, font=("Arial", 14)).pack(pady=20)
    tk.Button(testing_window, text="تست آفلاین", command=offline_testing, font=("Arial", 14)).pack(pady=20)

def on_help_selected():
    help_window = tk.Toplevel()
    help_window.title("راهنما")
    center_window(help_window, 700, 350)

    new_help_text = (
        ":سلام، جهت استفاده از مدل طراحی شده راهنمای زیر را مطالعه فرمایید\n\n\n"
        ".هر 3 مدل طراحی شده را میتوان به دو صورت آفلاین و آنلاین استفاده نمود\n"
        ".جهت اسفاده از تست آفلاین بایستی تصویر مورد نظر خود را انتخاب کنید\n"        
        ".جهت اسفاده از تست آنلاین بایستی لینک آی‌پی دستگاه خود را وارد نمایید\n\n"
        ".استفاده گردد IP Webcam پیشنهاد می‌شود جهت استفاده از تست آنلاین از نرم‌افزار \n"
        ".توجه داشته باشید که برای اتصال مدل به دستگاه بایستی هر دو به یک شبکه متصل باشند\n"
        "نمایش قیمت روز میوه در میادین میوه تره بار تهران از طریق سایت سروبان و فقط در \n.صورت اتصال به اینترنت در تست آنلاین صورت میپذیرد"
    )
    tk.Label(help_window, text=new_help_text, font=("Arial", 14), justify="right", anchor="e").pack(pady=20)

def create_gui():
    root = tk.Tk()
    root.title("دستیار گل و گیاه من")
    center_window(root, 600, 500)

    try:
        logo_image = Image.open(logo_path)
        logo_image = logo_image.resize((120, 120), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_image)
        logo_label = tk.Label(root, image=logo_photo)
        logo_label.image = logo_photo
        logo_label.pack(pady=10)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load logo image: {str(e)}")

    tk.Label(root, text=":یک مورد را انتخاب کنید", font=("Arial", 16)).pack(pady=20)

    tk.Button(root, text="تشخیص آسیب گیاه و ارائه راه حل", command=on_disease_selected, font=("Arial", 14), width=21).pack(pady=15)
    tk.Button(root, text="تشخیص نوع میوه و نمایش قیمت روز", command=on_fruit_selected, font=("Arial", 14), width=23).pack(pady=15)
    tk.Button(root, text="تشخیص نوع گل", command=on_flower_selected, font=("Arial", 14), width=11).pack(pady=15)
    tk.Button(root, text="راهنما", command=on_help_selected, font=("Arial", 14), width=5).pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    create_gui()

