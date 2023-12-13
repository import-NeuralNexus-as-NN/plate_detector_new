from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserListView
import sqlite3
import functions as f
import torch
import os
from datetime import datetime

Window.size = (600, 800)
Window.clearcolor = (100/255, 100/255, 100/255, 1)
Window.title = "Plate_Detector"


class MyApp(App):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.path = "default_image.jpg"
        self.plate_label = Label(text="", size_hint=(None, None))
        self.db_connection = sqlite3.connect("plates.db")
        self.create_table()

    def create_table(self):
        cursor = self.db_connection.cursor()
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS plates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT,
                        timestamp TEXT
                    )
                ''')
        self.db_connection.commit()

    def check_database_contents(self):
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT * FROM plates")
        rows = cursor.fetchall()
        print("Contents of 'plates' table:")
        for row in rows:
            print(row)

    def select_button_pressed(self, *args):
        file_chooser = FileChooserListView()
        file_chooser.path = os.getcwd()
        file_chooser.bind(on_submit=self.on_file_selected)

        popup = Popup(title='Select a File', content=file_chooser, size_hint=(None, None), size=(400, 400))
        popup.open()

    def on_file_selected(self, instance, selection, touch):
        if selection:
            selected_file = selection[0]
            print(f'Selected file: {selected_file}')
            self.path = selected_file
            self.update_image()

    def update_image(self):
        # Создаем новый виджет Image с обновленным путем к файлу
        new_image = Image(source=self.path, size=(600, 400), size_hint=(None, None))
        # Очищаем все дочерние виджеты из BoxLayout
        self.root.clear_widgets()
        # Добавляем новый виджет Image в BoxLayout
        self.root.add_widget(new_image)

        # Добавляем кнопки
        select_button = Button(text='Выбрать изображение')
        select_button.bind(on_press=self.select_button_pressed)

        answer_button = Button(text='Узнать номер')
        answer_button.bind(on_press=self.answer_button_pressed)

        # Добавляем кнопки обратно в BoxLayout
        self.root.add_widget(select_button)
        self.root.add_widget(answer_button)

        # Добавляем метку для отображения результата
        self.root.add_widget(self.plate_label)

    def answer_button_pressed(self, *args):

        new_path = 'corrected_plate.jpg'
        image_path = self.path
        first_path = 'first_part'
        second_path = 'second_part'
        weights_path = 'weights.pth'

        model = f.CustomCNN()
        model.load_state_dict(torch.load(weights_path))
        model.eval()

        f.detecting_plate(image_path, new_path)
        f.split_symbols(new_path, first_path, second_path)
        left = f.create_predictions(first_path, model)
        left.pop(0)
        right = f.create_predictions(second_path, model)

        f.delete_files_in_folder(first_path)
        f.delete_files_in_folder(second_path)

        description = left + right
        plate_is = ''.join(description)
        print(plate_is)
        self.plate_label.text = plate_is
        if plate_is != '':
            self.save_to_database(plate_is)
        else:
            self.plate_label.text = '          Номер не найден'
        self.check_database_contents()

    def save_to_database(self, plate_number):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor = self.db_connection.cursor()
        cursor.execute("INSERT INTO plates (plate_number, timestamp) VALUES (?, ?)", (plate_number, timestamp))
        self.db_connection.commit()

    def build(self):
        box = BoxLayout(orientation='vertical')

        self.image = Image(source=self.path, size=(600, 400), size_hint=(None, None))

        select_button = Button(text='Выбрать изображение')
        select_button.bind(on_press=self.select_button_pressed)

        answer_button = Button(text='Узнать номер')
        answer_button.bind(on_press=self.answer_button_pressed)

        box.add_widget(self.image)
        box.add_widget(select_button)
        box.add_widget(answer_button)

        return box


if __name__ == '__main__':
    MyApp().run()
