import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class ImageDataLoader:
    def __init__(self, data_paths, image_size=(128, 128), batch_size=32):
        self.data_paths = data_paths
        self.image_size = image_size
        self.batch_size = batch_size
        self.image_paths, self.labels = self._load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.image_paths, self.labels, test_size=0.2, random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.y_train)
        self.num_classes = len(self.label_encoder.classes_)
        self.current_index = 0

    def _load_data(self):
        image_paths = []
        labels = []
        for i, data_path in enumerate(self.data_paths):
            class_name = os.path.basename(data_path)
            for image_name in os.listdir(data_path):
                image_path = os.path.join(data_path, image_name)
                image_paths.append(image_path)
                labels.append(class_name)
        return image_paths, labels

    def _load_image(self, image_path):
        try:
            # .ipynb_checkpoints uzantılı dosyaları filtrele
            if ".ipynb_checkpoints" in image_path:
                return None

            image = cv2.imread(image_path)
            if image is None:
                # Görüntü başarıyla okunamazsa, hata mesajını yazdır ve işleme devam etmez
                print(f"Error loading image: {image_path}")
                return None

            image = cv2.resize(image, self.image_size)
            # İsteğe bağlı olarak görüntüyü normalize edebilirsiniz:
            # image = image / 255.0
            return image
        except Exception as e:
            print(f"Error processing image: {image_path} - {str(e)}")
            return None

    def data_generator(self, X, y):
        while True:
            batch_images = []
            batch_labels = []
            for i in range(self.batch_size):
                if self.current_index == len(X):
                    X, y = shuffle(X, y)
                    self.current_index = 0

                image_path = X[self.current_index]
                label = y[self.current_index]

                image = self._load_image(image_path)

                batch_images.append(image)
                batch_labels.append(label)

                self.current_index += 1

            batch_images = np.array(batch_images)
            batch_labels = self.label_encoder.transform(batch_labels)

            yield batch_images, batch_labels

    def train_model(self):
        num_epochs = 10  # Uygulamanıza ve veri setinize göre bu değeri güncelleyebilirsiniz
        random_forest_model = RandomForestClassifier()
        for epoch in range(num_epochs):
            for batch_images, batch_labels in self.data_generator(self.X_train, self.y_train):
                # Görüntüleri düzleştirme işlemi
                batch_size = batch_images.shape[0]
                flattened_batch = batch_images.reshape((batch_size, -1))
                # Modeli eğitin
                print(flattened_batch)
                random_forest_model.fit(flattened_batch, batch_labels)
        return random_forest_model

    def evaluate_model(self, model):
        num_test_batches = len(self.X_test)
        total_test_samples = num_test_batches * self.batch_size
        correct_pred = 0
        for _ in range(num_test_batches):
            batch_images, batch_labels = next(self.data_generator(self.X_test, self.y_test))
            batch_size = batch_images.shape[0]
            flattened_batch = batch_images.reshape((batch_size, -1))

            # Modeli değerlendir
            predictions = model.predict(flattened_batch)
            correct_pred += np.sum(predictions == batch_labels)
        accuracy = correct_pred / total_test_samples
        return accuracy


# Veri setlerinizin bulunduğu dizinler
data_path_parrot = os.path.join('data', 'parrot')

data_path_pigeon = os.path.join('data', 'pigeon')

image_data_loader = ImageDataLoader(data_paths=[data_path_parrot, data_path_pigeon], image_size=(128, 128),batch_size=1)
model = image_data_loader.train_model()
accuracy = image_data_loader.evaluate_model(model)
print("Model Accuracy: ", accuracy)