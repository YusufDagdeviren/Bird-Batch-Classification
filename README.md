(En)
# Bird Detection Model Training and Evaluation

This project is a Python script designed to train and evaluate a bird detection model based on images. It uses the OpenCV and Scikit-Learn libraries.

## Dependencies

To run this project, the following packages must be installed:

- opencv-python
- scikit-learn
- numpy

You can install these packages using pip with the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Ensure that the `data` directory contains subdirectories for each bird species, such as `parrot` and `pigeon`, with corresponding images.

2. **Running the Script**: Execute the `main.py` script via the terminal or command line. For example:

```bash
python main.py
```

3. **Training the Model**: The script will automatically load the data, split it into training and testing sets, and train a random forest classifier (`RandomForestClassifier`). The training process will iterate over the dataset for a specified number of epochs.

4. **Model Evaluation**: After training, the model will be evaluated against the test dataset. Performance metrics such as accuracy, precision, recall, and F1 score will be calculated and displayed.

## File Structure

- `main.py`: The main script file responsible for training and evaluating the model.
- `data/`: Directory containing subdirectories for each bird species with corresponding images.
- `requirements.txt`: Lists the required Python packages for this project.

## Perfomance
- **Model Accuracy:** 0.8125 
- **Model Precision:** 0.8636363636363636 
- **Model Recall:** 0.8125 
- **Model F1 Score:** 0.8056680161943319

## Notes

- The size and quality of the dataset can significantly impact the performance of the model.
- The performance of the model depends on the distribution of bird species in the dataset and the quality of the images.
- Random Forest Classifier may not provide optimal performance on image datasets. Consider exploring other algorithms for better results.
---
(Tr)

# Kuş Tespiti Modeli Eğitimi ve Değerlendirmesi

Bu proje, görüntü tabanlı bir kuş tespiti modelini eğitmek ve değerlendirmek için kullanılan bir Python scriptidir. Projede OpenCV ve Scikit-Learn kütüphaneleri kullanılmaktadır.

## Bağımlılıklar

Bu projeyi  çalıştırmak için aşağıdaki paketlerin kurulu olması gerekmektedir:

- opencv-python
- scikit-learn
- numpy

Bu paketleri pip ile aşağıdaki komutla kurabilirsiniz:

```bash
pip install -r requirements.txt
```

## Kullanım

1. **Veri Hazırlama**: `data` dizininde her bir kuş türü için ayrı alt klasörler oluşturulmalıdır. Örneğin, `parrot` ve `pigeon` adında iki alt klasör olmalıdır. Bu klasörlerde ilgili kuş türünün resimleri saklanmalıdır.

2. **Script  Çalıştırma**: Terminal veya komut satırı üzerinden `main.py` dosyasını  çalıştırabilirsiniz. Örneğin:

```bash
python main.py
```

3. **Model Eğitimi**: Script  çalıştırıldığında, veriler otomatik olarak yüklenir, eğitim ve test veri setlerine bölünür ve bir rastgele orman sınıflandırıcısı (`RandomForestClassifier`) kullanılarak eğitilir. Eğitim, belirtilen sayıda epoch'ta gerçekleştirilir.

4. **Model Değerlendirmesi**: Eğitim tamamlandıktan sonra, model test veri setine karşı değerlendirilir ve performans metrikleri (doğruluk, hassasiyet, hatırlama, F1 skoru) hesaplanır ve ekrana yazdırılır.

## Dosya Yapısı

- `main.py`: Ana script dosyası, modelin eğitimi ve değerlendirmesi için kullanılır.
- `data/`: Veri setlerinin bulunduğu klasör. Her bir kuş türü için ayrı alt klasörler oluşturulmalıdır.
- `requirements.txt`: Projede gereken Python paketlerinin listesi.

## Performans
- **Model Accuracy:** 0.8125 
- **Model Precision:** 0.8636363636363636 
- **Model Recall:** 0.8125 
- **Model F1 Score:** 0.8056680161943319

## Notlar

- Veri setlerinin boyutu ve kalitesi, modelin performansına etkileyebilir.
- Modelin performansı, veri setindeki kuş türlerinin dağılımına ve resimlerin kalitesine bağlıdır.
- Rastgele Orman Sınıflandırıcısı, genellikle görüntü tabanlı veri setlerinde iyi performans göstermez. Diğer algoritmaları denemek isteyebilirsiniz.
