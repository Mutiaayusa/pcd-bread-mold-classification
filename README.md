# Klasifikasi Roti Tawar Berjamur dan Tidak Berjamur Menggunakan Ekstraksi Fitur Tekstur GLCM dengan Perbandingan KNN, SVM, dan Random Forest

## Nama Anggota
###  MUTIA AYU SAFITRI          : F1D02410127
###  AYRA AULIA SAPUTRI HIDAYAT : F1D02410107
###  BINTANG AQRA WIBOWO        : F1D02410041
###  ZAKY                       : F1D02410143

# Project Overview
Pada project PCD ini, dilakukan eksperimen klasifikasi citra roti tawar yang dibagi menjadi dua kelas: **berjamur** dan **tidak berjamur**. Hal ini bertujuan untuk:
- Menguji kemampuan dalam mengimplementasikan teknik pengolahan citra digital untuk melakukan klasifikasi kondisi roti tawar berdasarkan tekstur permukaannya.
- Memilih tahapan preprocessing yang tepat sesuai dengan karakteristik data citra roti tawar (variasi warna jamur, pencahayaan, dan tekstur permukaan).

Pemilihan preprocessing menggunakan teknik yang telah dipelajari selama praktikum Modul 1 - 5. Setelah itu, dilakukan feature extraction menggunakan GLCM dan pembuatan model klasifikasi dengan tiga algoritma: **KNN**, **SVM**, dan **Random Forest**.

Perlu diperhatikan bahwa yang menjadi acuan pada project ini adalah tepatnya pemilihan `preprocessing` dan proses `extraction feature` yang dilakukan. Jadi, tidak perlu khawatir dengan hasil akhir akurasi yang mungkin tidak bagus. Selain itu, untuk melihat pemahaman dalam menganalisis, akan dilakukan eksperimen sebanyak **3 kali percobaan** dengan notebook yang berbeda. Pada setiap percobaannya, dilakukan improvement pada setiap preprocessing yang telah dibuat sebelumnya dengan cara menyesuaikan jumlah preprocessing pada setiap percobaan:

- **Percobaan Pertama** (1 Preprocessing: Grayscale + Histogram Equalization + Normalisasi)
- **Percobaan Kedua** (2 Preprocessing: Grayscale + Histogram Equalization + Normalisasi, Grayscale + Median Filter + Thresholding)
- **Percobaan Ketiga** (3 Preprocessing: Grayscale + Histogram Equalization + Normalisasi, Grayscale + Median Filter + Thresholding, Grayscale + Thresholding + Morfologi Opening + Morfologi Closing)
- **Percobaan Keeempat** (4 Preprocessing: Resize + Grayscale + Smoothing (Mean Filter) + Sharpening)

Dari setiap percobaan, perhatikan bagaimana perbedaan akurasinya untuk setiap model: Random Forest berapa, SVM berapa, KNN berapa. Berikut ini adalah Tahapan Umum yang digunakan dalam Machine Learning.

~ SELAMAT DATANG DI LAB 1 ~

# IMPORT LIBRARY
Import library yang dibutuhkan di cell code ini. Library pada template adalah library umum yang sekiranya sering digunakan pada Machine Learning dalam konteks klasifikasi citra, jadi gunakan library yang diperlukan saja ya.
``` python
import library
import library as lib
import library.library as lib
from library.library_yang_saya_butuhkan import library, library
```

# Load Data
Setelah import library, dilanjutkan dengan tahapan membaca dataset. Pada project ini, akan dibaca ratusan hingga ribuan image citra roti tawar beserta labelnya (**berjamur** / **tidak berjamur**). Sesuaikan code dengan label (nama setiap folder pada dataset) yang digunakan. Lakukan penyeragaman ukuran image dengan resize jika ukuran setiap image berbeda pada dataset. Sekedar informasi, semakin besar ukuran setiap image, maka proses loadingnya pun akan semakin lama, jadi usahakan ukuran image rendah, CMIIW~
``` python
data = []
labels = []
file_name = []
```

## Data Understanding
Selanjutnya, lakukan eksplorasi data untuk memahami karakteristik data yang digunakan, seperti kondisi background, keberadaan jamur, pencahayaan, distribusi data, dan sampel data. Hal ini bertujuan agar dapat memilih teknik preprocessing yang tepat untuk citra roti tawar ini.
``` python
jumlah.data = []
jumlah.labels = []
print(Output: file_name)
```

Output: Contoh Visualisasi Distribusi Data:

<img width="1389" height="848" alt="3b1049dd-b606-4079-8431-cc97d435875b" src="https://github.com/user-attachments/assets/9410e13e-387a-49ce-9d67-7e34e57eb1a6" />

Output: Contoh Sample Data:
![image](https://github.com/user-attachments/assets/0084d31f-386e-49f9-9de5-4863ec4d73de)

# Data Preparation
## Data Augmentation
Pada tahapan ini, terapkan teknik image augmentation untuk menambah jumlah data, **HANYA JIKA** data berada di bawah rentang 70-100 per kelas. Augmentasi dapat dilakukan dengan teknik yang ada pada Modul 1 (flip, rotate, brightness adjustment, dll).
``` python
augmented.data = []
augmented.labels = []
print(Output: file_name)
```

Output: Contoh Image Augmentation
![image](https://github.com/user-attachments/assets/9ea656a7-a69c-47fa-98fc-2a598b81c3a0)

## Preprocessing
Berikut adalah 3 tahapan preprocessing yang digunakan pada project ini. Pemilihan preprocessing ini disesuaikan dengan karakteristik citra roti tawar, di mana perlu menonjolkan tekstur permukaan untuk membedakan area berjamur dan tidak berjamur.

### Preprocessing 1: Grayscale + Histogram Equalization + Normalisasi
Konversi ke grayscale untuk menyederhanakan data menjadi 1 channel. Histogram Equalization diterapkan untuk meningkatkan kontras citra sehingga perbedaan tekstur antara area berjamur dan tidak berjamur lebih terlihat jelas, terutama pada citra dengan pencahayaan tidak merata. Normalisasi dilakukan untuk menyeragamkan rentang nilai piksel agar proses ekstraksi fitur lebih stabil.
```python
def prepro1(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    equalized = cv2.equalizeHist(gray)
    # Normalisasi
    normalized = equalized / 255.0
    return normalized
```

### Preprocessing 2: Grayscale + Median Filter + Thresholding
Konversi ke grayscale dilanjutkan dengan Median Filter untuk menghilangkan noise salt-and-pepper yang mungkin muncul pada permukaan roti tanpa merusak tepi tekstur. Thresholding kemudian diterapkan untuk memisahkan area berjamur (piksel gelap/berwarna) dari latar belakang (roti putih), sehingga memperjelas segmentasi objek.
```python
def prepro2(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Median Filter
    median = cv2.medianBlur(gray, 5)
    # Thresholding
    _, thresh = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY)
    return thresh
```

### Preprocessing 3: Grayscale + Thresholding + Morfologi Opening + Morfologi Closing
Konversi ke grayscale dan Thresholding untuk segmentasi awal. Morfologi Opening (erosi lalu dilasi) digunakan untuk menghilangkan noise kecil dan artefak pada hasil thresholding. Morfologi Closing (dilasi lalu erosi) digunakan untuk menutup celah kecil pada area jamur sehingga region jamur menjadi lebih solid dan utuh, membantu GLCM mengekstrak fitur tekstur yang lebih representatif.
```python
def prepro3(image):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    # Morfologi Opening
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # Morfologi Closing
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing
```

## Feature Extraction
Pada tahapan ini, dilakukan ekstraksi fitur dengan metode **Gray Level Co-occurrence Matrix (GLCM)**. GLCM dihitung dengan sudut 0°, 45°, 90°, dan 135° (simetris), serta dilakukan uji coba dengan distance 1-5. Fitur yang dihitung meliputi:

- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- Entropy
- ASM

Fitur-fitur tekstur ini sangat relevan untuk membedakan citra roti berjamur (tekstur kasar, tidak homogen) dengan roti tidak berjamur (tekstur halus, lebih homogen).
``` python
def glcm(image, derajat):
    ...........
```

## Feature Selection
Pada tahap ini, dilakukan seleksi fitur menggunakan teknik **correlation** untuk memilih fitur-fitur yang paling relevan dan menghilangkan fitur yang redundan atau tidak berkontribusi signifikan terhadap klasifikasi.
``` python
correlation = hasilEkstrak.drop(columns=['Label','Filename']).corr()
......
```

## Splitting Data
Pada tahap ini, data dibagi menjadi data training dan data testing. Gunakan perbandingan **80:20** atau **70:30** atau **90:10**.
``` python
Dataset, Dataset, Dataset, Dataset = train_test_split(Dataset, y, test_size=0.2, random_state=42)
print(Dataset.shape)
print(Dataset.shape)
```

## Normalization
Pada tahap ini, dilakukan normalisasi data menggunakan teknik **standarization** atau **min-max normalization** untuk menyeragamkan skala fitur sebelum dimasukkan ke model.
``` python
Dataset = (Dataset - Dataset.mean()) / Dataset.std()
Dataset = (Dataset - Dataset.mean()) / Dataset.std()
```

# Modeling
Pada tahap ini, dibuat tiga model klasifikasi untuk membandingkan performa dalam mengklasifikasikan roti tawar berjamur dan tidak berjamur:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest**

Gunakan akurasi sebagai metrik utama dalam menampilkan hasil klasifikasi.
```python
# Train Random Forest Classifier
rf.fit(dataset, dataset)
# Train SVM Classifier
svm.fit(dataset, dataset)
# Train KNN Classifier
knn.fit(dataset, dataset)

def inidiaClassificationReport(dataset, dataset):
    print(classification_report(dataset, dataset))
```

Output: Contoh Classification Report
|               | Accuracy | Precision | Recall   | F1-Score |
| ------------- | -------- | --------- | -------- | -------- |
| KNN           | 0.948667 | 0.948664  | 0.948667 | 0.948504 |
| SVM           | 0.976333 | 0.976319  | 0.976333 | 0.976333 |
| Random Forest | 0.959667 | 0.959822  | 0.959667 | 0.959615 |

# Evaluation
Pada bagian ini, evaluasi model klasifikasi yang telah dibuat dengan menampilkan **Confusion Matrix** dan **Classification Report** (Accuracy, Precision, Recall, F1 Score). **Jelaskan hasil evaluasi yang Anda dapatkan, analisis model mana yang paling baik dalam mengklasifikasikan roti tawar berjamur vs tidak berjamur, serta pengaruh penambahan preprocessing pada setiap percobaan terhadap performa model.**

``` python
def plot_confusion_matrix(dataset, dataset, title):
    print(confusion_matrix)
```

Output: Contoh Confusion Matrix
![image](https://github.com/user-attachments/assets/aec4ac9c-e687-4354-b02d-833caf26db6b)
