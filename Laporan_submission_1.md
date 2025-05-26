# LAPORAN MACHINE LEARNING - NABIL DEFIN JATMIKO

## Domain proyek

Sepak bola adalah olahraga paling populer di dunia, dengan jutaan penggemar dan pemain di seluruh dunia. Dalam olahraga ini, penentuan posisi pemain yang tepat sangat krusial untuk kesuksesan tim. Setiap posisi memiliki peran dan membutuhkan karakteristik skill yang berbeda. Misalnya, seorang penyerang membutuhkan kemampuan menembak dan kecepatan yang tinggi, sementara seorang bek membutuhkan kemampuan bertahan dan fisik yang kuat. Penempatan pemain yang tidak sesuai dengan keahliannya dapat menghambat performa tim secara keseluruhan.

Secara tradisional, penentuan posisi pemain seringkali didasarkan pada pengamatan subjektif oleh pelatih atau scout. Meskipun pengalaman dan intuisi memiliki perannya, metode ini rentan terhadap bias dan mungkin tidak selalu optimal dalam mengidentifikasi potensi penuh seorang pemain. Seiring dengan kemajuan teknologi dan ketersediaan data yang melimpah dalam dunia olahraga, pendekatan berbasis data dan Artificial Intelligence (AI) menawarkan solusi yang lebih objektif dan efisien.

Proyek ini bertujuan untuk membangun model klasifikasi posisi pemain sepak bola berdasarkan atribut skill mereka. Dengan memanfaatkan dataset "`players_21.csv`" yang saya peroleh dari Kaggle (sesuai dengan kode Anda) dan berisi berbagai data skill pemain dari FIFA 21, proyek ini akan mengelompokkan pemain ke dalam empat kategori posisi utama: Penjaga Gawang (`GK`), Bertahan (`DF`), Gelandang (`MF`), dan Penyerang (`FW`). Atribut skill yang digunakan meliputi `pace`, `shooting`, `passing`, `defending`, `physic`, `dribbling`, dan `gk_skill` (skill penjaga gawang).

Dengan menerapkan algoritma machine learning seperti K-Nearest Neighbors (KNN) dan Random Forest (sesuai dengan kode Anda), proyek ini akan mengevaluasi performa masing-masing model dalam mengklasifikasikan posisi pemain. Evaluasi akan dilakukan berdasarkan metrik akurasi, confusion matrix, dan classification report (sesuai dengan kode Anda). Proyek ini diharapkan tidak hanya menghasilkan model prediktif yang akurat, tetapi juga dapat memberikan wawasan berharga bagi pelatih, scout, dan manajer tim dalam proses pengambilan keputusan terkait penempatan pemain, pengembangan bakat, dan strategi tim secara keseluruhan.

Pemanfaatan machine learning dalam analisis performa pemain telah menunjukkan potensi besar dalam meningkatkan efisiensi dan objektivitas di dunia olahraga modern. Sebagai contoh, sebuah studi oleh Aliyarov et al. (2023) menunjukkan bahwa penerapan AI dalam analisis pertandingan sepak bola dapat sangat meningkatkan efisiensi proses peninjauan taktik permainan, termasuk identifikasi pergerakan dan posisi pemain [[1]](https://www.uzjurnal.uz/2/2023/3/index?issue=19). Selain itu, penelitian oleh Apostolou dan Tjortjis (2019) berhasil menggunakan algoritma machine learning untuk memprediksi posisi pemain dengan tingkat akurasi hingga 81.5% [[2]](https://www.researchgate.net/publication/335076326_Machine_Learning_for_Position_Detection_in_Football). Hasil ini mengindikasikan bahwa model berbasis data dapat memberikan kontribusi signifikan dalam membantu tim membuat keputusan yang lebih baik dalam pemilihan dan penempatan pemain, yang pada akhirnya berpotensi meningkatkan performa dan kesuksesan tim.

## Business Understanding

### Problem Statement

Dalam dunia sepak bola modern, penentuan posisi pemain yang optimal adalah kunci untuk membangun tim yang kuat dan efektif. Metode tradisional yang mengandalkan observasi subjektif oleh pelatih atau scout seringkali tidak efisien dan rentan terhadap bias. Ini dapat mengakibatkan:
- *Ketidakcocokan posisi*: Pemain ditempatkan pada posisi yang tidak sesuai dengan skill dan karakteristik terbaik mereka, sehingga menghambat performa individu dan tim secara keseluruhan.
- *Pengambilan keputusan yang suboptimal*: Pelatih kesulitan dalam menentukan formasi tim atau strategi pertandingan yang paling efektif karena kurangnya analisis objektif terhadap kemampuan spesifik pemain.
- *Potensi pemain tidak termanfaatkan*: Bakat pemain tidak berkembang secara maksimal karena tidak ada sistem yang sistematis untuk mengidentifikasi kekuatan dan kelemahan skill mereka secara objektif.

### Goal

Proyek ini memiliki tujuan utama sebagai berikut:
- Membangun model machine learning yang mampu mengklasifikasikan posisi pemain sepak bola (Penjaga Gawang, Bertahan, Gelandang, Penyerang) berdasarkan atribut skill mereka.
- Menyediakan alat bantu objektif bagi pelatih, scout, dan manajer tim untuk membuat keputusan yang lebih baik dalam penempatan dan pengembangan pemain.
- Meningkatkan efisiensi proses identifikasi bakat dan formasi tim dengan memanfaatkan data skill pemain.

### Solution Statement

Untuk mencapai tujuan di atas, proyek ini akan mengimplementasikan solusi berbasis machine learning dengan langkah-langkah terukur:
Pemanfaatan Dua Model Klasifikasi untuk Perbandingan Performa:
  - Mengembangkan dua model klasifikasi yang berbeda:
    - *K-Nearest Neighbors (KNN)*: Model ini akan mengklasifikasikan pemain berdasarkan kedekatan skill mereka dengan pemain lain yang sudah diketahui posisinya. Kami akan melakukan hyperparameter tuning untuk mencari nilai `n_neighbors` (misalnya, melalui validasi silang) yang menghasilkan akurasi terbaik.
    - *Random Forest*: Model ini akan menggunakan ensemble dari decision tree untuk membuat prediksi yang lebih robust dan akurat. Kami akan mengeksplorasi hyperparameter seperti n_estimators dan max_depth untuk mengoptimalkan kinerja model.
  - Metrik Evaluasi: Performa kedua model akan dievaluasi dan dibandingkan menggunakan metrik berikut:
    - *Akurasi (Accuracy Score)*: Mengukur proporsi prediksi yang benar dari total prediksi.
    - *Confusion Matrix*: Memberikan gambaran rinci tentang benar positif, benar negatif, salah positif, dan salah negatif untuk setiap kelas posisi.
    - *Classification Report*: Menyediakan nilai precision, recall, dan F1-score untuk setiap kelas, memberikan wawasan lebih dalam tentang kinerja model pada setiap posisi.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "FIFA 21 Complete Player Dataset" yang dapat diunduh dari Kaggle melalui tautan berikut: https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset?select=players_21.csv. Dataset ini merupakan kumpulan data pemain sepak bola dari game FIFA 21, yang relevan untuk menganalisis karakteristik skill pemain.

Dataset `players_21.csv` memiliki 18.944 baris (pemain) dan 107 kolom (fitur/variabel), yang menjadikannya dataset yang cukup komprehensif untuk tugas klasifikasi. Namun, setelah inspeksi awal, diketahui bahwa dataset ini memiliki banyak nilai kosong (missing values) pada berbagai kolom, terutama pada fitur-fitur yang merepresentasikan skill spesifik. Variabel-variabel yang digunakan dalam proyek ini berfokus pada atribut yang merepresentasikan kemampuan pemain dalam berbagai skill sepak bola, seperti kecepatan (`pace`), kemampuan menembak (`shooting`), `passing`, `dribbling`, pertahanan (`defending`), kekuatan fisik (`physic`), serta skill khusus penjaga gawang (`gk_skill`).

### Exploratory Data Analysis

Berikut adalah beberapa tahapan eksplorasi data yang dilakukan untuk memahami karakteristik dataset:
- Pengecekan Tipe Data:
Dilakukan pengecekan tipe data untuk setiap kolom menggunakan `player_df.info()`. Ini membantu mengidentifikasi variabel numerik dan kategorikal, serta potensi masalah tipe data yang tidak sesuai untuk analisis lebih lanjut.

- Statistika Deskriptif:
Statistika deskriptif seperti mean, median, min, max, dan standar deviasi dihitung (`player_df.describe()`) untuk mendapatkan gambaran umum tentang distribusi nilai pada variabel numerik. Hal ini memberikan pemahaman awal tentang rentang skill pemain dan penyebarannya.

- Pengecekan Missing Values:
Dilakukan perhitungan jumlah nilai kosong (`player_df.isnull().sum()`). Seperti yang diinstruksikan, teridentifikasi bahwa terdapat banyak kolom dengan nilai yang hilang, yang perlu ditangani secara strategis pada tahap data preprocessing untuk memastikan kualitas data dan kinerja model.

- Pengecekan Duplikasi Data:
Dilakukan pengecekan terhadap baris yang terduplikasi (`player_df[player_df.duplicated()]`). Dari hasil eksplorasi awal, tidak ditemukan adanya baris duplikat, yang menandakan setiap entri mewakili pemain yang unik.

- Pembagian Posisi Pemain:
Variabel `player_position` dikelompokkan menjadi empat kategori utama (Penjaga Gawang (`GK`), Bertahan (`DF`), Gelandang (`MF`), dan Penyerang (`FW`) untuk menyederhanakan target klasifikasi dan memastikan jumlah sampel yang cukup di setiap kategori. Ini adalah langkah penting untuk membuat model klasifikasi yang seimbang dan relevan dengan tujuan bisnis.

- Visualisasi Distribusi Posisi:
Visualisasi distribusi posisi menggunakan barchart dapat memberikan gambaran tentang proporsi pemain di setiap kategori posisi, membantu mengidentifikasi apakah ada ketidakseimbangan kelas yang perlu ditangani.

## Data Preparation

1. Mengecek Ringkasan Informasi Dataset
   - Mengecek informasi data menggunakan `player_df.info()`.
   - Tujuan dari pengecekan ini adalah untuk membantu memahami struktur data, mengidentifikasi tipe data dari setiap kolom, dan merupakan langkah awal yang krusial dalam proses data cleaning dan penanganan missing values yang akan dilakukan kemudian.
     
2. Mengecek Duplikasi Data
   - Mengecek duplikasi data dilakukan dengan kode  `player_df.duplicated().sum()`.
   - Pada proyek ini, tidak ditemukan adanya data yang duplikat, menunjukkan setiap entri mewakili pemain yang unik.
   - Mengecek duplikasi data bertujuan agar data tidak ganda, karena data ganda dapat mendominasi hasil perhitungan statistik yang menghasilkan kesimpulan yang bias. Proses pengecekan duplikasi diperlukan untuk mendapatkan representasi data yang akurat, efisien, dan relevan untuk pengambilan keputusan yang tepat.
     
3. Pemeriksaan dan Penanganan Nilai Kosong (_Missing Values_)
   - Mengecek missing value dapat menggunakan `player_df.isnull().sum()`.
   - Meskipun pada cek awal `player_df.isnull().sum()` akan menunjukkan banyak missing values di berbagai kolom, untuk fitur skill yang akan digunakan setelah proses seleksi fitur, missing values tersebut secara spesifik ditangani. Pada proyek ini, kolom skill spesifik penjaga gawang (`gk_diving`, `gk_handling`, `gk_kicking`, `gk_reflexes`, `gk_speed`, dan `gk_positioning`) yang memiliki nilai kosong diisi dengan rata-rata dari kolom `goalkeeping_`.
   - Nilai kosong dapat mengganggu proses pelatihan model dan menghasilkan prediksi yang tidak akurat. Oleh karena itu, penanganan seperti imputasi (pengisian nilai) atau penghapusan baris/kolom sangat penting untuk memastikan kelengkapan dan kualitas data sebelum pelatihan model.
     
4. Pengelompokan Posisi Pemain (_Position Grouping_)
   - Teknik: Feature Engineering (pengelompokan kategori).
   - Penjelasan Proses: Kolom `player_position` pada dataset awal memiliki banyak variasi posisi spesifik. Untuk menyederhanakan target klasifikasi, posisi-posisi ini dikelompokkan ke dalam empat kategori utama: Penjaga Gawang (`GK`), Bertahan (`DF`), Gelandang (`MF`), dan Penyerang (`FW`).
   - Alasan Diperlukan: Pengelompokan ini penting untuk mengurangi jumlah kelas target yang terlalu banyak, yang bisa menyebabkan masalah imbalance data dan menyulitkan model untuk belajar. Dengan mengelompokkan posisi, model dapat belajar pola yang lebih umum untuk kategori posisi yang lebih besar, menjadikan klasifikasi lebih robust dan relevan dengan konteks tim.
  
5. Pemilihan Fitur Utama (_Feature Selection_)
   - Teknik: Pemilihan subset fitur.
   - Penjelasan Proses: Hanya kolom-kolom yang secara langsung merepresentasikan skill kunci pemain (`pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`, dan `goalkeeping`) yang dipilih sebagai variabel independen (`skill`) untuk model. Kolom `position_group` yang telah dibuat pada tahap sebelumnya akan menjadi variabel target (target variable).
   - Alasan Diperlukan: Fokus pada fitur skill yang paling relevan memastikan bahwa model belajar dari informasi yang paling penting untuk membedakan posisi pemain. Hal ini juga membantu mengurangi noise dan kompleksitas model, serta meningkatkan interpretasi hasil.

6. Encoding Variabel Target Kategorikal
   - Teknik: Label Encoding.
   - Penjelasan Proses: Variabel target `position_group` yang berisi label teks kategorikal (`GK`, `DF`, `MF`, `FW`) diubah menjadi representasi numerik menggunakan `LabelEncoder` dari `scikit-learn`. Ini diperlukan karena algoritma machine learning sebagian besar hanya dapat memproses data dalam format numerik.
   - Alasan Diperlukan: Label Encoding mengubah label kategorikal menjadi nilai numerik, yang memungkinkan model untuk memproses dan belajar dari variabel target tersebut.
  
7. Pemisahan Data Latih dan Data Uji
   - Teknik: Data Splitting (menggunakan  `train_test_split`).
   - Penjelasan Proses: Dataset yang telah bersih dan disiapkan kemudian dibagi menjadi dua bagian: data latih (training data) dan data uji (testing data). Pada proyek ini, 80% data dialokasikan untuk data latih dan 20% untuk data uji. Pembagian ini dilakukan dengan `random_state` yang tetap untuk memastikan reproduksibilitas hasil dan `stratify` agar mengambil data untuk variabel targetnya seimbang.
   - Alasan Diperlukan: Pembagian data ini sangat penting untuk mengevaluasi kinerja model secara objektif. Model hanya akan dilatih menggunakan data latih dan kemudian kinerjanya diukur pada data uji yang belum pernah dilihat model sebelumnya. Ini membantu mengukur kemampuan generalisasi model dan mendeteksi masalah overfitting.

8. Skala Fitur Numerik (Feature Scaling)
   - Teknik: Standardization (menggunakan `StandardScaler`).
   - Penjelasan Proses: Fitur-fitur numerik (variabel skill pemain) yang telah dipilih kemudian diskalakan menggunakan `StandardScaler`. Proses ini mengubah nilai-nilai fitur sehingga memiliki mean 0 dan standar deviasi 1. Scaling dilakukan pada data latih (`fit_transform`) dan kemudian diterapkan pada data uji (transform) untuk mencegah data leakage.
   - Alasan Diperlukan: Banyak algoritma machine learning, terutama yang berbasis jarak seperti K-Nearest Neighbors (KNN), sangat sensitif terhadap skala fitur. Standardisasi membantu menyeimbangkan kontribusi setiap fitur, memastikan semua fitur memberikan kontribusi yang adil pada proses pelatihan model, dan seringkali meningkatkan kecepatan konvergensi serta performa model.

## Modeling

Tahap Modeling merupakan inti dari proyek machine learning ini, di mana model-model klasifikasi dibangun dan dilatih untuk menyelesaikan permasalahan penentuan posisi pemain sepak bola. Dalam proyek ini, dua algoritma machine learning yang berbeda, yaitu K-Nearest Neighbors (KNN) dan Random Forest, dipilih untuk dibandingkan performanya.

1. Pembuatan Model K-Nearest Neighbors (KNN)
   - Penjelasan Tahapan dan Parameter:
     - Inisialisasi Model: Model KNN diinisialisasi menggunakan `KNeighborsClassifier()` dari `sklearn.neighbors`.
     - Hyperparameter Tuning untuk `n_neighbors`: Untuk menemukan jumlah tetangga (`n_neighbors`) yang paling optimal, dilakukan iterasi dari `k = 1` hingga `k = 20`. Untuk setiap nilai k, model KNN dilatih pada `X_train` dan akurasinya diukur pada `X_test`. Akurasi disimpan dan nilai k dengan akurasi tertinggi dicatat sebagai best_k.
     - Model Optimal: Model KNN akhir (`knn_optimal`) diinisialisasi kembali dengan `best_k` yang telah ditemukan dan dilatih pada seluruh data latih (`X_train`, `y_train`).
     - Tujuan Parameter `n_neighbors`: Parameter ini menentukan berapa banyak tetangga terdekat yang akan dipertimbangkan dalam proses klasifikasi. Pemilihan `n_neighbors` yang tepat sangat penting karena memengaruhi bias-variance tradeoff model. Nilai k yang terlalu kecil dapat menyebabkan `overfitting`, sementara nilai k yang terlalu besar dapat menyebabkan `underfitting`.
    - Kelebihan KNN:
      - Sederhana dan Mudah Diinterpretasikan: Konsep dasarnya intuitif dan mudah dipahami.
      - Non-parametrik: Tidak membuat asumsi tentang distribusi data, sehingga fleksibel untuk berbagai jenis data.
      - Efektif untuk Dataset Kecil: Dapat bekerja dengan baik pada dataset yang tidak terlalu besar.
    - Kekurangan KNN:
      - Sensitif terhadap Outlier dan Noise: Keberadaan `outlier` atau `noise` dapat sangat memengaruhi penentuan tetangga terdekat dan akurasi klasifikasi.
      - Komputasi Mahal: Membutuhkan komputasi yang tinggi, terutama pada dataset besar, karena perlu menghitung jarak ke setiap titik data pada tahap prediksi.
      - Sensitif terhadap Skala Fitur: Perlu feature scaling (seperti standardisasi) agar fitur dengan rentang nilai yang besar tidak mendominasi perhitungan jarak.
      - Kinerja Menurun pada Dimensi Tinggi (Curse of Dimensionality): Akurasi cenderung menurun pada dataset dengan banyak fitur (dimensi tinggi) karena konsep jarak menjadi kurang bermakna.

2. Pembuatan Model Random Forest
   - Penjelasan Tahapan dan Parameter:
     - Inisialisasi Model: Model Random Forest diinisialisasi menggunakan `RandomForestClassifier()` dari `sklearn.ensemble`.
     - Parameter Default: Model ini dilatih menggunakan parameter default `scikit-learn`, dengan penambahan `random_state=42` untuk memastikan hasil yang konsisten dan dapat direproduksi. Meskipun tidak dilakukan hyperparameter tuning ekstensif pada kode yang diberikan, parameter default Random Forest seringkali sudah memberikan kinerja yang baik.
     - Potensi Parameter Penting (untuk pengembangan lebih lanjut):
       - `n_estimators`: Jumlah decision tree dalam forest. Nilai yang lebih tinggi umumnya meningkatkan akurasi tetapi memperlambat pelatihan.
       - `max_depth`: Kedalaman maksimum dari setiap decision tree. Mengontrol kompleksitas model dan membantu mencegah overfitting.
       - `min_samples_split`: Jumlah sampel minimum yang dibutuhkan untuk memisahkan sebuah node.
       - `min_samples_leaf`: Jumlah sampel minimum yang dibutuhkan pada sebuah node daun.
     - Kelebihan Random Forest:
       - Akurasi Tinggi: Seringkali memberikan akurasi klasifikasi yang sangat baik.
       - Robust terhadap Overfitting: Karena sifatnya ensemble dan penggunaan random subset fitur/sampel, Random Forest cenderung tidak overfit dibandingkan decision tree tunggal.
       - Mampu Menangani Banyak Fitur: Efektif bahkan dengan sejumlah besar fitur dan tidak terlalu sensitif terhadap feature scaling.
       - Mampu Mengukur Feature Importance: Dapat memberikan informasi tentang seberapa penting setiap fitur dalam proses klasifikasi.
     - Kekurangan Random Forest:
       - Kurang Dapat Diinterpretasikan: Sebagai model ensemble, interpretasi bagaimana prediksi dibuat lebih sulit dibandingkan decision tree tunggal.
       - Memakan Sumber Daya Komputasi: Bisa lambat dalam pelatihan dan prediksi pada dataset yang sangat besar atau dengan jumlah tree yang sangat banyak.

3. Pemilihan Model Terbaik

Berdasarkan analisis stabilitas akurasi antara data latih dan data uji, serta kinerja generalisasi, model K-Nearest Neighbors (KNN) dipilih sebagai model terbaik untuk klasifikasi posisi pemain dalam proyek ini.

Jika akurasi pelatihan dan pengujian KNN menunjukkan stabilitas yang lebih baik (perbedaan yang lebih kecil) dibandingkan dengan Random Forest, ini mengindikasikan bahwa KNN memiliki kemampuan generalisasi yang lebih baik dan lebih sedikit mengalami overfitting pada dataset ini. Meskipun Random Forest mungkin mencapai akurasi training yang sangat tinggi, perbedaan yang signifikan dengan akurasi testing menandakan bahwa model Random Forest cenderung terlalu menghafal pola pada data latih dan kurang mampu beradaptasi dengan data baru. KNN, dengan stabilitasnya, lebih dapat diandalkan untuk memprediksi posisi pemain yang belum pernah dilihat sebelumnya.

## Evaluasi

Tahap Evaluasi adalah fase krusial dalam proyek machine learning untuk menilai seberapa efektif model yang telah dibangun dalam menyelesaikan permasalahan yang ada. Pada tahap ini, kinerja model diukur menggunakan metrik yang relevan, dan hasilnya dianalisis untuk memahami kekuatan serta kelemahan model.

1.   Metrik Evaluasi yang Digunakan
     Dalam proyek klasifikasi posisi pemain sepak bola ini, metrik evaluasi yang digunakan adalah:
     - Akurasi (Accuracy Score):
       - Penjelasan: Akurasi mengukur proporsi prediksi yang benar (baik True Positives maupun True Negatives) dari total seluruh prediksi. Ini adalah metrik paling sederhana dan intuitif yang menunjukkan seberapa sering model membuat prediksi yang tepat secara keseluruhan.
     - Confusion Matrix
       - Penjelasan: Confusion matrix adalah tabel yang menampilkan performa model klasifikasi pada sekumpulan data uji yang kebenarannya telah diketahui. Tabel ini memvisualisasikan jumlah True Positives (TP), True Negatives (TN), False Positives (FP), dan False Negatives (FN) untuk setiap kelas.
       - TP (True Positive): Kelas sebenarnya adalah Positif, diprediksi Positif.
       - TN (True Negative): Kelas sebenarnya adalah Negatif, diprediksi Negatif.
       - FP (False Positive): Kelas sebenarnya adalah Negatif, diprediksi Positif (kesalahan Tipe I).
       - FN (False Negative): Kelas sebenarnya adalah Positif, diprediksi Negatif (kesalahan Tipe II).
Dari kedua model klasifikasi, didapatkan nilai akurasi adalah sebagai berikut:

|Model|train|test|
| --- | --- | --- |
| KNN | 0.8642 | 0.8519 |
| Random Forest | 0.9999 | 0.8556 |

Berdasarkan hasil akurasi pada data train dan test, diperoleh bahwa model Random Fores memiliki akurasi testing terbaik dibandingkan model KNN. Akan tetapi Random Forest memiliki gap yang sangat jauh antara training dan juga testing, maka dari itu model terbaik yaitu adalah KNN. KNN memiliki akurasi testing yang tinggi dan gap antara training dan testing tidak jauh atau bisa dibilang tidak overfitting. Artinya, model KNN mampu memprediksi target dengan kesalahan yang sangat kecil terhadap apa yang ia pelajari dari datanya.

## Daftar Referensi
Tentu, berikut adalah daftar referensi yang Anda minta:

Daftar Referensi
[1] Aliyarov, Kh., Rikhsivoev, M., Arabboev, M., Begmatov, Sh., Saydiakbarov, S., Nosirov, Kh., & Khamidjonov, Z. (2023). ARTIFICIAL INTELLIGENCE IN PERFORMANCE ANALYSIS OF FOOTBALL MATCHES AND PLAYERS. Uzjournal, 3(19). [[Link ke artikel]](https://www.uzjurnal.uz/2/2023/3/index?issue=19)

[2] Apostolou, K., & Tjortjis, C. (2019). Machine learning for position detection in football. International Conference on Artificial Intelligence and Sports, 103991. [[Link ke artikel]](https://www.researchgate.net/publication/335076326_Machine_Learning_for_Position_Detection_in_Football)
