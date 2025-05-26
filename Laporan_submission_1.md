# LAPORAN MACHINE LEARNING - NABIL DEFIN JATMIKO

## Domain proyek

Sepak bola merupakan cabang olahraga yang paling banyak digemari di seluruh dunia, dengan jutaan penggemar serta pemain yang tersebar di berbagai penjuru. Di dalam cabang ini, penempatan posisi pemain yang tepat sangat penting untuk keberhasilan tim. Setiap posisi memiliki fungsi tersendiri dan memerlukan kemampuan yang berbeda. Contohnya, seorang penyerang harus memiliki keterampilan menembak dan kecepatan tinggi, sedangkan seorang bek diharuskan memiliki kekuatan fisik dan kemampuan bertahan. Penempatan pemain yang tidak sesuai dengan keahlian masing-masing dapat mengganggu kinerja keseluruhan tim.

Secara umum, penentuan posisi pemain sering kali didasarkan pada pandangan subjektif dari pelatih atau pengamat. Walau pengalaman dan intuisi dapat berperan, cara ini sering kali dipengaruhi oleh bias dan mungkin tidak selalu efektif dalam menemukan potensi maksimal seorang pemain. Dengan perkembangan teknologi dan akses data yang melimpah dalam bidang olahraga, pendekatan yang berlandaskan data dan Kecerdasan Buatan (AI) menawarkan cara yang lebih objektif dan efisien.

Proyek ini bertujuan untuk membangun model klasifikasi posisi pemain sepak bola berdasarkan atribut skill mereka. Dengan memanfaatkan dataset "`players_21.csv`" yang saya peroleh dari Kaggle (sesuai dengan kode Anda) dan berisi berbagai data skill pemain dari FIFA 21, proyek ini akan mengelompokkan pemain ke dalam empat kategori posisi utama: Penjaga Gawang (`GK`), Bertahan (`DF`), Gelandang (`MF`), dan Penyerang (`FW`). Atribut skill yang digunakan meliputi `pace`, `shooting`, `passing`, `defending`, `physic`, `dribbling`, dan `gk_skill` (skill penjaga gawang).

Dengan menerapkan algoritma machine learning seperti K-Nearest Neighbors (KNN) dan Random Forest (sesuai dengan kode Anda), proyek ini akan mengevaluasi performa masing-masing model dalam mengklasifikasikan posisi pemain. Evaluasi akan dilakukan berdasarkan metrik akurasi, confusion matrix, dan classification report (sesuai dengan kode Anda). Proyek ini diharapkan tidak hanya menghasilkan model prediktif yang akurat, tetapi juga dapat memberikan wawasan berharga bagi pelatih, scout, dan manajer tim dalam proses pengambilan keputusan terkait penempatan pemain, pengembangan bakat, dan strategi tim secara keseluruhan.

Pemanfaatan machine learning dalam analisis performa pemain telah menunjukkan potensi besar dalam meningkatkan efisiensi dan objektivitas di dunia olahraga modern. Sebagai contoh, sebuah studi oleh Aliyarov et al. (2023) menunjukkan bahwa penerapan AI dalam analisis pertandingan sepak bola dapat sangat meningkatkan efisiensi proses peninjauan taktik permainan, termasuk identifikasi pergerakan dan posisi pemain [[1]](https://www.uzjurnal.uz/2/2023/3/index?issue=19). Selain itu, penelitian oleh Apostolou dan Tjortjis (2019) berhasil menggunakan algoritma machine learning untuk memprediksi posisi pemain dengan tingkat akurasi hingga 81.5% [[2]](https://www.researchgate.net/publication/335076326_Machine_Learning_for_Position_Detection_in_Football). Hasil ini mengindikasikan bahwa model berbasis data dapat memberikan kontribusi signifikan dalam membantu tim membuat keputusan yang lebih baik dalam pemilihan dan penempatan pemain, yang pada akhirnya berpotensi meningkatkan performa dan kesuksesan tim.

## Business Understanding

### Problem Statement

Dalam sepak bola masa kini, menentukan posisi ideal untuk para pemain sangat penting dalam membentuk tim yang tangguh dan efisien. Pendekatan klasik yang bergantung pada penilaian subjektif dari pelatih atau pencari bakat seringkali tidak efektif dan rentan terhadap prasangka. Hal ini bisa menyebabkan:
- *Ketidaksesuaian posisi*: Pemain diletakkan pada posisi yang tidak sejalan dengan kemampuan dan karakteristik terbaik mereka, sehingga menghambat kinerja individu dan tim secara keseluruhan.
- *Keputusan yang tidak optimal*: Pelatih menghadapi kesulitan dalam memilih formasi tim atau strategi pertandingan yang paling tepat karena minimnya analisis objektif tentang kemampuan spesifik pemain.
- *Potensi pemain terabaikan*: Bakat pemain tidak dapat berkembang sepenuhnya karena tidak ada sistem yang teratur untuk mengenali kekuatan dan kelemahan keterampilan mereka secara objektif.

### Goal

Proyek ini memiliki sasaran utama sebagai berikut:
- Mengembangkan model pembelajaran mesin yang dapat mengategorikan posisi pemain sepak bola (Penjaga Gawang, Bertahan, Gelandang, Penyerang) berdasarkan karakteristik kemampuan mereka.
- Menawarkan alat yang bersifat objektif bagi pelatih, pencari bakat, dan manajer tim untuk mengambil keputusan yang lebih baik terkait penempatan dan pengembangan pemain.
- Meningkatkan efisiensi dalam proses menemukan bakat dan menyusun tim dengan menggunakan data tentang keterampilan pemain.

### Solution Statement

Untuk mencapai tujuan di atas, proyek ini akan menerapkan solusi yang didasarkan pada machine learning dengan langkah-langkah yang dapat diukur:

Pemakaian Dua Model Klasifikasi untuk Membandingkan Kinerja:
- Mengembangkan dua model klasifikasi yang berlainan:
- *K-Nearest Neighbors (KNN)*: Model ini akan mengklasifikasikan pemain berdasarkan kesamaan skill mereka dengan pemain yang sudah memiliki posisi yang diketahui. Kami akan melakukan tuning hyperparameter untuk menemukan nilai `n_neighbors` (misalnya, melalui teknik validasi silang) yang memberikan akurasi tertinggi.
- *Random Forest*: Model ini akan memanfaatkan kumpulan decision tree untuk membuat prediksi yang lebih kuat dan tepat. Kami akan menyelidiki hyperparameter seperti n_estimators dan max_depth untuk memaksimalkan performa model.
- Metrik Evaluasi: Kinerja kedua model akan dianalisis dan dibandingkan menggunakan metrik berikut:
- *Akurasi (Accuracy Score)*: Mengukur persentase prediksi yang benar dari keseluruhan prediksi.
- *Confusion Matrix*: Menyediakan gambaran terperinci tentang angka benar positif, benar negatif, salah positif, dan salah negatif untuk setiap kategori posisi.
- *Classification Report*: Menyajikan nilai precision, recall, dan F1-score untuk masing-masing kategori, memberikan wawasan mendalam mengenai performa model di setiap posisi.

## Data Understanding

Dataset yang dipakai dalam proyek ini adalah "FIFA 21 Complete Player Dataset" yang bisa diambil dari Kaggle melalui tautan berikut: https://www. kaggle. com/datasets/stefanoleone992/fifa-21-complete-player-dataset? select=players_21. csv. Dataset ini terdiri dari data pemain sepak bola yang berasal dari permainan FIFA 21, yang penting untuk menganalisis ciri-ciri kemampuan pemain.

Dataset `players_21. csv` mencakup 18.944 baris (pemain) dan 107 kolom (fitur/variabel), sehingga menjadikannya dataset yang cukup lengkap untuk tugas klasifikasi. Namun, setelah pemeriksaan awal, ditemukan bahwa dataset ini memiliki sejumlah nilai yang hilang (missing values) di berbagai kolom, terutama pada fitur-fitur yang menggambarkan keterampilan tertentu. Variabel yang digunakan dalam proyek ini lebih berfokus pada atribut yang menunjukkan kemampuan pemain dalam berbagai aspek sepak bola, seperti kecepatan (`pace`), kemampuan menembak (`shooting`), `passing`, `dribbling`, pertahanan (`defending`), kekuatan fisik (`physic`), serta keterampilan khusus untuk penjaga gawang (`gk_skill`).

### Exploratory Data Analysis

Berikut adalah beberapa langkah eksplorasi data yang dilakukan untuk memahami karakteristik dataset:

- Pengecekan Tipe Data:
Tipe data untuk setiap kolom diperiksa menggunakan `player_df. info()`. Ini membantu dalam mengenali variabel yang bersifat numerik dan kategorikal, serta mengidentifikasi kemungkinan masalah terkait tipe data yang tidak sesuai untuk analisis selanjutnya.

- Statistika Deskriptif:
Statistik deskriptif seperti rata-rata, median, nilai minimum, maksimum, dan deviasi standar dihitung (`player_df. describe()`) untuk memberikan gambaran umum mengenai sebaran nilai pada variabel numerik. Ini memberikan wawasan awal tentang variasi keterampilan pemain dan distribusinya.

- Pengecekan Missing Values:
Dilakukan perhitungan jumlah nilai yang hilang (`player_df. isnull(). sum()`). Sesuai instruksi, terungkap bahwa ada beberapa kolom yang memiliki nilai yang hilang, dan isu ini perlu ditangani dengan strategi yang tepat selama tahap preprocessing data agar kualitas data dan performa model tetap terjaga.

- Pengecekan Duplikasi Data:
Dilakukan pemeriksaan terhadap baris yang mungkin terduplikasi (`player_df[player_df. duplicated()]`). Dari hasil eksplorasi awal, tidak ditemukan baris yang terduplikasi, yang menunjukkan bahwa setiap entri adalah representasi pemain yang unik.

- Pembagian Posisi Pemain:
Variabel `player_position` dibagi menjadi empat kelompok utama yaitu Penjaga Gawang (`GK`), Bertahan (`DF`), Gelandang (`MF`), dan Penyerang (`FW`) untuk memudahkan dalam klasifikasi dan memastikan terdapat cukup sampel dalam masing-masing kategori. Ini merupakan langkah krusial untuk mengembangkan model klasifikasi yang seimbang dan sesuai dengan tujuan bisnis.

- Pemetaan Distribusi Posisi:
Pemetaan distribusi posisi melalui diagram batang bisa memberikan gambaran mengenai proporsi pemain di setiap kategori posisi, yang membantu mengidentifikasi apakah terdapat ketidakseimbangan dalam kelas yang perlu diselesaikan.

## Data Preparation

1. Mengecek Ringkasan Informasi Dataset
- Memeriksa rincian informasi dataset dapat dilakukan dengan menggunakan `player_df. info()`.
- Pengecekan ini bertujuan untuk memahami struktur dari data, mengenali tipe data di setiap kolom, serta menjadi langkah awal yang penting dalam proses pembersihan data dan penanganan nilai yang hilang yang akan dilakukan selanjutnya.

2. Mengecek Duplikasi Data
- Pengecekan untuk mendeteksi adanya data yang sama menggunakan kode `player_df. duplicated(). sum()`.
- Dalam proyek ini, tidak ditemukan data yang terduplikasi, yang menunjukkan bahwa setiap entri merepresentasikan pemain yang unik.
- Memeriksa duplikasi data dilakukan untuk memastikan tidak ada data yang berulang, karena adanya data ganda dapat mempengaruhi hasil analisis statistik, yang berpotensi menghasilkan kesimpulan yang menyesatkan. Proses ini sangat penting untuk mendapatkan representasi data yang tepat, efisien, dan sesuai untuk pengambilan keputusan yang baik.

3. Pemeriksaan dan Penanganan Nilai Kosong (Missing Values)
- Untuk memeriksa nilai yang hilang, dapat digunakan `player_df. isnull(). sum()`.
- Meskipun pada pemeriksaan awal `player_df. isnull(). sum()` menunjukkan banyaknya nilai kosong di berbagai kolom, nilai hilang untuk fitur keterampilan yang akan digunakan setelah proses pemilihan fitur akan ditangani secara spesifik. Dalam proyek ini, kolom keterampilan khusus untuk penjaga gawang (`gk_diving`, `gk_handling`, `gk_kicking`, `gk_reflexes`, `gk_speed`, dan `gk_positioning`) yang memiliki nilai kosong akan diisi dengan rata-rata dari kolom `goalkeeping_`.
- Nilai yang hilang dapat mengganggu proses pelatihan model dan mengakibatkan prediksi yang tidak akurat. Oleh sebab itu, penanganan seperti imputasi (pengisian nilai) atau penghapusan baris/kolom sangat diperlukan untuk memastikan bahwa data lengkap dan berkualitas sebelum pelatihan model dilakukan.
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

Berdasarkan kajian mengenai kestabilan akurasi antara data latih dan data uji, serta kemampuan generalisasi, model K-Nearest Neighbors (KNN) dipilih sebagai model yang paling unggul untuk mengklasifikasikan posisi pemain dalam proyek ini.

Apabila akurasi pelatihan dan pengujian KNN menunjukkan kestabilan yang lebih baik (dengan perbedaan yang lebih kecil) dibandingkan dengan Random Forest, hal ini menandakan bahwa KNN memiliki kemampuan generalisasi yang lebih baik dan mengalami overfitting yang lebih sedikit pada dataset ini. Walaupun Random Forest dapat mencapai tingkat akurasi yang sangat baik pada data pelatihan, perbedaan besar dalam akurasi pengujian menunjukkan bahwa model ini cenderung terlalu mengingat pola pada data latih dan kurang efektif dalam menyesuaikan diri dengan data yang baru. KNN, karena stabilitasnya, lebih tepercaya dalam memprediksi lokasi pemain yang belum pernah diperhatikan sebelumnya.

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
