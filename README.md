# LAPORAN MACHINE LEARNING - NABIL DEFIN JATMIKO

## Domai proyek

Sepak bola adalah olahraga paling populer di dunia, dengan jutaan penggemar dan pemain di seluruh dunia. Dalam olahraga ini, penentuan posisi pemain yang tepat sangat krusial untuk kesuksesan tim. Setiap posisi memiliki peran dan membutuhkan karakteristik skill yang berbeda. Misalnya, seorang penyerang membutuhkan kemampuan menembak dan kecepatan yang tinggi, sementara seorang bek membutuhkan kemampuan bertahan dan fisik yang kuat. Penempatan pemain yang tidak sesuai dengan keahliannya dapat menghambat performa tim secara keseluruhan.

Secara tradisional, penentuan posisi pemain seringkali didasarkan pada pengamatan subjektif oleh pelatih atau scout. Meskipun pengalaman dan intuisi memiliki perannya, metode ini rentan terhadap bias dan mungkin tidak selalu optimal dalam mengidentifikasi potensi penuh seorang pemain. Seiring dengan kemajuan teknologi dan ketersediaan data yang melimpah dalam dunia olahraga, pendekatan berbasis data dan Artificial Intelligence (AI) menawarkan solusi yang lebih objektif dan efisien.

Proyek ini bertujuan untuk membangun model klasifikasi posisi pemain sepak bola berdasarkan atribut skill mereka. Dengan memanfaatkan dataset "players_21.csv" yang saya peroleh dari Kaggle (sesuai dengan kode Anda) dan berisi berbagai data skill pemain dari FIFA 21, proyek ini akan mengelompokkan pemain ke dalam empat kategori posisi utama: Penjaga Gawang (GK), Bertahan (DF), Gelandang (MF), dan Penyerang (FW). Atribut skill yang digunakan meliputi pace, shooting, passing, defending, physic, dribbling, dan gk_skill (skill penjaga gawang).

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
    - *K-Nearest Neighbors (KNN)*: Model ini akan mengklasifikasikan pemain berdasarkan kedekatan skill mereka dengan pemain lain yang sudah diketahui posisinya. Kami akan melakukan hyperparameter tuning untuk mencari nilai 'n_neighbors' (misalnya, melalui validasi silang) yang menghasilkan akurasi terbaik.
    - *Random Forest*: Model ini akan menggunakan ensemble dari decision tree untuk membuat prediksi yang lebih robust dan akurat. Kami akan mengeksplorasi hyperparameter seperti n_estimators dan max_depth untuk mengoptimalkan kinerja model.
  - Metrik Evaluasi: Performa kedua model akan dievaluasi dan dibandingkan menggunakan metrik berikut:
    - *Akurasi (Accuracy Score)*: Mengukur proporsi prediksi yang benar dari total prediksi.
    - *Confusion Matrix*: Memberikan gambaran rinci tentang benar positif, benar negatif, salah positif, dan salah negatif untuk setiap kelas posisi.
    - *Classification Report*: Menyediakan nilai precision, recall, dan F1-score untuk setiap kelas, memberikan wawasan lebih dalam tentang kinerja model pada setiap posisi.

## Data Understanding

Untuk membangun model klasifikasi posisi pemain sepak bola, pemahaman mendalam terhadap data yang digunakan adalah langkah fundamental. Bagian ini akan menguraikan informasi kunci mengenai dataset yang dipakai dalam proyek ini.
- Dataset : 
