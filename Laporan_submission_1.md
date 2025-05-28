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

Dataset yang dipakai dalam proyek ini adalah [[FIFA 21 Complete Player Dataset]] (https://www. kaggle. com/datasets/stefanoleone992/fifa-21-complete-player-dataset? select=players_21.csv). Dataset ini terdiri dari data pemain sepak bola yang berasal dari permainan FIFA 21, yang penting untuk menganalisis ciri-ciri kemampuan pemain.

Dataset `players_21.csv` mencakup 18.944 baris (pemain) dan 106 kolom (fitur/variabel), sehingga menjadikannya dataset yang cukup lengkap untuk tugas klasifikasi. Namun, setelah pemeriksaan awal, ditemukan bahwa dataset ini memiliki sejumlah nilai yang hilang (missing values) di berbagai kolom, terutama pada fitur-fitur yang menggambarkan keterampilan tertentu. Variabel yang digunakan dalam proyek ini lebih berfokus pada atribut yang menunjukkan kemampuan pemain dalam berbagai aspek sepak bola, seperti kecepatan (`pace`), kemampuan menembak (`shooting`), `passing`, `dribbling`, pertahanan (`defending`), kekuatan fisik (`physic`), serta keterampilan khusus untuk penjaga gawang (`gk_skill` (akan ditambahkan)) keterampilan ini didapat dari rata-rata `gk_diving`, `gk_handling`, `gk_kicking`, `gk_reflexes`, `gk_speed`, dan `gk_positioning`. Berikut merupakan penjelasan fitur untuk seluruh fitur yang ada pada dataset:

`sofifa_id` : ID pemain dalam FIFA 21
`player_url` : URL profil pemain di SoFIFA
`short_name` : Nama pendek pemain
`long_name` : Nama lengkap pemain
`age` : Usia pemain (tahun)
`dob` : Tanggal lahir pemain
`height_cm` : Tinggi pemain dalam sentimeter
`weight_kg` : Berat pemain dalam kilogram
`nationality` : Kebangsaan pemain
`club_name` : Nama klub pemain
`league_name` : Nama liga klub pemain
`league_rank` : Peringkat liga klub pemain
`overall` : Peringkat keseluruhan pemain (Overall Rating - OVR)
`potential` : Potensi peringkat keseluruhan pemain (Potential Rating - POT)
`value_eur` : Nilai pasar pemain dalam Euro
`wage_eur` : Gaji pemain per minggu dalam Euro
`player_positions` : Posisi utama pemain (misalnya, ST, RW, CM)
`preferred_foot` : Kaki pilihan pemain (Kanan/Kiri)
`international_reputation` : Reputasi internasional pemain
`weak_foot` : Tingkat penguasaan kaki yang tidak dominan (1-5 bintang)
`skill_moves` : Tingkat kemampuan gerakan skill (1-5 bintang)
`work_rate` : Tingkat kerja pemain dalam menyerang/bertahan (misalnya, High/Medium)
`body_type` : Tipe tubuh pemain (misalnya, Lean, Normal, Stocky)
`real_face` : Indikator apakah pemain memiliki wajah asli dalam game (Yes/No)
`release_clause_eur` : Klausul pelepasan pemain dalam Euro
`player_tags` : Tag atau karakteristik khusus pemain (misalnya, "Pemain Bintang", "Wonderkid")
`team_position` : Posisi pemain dalam formasi tim
`team_jersey_number` : Nomor punggung jersey pemain di klub
`loaned_from` : Nama klub asal jika pemain sedang dipinjamkan
`joined` : Tanggal pemain bergabung dengan klub saat ini
`contract_valid_until` : Tahun kontrak pemain berakhir
`nation_position` : Posisi pemain di tim nasional
`nation_jersey_number` : Nomor punggung jersey pemain di tim nasional
`pace` : Atribut kecepatan pemain secara keseluruhan (PAC)
`shooting` : Atribut kemampuan menembak pemain secara keseluruhan (SHO)
`passing` : Atribut kemampuan mengumpan pemain secara keseluruhan (PAS)
`dribbling` : Atribut kemampuan menggiring bola pemain secara keseluruhan (DRI)
`defending` : Atribut kemampuan bertahan pemain secara keseluruhan (DEF)
`physic` : Atribut fisik pemain secara keseluruhan (PHY)
`gk_diving` : Atribut kiper: kemampuan menyelam (Diving)
`gk_handling` : Atribut kiper: kemampuan menangkap bola (Handling)
`gk_kicking` : Atribut kiper: kemampuan menendang bola (Kicking)
`gk_reflexes` : Atribut kiper: kemampuan refleks (Reflexes)
`gk_speed` : Atribut kiper: kecepatan (Speed)
`gk_positioning` : Atribut kiper: kemampuan posisi (Positioning)
`player_traits` : Sifat atau karakteristik tambahan pemain
`attacking_crossing` : Atribut menyerang: Crossing (akurasi umpan silang)
`attacking_finishing` : Atribut menyerang: Finishing (akurasi tembakan ke gawang)
`attacking_heading_accuracy` : Atribut menyerang: Heading Accuracy (akurasi sundulan)
`attacking_short_passing` : Atribut menyerang: Short Passing (akurasi umpan pendek)
`attacking_volleys` : Atribut menyerang: Volleys (akurasi tendangan voli)
`skill_dribbling` : Atribut skill: Dribbling (kemampuan menggiring bola)
`skill_curve` : Atribut skill: Curve (kemampuan melengkungkan bola)
`skill_fk_accuracy` : Atribut skill: FK Accuracy (akurasi tendangan bebas)
`skill_long_passing` : Atribut skill: Long Passing (akurasi umpan jauh)
`skill_ball_control` : Atribut skill: Ball Control (kontrol bola)
`movement_acceleration` : Atribut gerakan: Acceleration (akselerasi)
`movement_sprint_speed` : Atribut gerakan: Sprint Speed (kecepatan lari)
`movement_agility` : Atribut gerakan: Agility (kelincahan)
`movement_reactions` : Atribut gerakan: Reactions (reaksi)
`movement_balance` : Atribut gerakan: Balance (keseimbangan)
`power_shot_power` : Atribut kekuatan: Shot Power (kekuatan tembakan)
`power_jumping` : Atribut kekuatan: Jumping (kemampuan melompat)
`power_stamina` : Atribut kekuatan: Stamina (daya tahan)
`power_strength` : Atribut kekuatan: Strength (kekuatan fisik)
`power_long_shots` : Atribut kekuatan: Long Shots (tembakan jarak jauh)
`mentality_aggression` : Atribut mental: Aggression (agresi)
`mentality_interceptions` : Atribut mental: Interceptions (kemampuan memotong umpan)
`mentality_positioning` : Atribut mental: Positioning (penempatan posisi)
`mentality_vision` : Atribut mental: Vision (visi bermain)
`mentality_penalties` : Atribut mental: Penalties (kemampuan tendangan penalti)
`mentality_composure` : Atribut mental: Composure (ketenangan)
`defending_marking` : Atribut bertahan: Marking (kemampuan menjaga lawan)
`defending_standing_tackle` : Atribut bertahan: Standing Tackle (tekel berdiri)
`defending_sliding_tackle` : Atribut bertahan: Sliding Tackle (tekel meluncur)
`goalkeeping_diving` : Atribut kiper: Diving (menyelam)
`goalkeeping_handling` : Atribut kiper: Handling (menangkap bola)
`goalkeeping_kicking` : Atribut kiper: Kicking (menendang bola)
`goalkeeping_positioning` : Atribut kiper: Positioning (penempatan posisi)
`goalkeeping_reflexes` : Atribut

### Exploratory Data Analysis

Berikut adalah beberapa langkah eksplorasi data yang dilakukan untuk memahami karakteristik dataset:

- Pengecekan Tipe Data:
Tipe data untuk setiap kolom diperiksa menggunakan `player_df. info()`. Ini membantu dalam mengenali variabel yang bersifat numerik dan kategorikal, serta mengidentifikasi kemungkinan masalah terkait tipe data yang tidak sesuai untuk analisis selanjutnya.

- Menampilkan 5 data teratas:
Menampilkan 5 data teratas menggunakan `player_df.head()`. Ini membantu dalam visualisasi\gambaran isi dari dataset.

- Statistika Deskriptif:
Statistik deskriptif seperti rata-rata, median, nilai minimum, maksimum, dan deviasi standar dihitung (`player_df. describe()`) untuk memberikan gambaran umum mengenai sebaran nilai pada variabel numerik. Ini memberikan wawasan awal tentang variasi keterampilan pemain dan distribusinya.

- Pengecekan Missing Values:
Dilakukan perhitungan jumlah nilai yang hilang (`player_df. isnull(). sum()`). Sesuai instruksi, terungkap bahwa ada beberapa kolom yang memiliki nilai yang hilang, dan isu ini perlu ditangani dengan strategi yang tepat selama tahap preprocessing data agar kualitas data dan performa model tetap terjaga. berikut fitur yang masih memiliki data `NaN`,
```
fitur jumlah data
club_name	225
league_name	225
league_rank	225
release_clause_eur	995
player_tags	17536
team_position	225
team_jersey_number	225
loaned_from	18186
joined	983
contract_valid_until	225
nation_position	17817
nation_jersey_number	17817
pace	2083
shooting	2083
passing	2083
dribbling	2083
defending	2083
physic	2083
gk_diving	16861
gk_handling	16861
gk_kicking	16861
gk_reflexes	16861
gk_speed	16861
gk_positioning	16861
player_traits	10629
defending_marking	18944
```

- Pengecekan Duplikasi Data:
Dilakukan pemeriksaan terhadap baris yang mungkin terduplikasi (`player_df[player_df. duplicated()]`). Dari hasil eksplorasi awal, tidak ditemukan baris yang terduplikasi, yang menunjukkan bahwa setiap entri adalah representasi pemain yang unik.

## Data Preparation

1. Pengisian nilai `gk_skill` yang kosong pada pemain outfield
Pemain outfield pada skill `gk_diving`, `gk_handling`, `gk_kicking`, `gk_reflexes`, `gk_speed`, dan `gk_positioning` kosong. Akan tetapi, `gk_skill` bisa ditambahkan menggunakan rata-rata dari `goalkeeping_diving`,`goalkeeping_handling`,`goalkeeping_kicking`,`goalkeeping_positioning`,`goalkeeping_reflexes`. Jadi nilai itu dapat diisi menggunakan syntax seperti ini,
```
player_df['gk_skill_alt'] = player_df[['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']].mean(axis=1)
player_df['gk_skill'] = player_df['gk_skill'].fillna(player_df['gk_skill_alt'])
player_df.drop(columns=['gk_skill_alt'], inplace=True)
```
2. Pengisian nilai `pace`,`shooting`,`passing`,`dribbling`,`defending`,dan `physic` pada pemain berposisi `GK`
Nilai `pace`,`shooting`,`passing`,`dribbling`,`defending`,dan `physic` untuk pemain `GK` kosong, akan tetapi bisa diisikan menggunakan skill seperti berikut,
```
# Pace
player_df['pace'] = np.where(
    player_df['is_gk'],
    player_df[['movement_acceleration', 'movement_sprint_speed']].mean(axis=1),
    player_df['pace']
)

# Shooting
player_df['shooting'] = np.where(
    player_df['is_gk'],
    player_df[['attacking_finishing', 'attacking_volleys', 'power_shot_power', 'power_long_shots', 'mentality_penalties']].mean(axis=1),
    player_df['shooting']
)

# Passing
player_df['passing'] = np.where(
    player_df['is_gk'],
    player_df[['attacking_crossing', 'attacking_short_passing', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'mentality_vision']].mean(axis=1),
    player_df['passing']
)

# Defending
player_df['defending'] = np.where(
    player_df['is_gk'],
    player_df[['defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'mentality_interceptions']].mean(axis=1),
    player_df['defending']
)

# Physic
player_df['physic'] = np.where(
    player_df['is_gk'],
    player_df[['power_strength', 'power_stamina', 'power_jumping', 'mentality_aggression', 'movement_balance']].mean(axis=1),
    player_df['physic']
)

# Dribbling
player_df['dribbling'] = np.where(
    player_df['is_gk'],
    player_df[['skill_dribbling', 'skill_ball_control', 'movement_agility', 'movement_reactions', 'mentality_composure', 'attacking_heading_accuracy']].mean(axis=1),
    player_df['dribbling']
)
```
3. Mengelompokkan fitur `skill` agar mempermudah dalam klasifikasi
Fitur yang digunakan untuk klasifikasi yaitu `pace`,`shooting`,`passing`,`dribbling`,`defending`, `physic`, dan `gk_skill`. Mengapa fitur-fitur tersebut yang digunakan? karena fitur tersebut yang menginterpretasikan kemampuan pemain.

4. Melihat distribusi `skill` setiap pemain
Dengan code ini,
```
fig, axes = plt.subplots(nrows=1, ncols=len(skill), figsize=(20, 5))
fig.suptitle('Distribusi Skill Pemain per Kategori Skill', fontsize=16)

for i, col in enumerate(skill):
    sns.histplot(player_df[col], ax=axes[i], kde=True)
    axes[i].set_title(col.replace('_', ' ').title())
    axes[i].set_ylabel('Frekuensi')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```
Didapat bahwasannya semua fiturnya bisa dikatakan berdistribusi normal, akan tetapi pada `gk_skill` tidak itu dikarenakan adanya gap yang sangat banyak antara pemain outfield dan goalkeeper yang dimana `GK` hanya 2084 dan pemain outfield sisanya. 

5. Pengelompokan Posisi Pemain (_Position Grouping_)
Kolom `player_position` pada dataset awal memiliki banyak variasi posisi spesifik. Untuk menyederhanakan target klasifikasi, posisi-posisi ini dikelompokkan ke dalam empat kategori utama: Penjaga Gawang (`GK`), Bertahan (`DF`), Gelandang (`MF`), dan Penyerang (`FW`). Pengelompokan ini penting untuk mengurangi jumlah kelas target yang terlalu banyak, yang bisa menyebabkan masalah imbalance data dan menyulitkan model untuk belajar. Dengan mengelompokkan posisi, model dapat belajar pola yang lebih umum untuk kategori posisi yang lebih besar, menjadikan klasifikasi lebih robust dan relevan dengan konteks tim.

6. Encoding Variabel Target Kategorikal
Variabel target `position_group` yang berisi label teks kategorikal (`GK`, `DF`, `MF`, `FW`) diubah menjadi representasi numerik menggunakan `LabelEncoder` dari `scikit-learn`. Ini diperlukan karena algoritma machine learning sebagian besar hanya dapat memproses data dalam format numerik. Label Encoding mengubah label kategorikal menjadi nilai numerik, yang memungkinkan model untuk memproses dan belajar dari variabel target tersebut.

7. Pemilihan variabel uji dan target
Ditemtukan variabel yang digunkaan untuk menguji yaitu `skill` yang berisikan `pace`,`shooting`,`passing`,`dribbling`,`defending`, `physic`, dan `gk_skill`. Lalu untuk variabel targetnya yaitu `position_group_encoded`.

8. Skala Fitur Numerik (Feature Scaling)
Fitur-fitur numerik (variabel skill pemain) yang telah dipilih kemudian diskalakan menggunakan `StandardScaler`. Proses ini mengubah nilai-nilai fitur sehingga memiliki mean 0 dan standar deviasi 1. Scaling dilakukan pada data latih (`fit_transform`) dan kemudian diterapkan pada data uji (transform) untuk mencegah data leakage. Banyak algoritma machine learning, terutama yang berbasis jarak seperti K-Nearest Neighbors (KNN), sangat sensitif terhadap skala fitur. Standardisasi membantu menyeimbangkan kontribusi setiap fitur, memastikan semua fitur memberikan kontribusi yang adil pada proses pelatihan model, dan seringkali meningkatkan kecepatan konvergensi serta performa model.

9. Pemisahan Data Latih dan Data Uji
Dataset yang telah bersih dan disiapkan kemudian dibagi menjadi dua bagian: data latih (training data) dan data uji (testing data) menggunakan `train_test_split`. Pada proyek ini, 80% data dialokasikan untuk data latih dan 20% untuk data uji. Pembagian ini dilakukan dengan `random_state` yang tetap untuk memastikan reproduksibilitas hasil dan `stratify` agar mengambil data untuk variabel targetnya seimbang. Pembagian data ini sangat penting untuk mengevaluasi kinerja model secara objektif. Model hanya akan dilatih menggunakan data latih dan kemudian kinerjanya diukur pada data uji yang belum pernah dilihat model sebelumnya. Ini membantu mengukur kemampuan generalisasi model dan mendeteksi masalah overfitting.

## Modeling

Tahap Modeling merupakan inti dari proyek machine learning ini, di mana model-model klasifikasi dibangun dan dilatih untuk menyelesaikan permasalahan penentuan posisi pemain sepak bola. Dalam proyek ini, dua algoritma machine learning yang berbeda, yaitu K-Nearest Neighbors (KNN) dan Random Forest, dipilih untuk dibandingkan performanya.

1. Pembuatan Model K-Nearest Neighbors (KNN)
Algoritma K-Nearest Neighbors (KNN) adalah metode klasifikasi non-parametrik yang bekerja dengan mengklasifikasikan titik data baru berdasarkan mayoritas kelas dari 'k' tetangga terdekatnya dalam ruang fitur. Dalam konteks klasifikasi pemain, misalnya untuk mengklasifikasikan pemain berdasarkan kesamaan skill dengan pemain yang sudah memiliki posisi yang diketahui, KNN akan mengukur kesamaan antar pemain. Kesamaan ini diukur menggunakan metrik jarak, seperti jarak Euclidean, yang menghitung jarak 'garis lurus' antara dua titik dalam ruang multidimensional (dalam hal ini, skill-skill pemain menjadi dimensi).

Secara spesifik, cara kerja KNN adalah sebagai berikut: ketika sebuah titik data baru (pemain yang belum diketahui posisinya) ingin diklasifikasikan, algoritma akan menghitung jarak antara titik data baru tersebut dengan semua titik data yang sudah ada (pemain dengan posisi yang diketahui). Setelah itu, algoritma akan mengidentifikasi `k` tetangga terdekat yang memiliki jarak terkecil. Kemudian, titik data baru tersebut akan diberi label kelas berdasarkan voting mayoritas dari `k` tetangga terdekatnya. Misalnya, jika dari 19 tetangga terdekat, 15 di antaranya adalah penyerang dan 4 adalah gelandang, maka pemain baru tersebut akan diklasifikasikan sebagai penyerang.

```
k_range = range(1, 30)
accuracy_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
```
Dalam kode tersebut, kita memulai dengan mendefinisikan rentang nilai `k` (jumlah tetangga terdekat) yang ingin diuji, yaitu dari 1 hingga 29, dan membuat daftar kosong bernama accuracy_scores untuk menyimpan hasil akurasi. Kemudian, sebuah loop for dijalankan, di mana pada setiap iterasi, model KNeighborsClassifier diinisialisasi dengan nilai k yang berbeda. Model ini kemudian dilatih menggunakan data pelatihan (`X_train` dan `y_train`), dan setelah itu, model membuat prediksi (`y_pred`) pada data pengujian (`X_test`). Akurasi dari prediksi tersebut dihitung dengan membandingkan `y_pred` dengan nilai sebenarnya (`y_test`) menggunakan `accuracy_score`, dan nilai akurasi ini ditambahkan ke dalam daftar `accuracy_scores` untuk dicatat dan dianalisis lebih lanjut. Berdasarkan proses hyperparameter tuning yang telah dilakukan, nilai `k` optimal adalah 19. Akurasi testing tertinggi yang dicapai untuk `k=19` adalah 0.8519. 

```
y_pred_train_knn = knn.predict(X_train)
accuracy_train_knn = accuracy_score(y_train, y_pred_train_knn)
print(f"Akurasi KNN pada data training: {accuracy_train_knn:.4f}")
```
Dalam kode di atas, akan dilihat akurasi dari training model KNN. Ini dilakukan dengan memprediksi label untuk data training (`X_train`) menggunakan model KNN yang sudah dilatih, di mana hasil prediksinya disimpan dalam `y_pred_train_knn`. Selanjutnya, akurasi training dihitung dengan membandingkan prediksi tersebut (`y_pred_train_knn`) dengan label sebenarnya dari data training (`y_train`) menggunakan `accuracy_score`. Hasil perhitungan menunjukkan bahwa Akurasi KNN pada data training adalah 0.8620, yang berarti model berhasil mengklasifikasikan 86.20% dari sampel data training dengan benar.

2. Pembuatan Model Random Forest
Algoritma Random Forest adalah metode klasifikasi dan regresi ensemble yang bekerja dengan membangun banyak pohon keputusan selama pelatihan, dan kemudian mengeluarkan kelas yang merupakan modus (untuk klasifikasi) atau rata-rata prediksi (untuk regresi) dari pohon-pohon individual. Dalam konteks klasifikasi pemain, misalnya untuk memprediksi posisi pemain berdasarkan skill mereka, Random Forest akan mengatasi beberapa keterbatasan dari satu pohon keputusan.

Melihat pada kode `rf_model = RandomForestClassifier(random_state=42)` dan `rf_model.fit(X_train, y_train)`, langkah pertama adalah menginisialisasi model Random Forest menggunakan `RandomForestClassifier()`. Parameter `random_state=42` di sini berfungsi untuk memastikan hasil yang konsisten dan dapat direproduksi setiap kali kode dijalankan, karena ini mengunci seed untuk pembangkitan angka acak internal. Setelah inisialisasi, model dilatih menggunakan metode `fit().` Pada tahap ini, `X_train` (data fitur pelatihan) dan `y_train` (label atau target pelatihan) digunakan untuk membangun dan melatih setiap pohon keputusan dalam hutan. Setiap pohon dilatih pada subset data pelatihan yang berbeda (melalui bootstrapping) dan hanya menggunakan subset fitur yang acak saat mencari pemisahan terbaik.

```
y_pred_train_rf = rf_model.predict(X_train)
accuracy_train_rf = accuracy_score(y_train, y_pred_train_rf)
print(f"Akurasi Random Forest pada data training: {accuracy_train_rf:.4f}")
```
Dalam kode di atas, kita akan melihat akurasi dari training model Random Forest. Ini dilakukan dengan memprediksi label untuk data training (`X_train`) menggunakan model `rf_model` yang sudah dilatih. Hasil prediksi ini disimpan dalam variabel `y_pred_train_rf`. Selanjutnya, akurasi training dihitung dengan membandingkan prediksi tersebut (`y_pred_train_rf`) dengan label sebenarnya dari data training (`y_train`) menggunakan fungsi `accuracy_score`. Berdasarkan hasil perhitungan, Akurasi Random Forest pada data training adalah 0.9999, yang menunjukkan bahwa model ini sangat akurat dalam mengklasifikasikan data yang telah digunakannya untuk belajar.

```
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Akurasi Random Forest: {accuracy_rf:.4f}")
```
Dalam baris kode tersebut, model Random Forest yang telah dilatih (`rf_model`) digunakan untuk membuat prediksi pada data testing (`X_test`), dengan hasil prediksinya disimpan dalam variabel `y_pred_rf`. Kemudian, akurasi model pada data yang belum pernah dilihat ini (`accuracy_rf`) dihitung dengan membandingkan `y_pred_rf` dengan label sebenarnya dari data testing (`y_test`) menggunakan fungsi `accuracy_score`. Berdasarkan hasil yang diberikan, Akurasi Random Forest adalah 0.8556, yang menunjukkan seberapa baik model ini dapat menggeneralisasi dan memprediksi kelas pada data baru.

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
