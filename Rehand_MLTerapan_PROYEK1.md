# Laporan Proyek Prediksi COPPA Risk - Rehand Naifisurya Hermansyah
## Domain Proyek
Dunia digital semakin mudah diakses oleh anak-anak, dengan aplikasi seluler yang memainkan peran penting dalam hiburan, pendidikan, dan interaksi sosial mereka. Namun, peningkatan akses ini juga membawa potensi risiko, terutama terkait pengumpulan dan penggunaan informasi pribadi anak-anak. Children's Online Privacy Protection Act (COPPA) di Amerika Serikat, dan peraturan serupa di seluruh dunia, bertujuan untuk melindungi privasi anak-anak secara online dengan mewajibkan pengembang aplikasi untuk mendapatkan izin orang tua sebelum mengumpulkan data dari pengguna di bawah usia 13 tahun.[1]

Model machine learning yang dapat memprediksi apakah sebuah aplikasi seluler berisiko melanggar COPPA. Dengan mengidentifikasi aplikasi yang berpotensi tidak patuh, kami dapat membantu toko aplikasi, pengembang, dan orang tua untuk menciptakan lingkungan online yang lebih aman bagi anak-anak. Model akan menganalisis berbagai karakteristik aplikasi, termasuk genre, target audiens (tersirat dari rentang unduhan), fitur kebijakan privasi, dan informasi pengembang, untuk menilai kemungkinan ketidakpatuhan terhadap COPPA.[2]

Perlindungan anak menjadi penting karena anak-anak merupakan pengguna rentan yang belum memahami risiko pelanggaran privasi. Peran regulasi hukum berperan sekali terhadap pemberian perlindungan ekstra pada anak-anak. Selain itu, jumlah pemakai aplikasi yang digunakan oleh anak-anak semakin banyak. Banyak sekali praktik komersial yang tidak pantas secara otomatis terdapat pada aplikasi. Iklan yang tidak pantas ditonton untuk anak usia yang cukup umur.[3]

Model yang dibangun mempunyai tujuan pada prediksi risiko dalam ketidakpatuhan aplikasi terhadap peraturan COPPA. Analisis dilakukan seperti karakterisktik aplikasi genre, jumlah unduhan, informasi, dan lain sebagainya. Model ini, diharapkan membantu pihak-pihak terkait untuk melakukan identifiaski aplikasi yang memungkinkan melakukan pelanggaran COPPA secara otomatis. [4]


## Business Understanding
Perusahaan atau organisasi yang mengoperasikan aplikasi di Google Play Store menghadapi kendala dalam mematuhi regulasi privasi anak-anak, terutama COPPA (Children’s Online Privacy Protection Act). COPPA mengatur cara aplikasi seharusnya mengelola data pribadi dari pengguna yang berusia di bawah 13 tahun. Untuk mendukung regulator atau pengelola aplikasi dalam mengenali serta memahami potensi pelanggaran COPPA, dikembangkanlah sebuah model klasifikasi yang dinamakan CoppaRisk. Oleh karena itu diperlukan cara bagaimana model ini dapat memprediksi Coppa Risk dengan baik.Salah satu cara untuk mengoptimalkan sesuai dengan statement 'Garbage in Garbage Out' diperlukan data yang bersih.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Bagaimana cara identifikasi risiko tinggi atau rendah terhadap pelanggaran COPPA?
- Bagaimana proses yang dilakukan agar data CoppaRisk menjadi lebih bersih?

### Goals
- Membangun model klasifikasi risiko COPPA (rendah vs tinggi).
- Melakukan berbagai teknik cleaning data agar data siap melakukan training.


### Solution statements
Dalam proyek ini, diterapkan dua pendekatan model, yaitu Deep Learning dan Support Vector Machine (SVM). Deep Learning dipilih karena kemampuannya untuk mengidentifikasi pola-pola rumit melalui struktur jaringan saraf yang mendalam, sedangkan SVM digunakan sebagai model perbandingan karena konsistensinya dalam melakukan klasifikasi dengan batas yang jelas. Agar mendapatkan kinerja optimal dari SVM, perlu dilakukan penyesuaian hyperparameter pada parameter-parameter seperti kernel, nilai C, dan gamma. Penilaian model dilakukan dengan menggunakan metrik akurasi untuk menilai kinerja secara keseluruhan dan matriks kebingungan untuk menganalisis sebaran kesalahan prediksi pada setiap kelas dengan lebih rinci.

  - Support Vector Machine (SVM) merupakan algoritma dalam pembelajaran mesin yang sangat efisien untuk tugas klasifikasi, regresi, serta deteksi anomali. SVM berfungsi dengan mencari hyperplane (garis pembatas dalam dimensi tinggi) yang secara ideal memisahkan data dari dua kategori. Tujuannya adalah untuk memaksimalkan margin, yaitu jarak antara hyperplane dan titik data terdekat dari setiap kelas (yang disebut support vectors), sehingga model menjadi lebih umum dan tidak mudah mengalami overfitting.SVM akan mencari hyperplane optimal yang memisahkan data ke dalam dua kelas (tinggi dan rendah risiko COPPA).Efektif untuk dataset yang tidak terlalu besar dengan fitur yang terstandarisasi.[Referensi](https://scikit-learn.org/stable/modules/svm.html)
  
  - Model Deep Learning yang digunakan adalah jaringan saraf tiruan (artificial neural network) dengan beberapa lapisan tersembunyi. Model ini dilatih untuk mempelajari pola kompleks dan hubungan non-linear antar fitur-fitur dalam dataset. Dengan kemampuan pembelajaran berlapis (layered learning), Deep Learning dapat mengidentifikasi kombinasi fitur yang secara bersama-sama mempengaruhi kemungkinan suatu aplikasi memiliki risiko COPPA tinggi.[Referensi](https://www.tensorflow.org/guide/keras/sequential_model)
  

## Data Understanding
### Informasi
 Dataset ini berisi informasi yang diambil dari major app marketplace, bersama dengan fitur turunan yang terkait dengan privasi dan kepatuhan. Jumlah Dataset sebanyak 7000. Dataset ini mencakup campuran fitur kategorikal, numerik, dan boolean. Nilai yang hilang diwakili oleh string kosong. Dari hasil identifikasi, ditemukan adanya nilai hilang pada beberapa kolom, seperti countryCode, downloads, ratingCount, softwareVersion, dan lainnya, dengan total yang bervariasi—misalnya countryCode memiliki 64 nilai yang hilang. Selain itu, ditemukan sebanyak 3 baris data duplikat yang perlu ditangani pada tahap praproses. Analisis terhadap outlier dilakukan pada fitur numerik menggunakan metode IQR (Interquartile Range). Ditemukan bahwa fitur seperti userRatingCount, isCorporateEmailScore, dan price memiliki jumlah outlier yang signifikan, yang menunjukkan adanya nilai ekstrem yang berbeda jauh dari mayoritas data lainnya. Dataset ini juga memiliki tipe data yang bervariasi, mencakup data numerik dan kategorikal, yang akan memengaruhi strategi praproses selanjutnya. Informasi ini memberikan gambaran awal mengenai kompleksitas dan tantangan dalam pengolahan data sebelum tahap pemodelan dilakukan.  Fitur adSpent memiliki jumlah nilai hilang terbanyak, yaitu sebanyak 5.679 entri, diikuti oleh appContentBrandSafetyRating sebanyak 6.162 entri, dan hasTermsOfServiceLink serta hasTermsOfServiceLinkRating masing-masing sebanyak 4.635 entri yang kosong. Selain itu, fitur averageUserRating juga memiliki 1.232 nilai hilang, sedangkan isCorporateEmailScore memiliki 1.128 entri kosong. Nilai hilang lainnya terdapat pada downloads (2.149), hasPrivacyLink (750), appAge (50), dan countryCode (64). Sementara itu, fitur-fitur seperti developerCountry, userRatingCount, primaryGenreName, deviceType, appDescriptionBrandSafetyRating, mfaRating, dan coppaRisk tidak memiliki nilai hilang sama sekali. Adanya jumlah missing values yang cukup besar pada beberapa kolom menunjukkan bahwa perlu dilakukan penanganan khusus, seperti imputasi atau penghapusan baris, agar kualitas data tetap terjaga sebelum digunakan untuk analisis lanjutan atau pelatihan model.



### Sumber Data 
Dataset yang digunakan  adalah "Help Protect Children Online: Develop a Model to Identify Apps Potentially Violating COPPA" yang dapat diunduh dari Kaggle ([link](https://www.kaggle.com/competitions/data-analytics-competition-find-it-2025)).

### Variabel atau fitur pada Dataset

- developerCountry: Negara tempat pengembang aplikasi terdaftar. Hal ini dapat memberikan wawasan tentang kesadaran dan penegakan hukum privasi regional. Nilai seperti "ALAMAT TIDAK TERDAFTAR DI PLAYSTORE" dan "TIDAK DAPAT MENGENAL NEGARA" menunjukkan informasi yang hilang atau tidak dapat diperoleh.

- countryCode: Target pasar atau wilayah untuk aplikasi (misalnya, GLOBAL, NA, EMEA, LATAM, APAC). NA kemungkinan berarti Amerika Utara. GLOBAL adalah kategori yang luas, sementara yang lain lebih spesifik secara geografis.

- userRatingCount: Jumlah total peringkat pengguna yang telah diterima aplikasi.
- primaryGenreName: Kategori utama aplikasi yang terdaftar di bawahnya (misalnya, Game, Buku & Referensi, Pendidikan, dll.).
- Unduhan: Kisaran perkiraan berapa kali aplikasi telah diunduh dengan format (min-max) (misalnya, "100000 - 500000").
- deviceType: Menunjukkan jenis perangkat yang kompatibel dengan aplikasi (misalnya, ponsel cerdas, tablet, tv/ott yang tersambung, belum ditentukan).
- hasPrivacyLink: Apakah daftar aplikasi menyertakan tautan ke kebijakan privasi. Keberadaan tautan merupakan langkah kepatuhan dasar.
- hasTermsOfServiceLink: Apakah daftar aplikasi menyertakan tautan ke persyaratan layanan.
- hasTermsOfServiceLinkRating: Peringkat (misalnya, "rendah", "sedang", "tinggi") yang berpotensi mencerminkan kualitas atau kelengkapan persyaratan layanan.
- isCorporateEmailScore: Skor (kemungkinan antara 0 dan 100) yang dapat mengindikasikan kemungkinan bahwa alamat email pengembang adalah email perusahaan dan bukan email pribadi. Skor yang lebih tinggi dapat mengindikasikan pengembang yang lebih mapan, yang mungkin memiliki praktik kepatuhan yang lebih baik.
- adSpent: Jumlah uang yang dihabiskan untuk iklan.
- appAge: Usia aplikasi dalam hitungan hari (kemungkinan besar sejak terdaftar di toko aplikasi). Aplikasi yang lebih tua mungkin cenderung tidak diperbarui dengan praktik terbaik privasi saat ini.
- averageUserRating: Peringkat rata-rata pengguna aplikasi (misalnya, pada skala 1 hingga 5).
- appContentBrandSafetyRating: Peringkat (misalnya, "rendah", "sedang", "tinggi") yang mungkin mencerminkan kesesuaian konten aplikasi untuk audiens umum. Ini tidak secara langsung berkaitan dengan privasi, tetapi dengan moderasi konten.
- appDescriptionBrandSafetyRating: (Kategorikal) Serupa dengan yang di atas, tetapi secara khusus memberi peringkat pada deskripsi aplikasi, dan bukan pada aplikasi itu sendiri.
- mfaRating: (Kategorikal) Peringkat (misal, "rendah", "sedang", "tinggi") yang mungkin berkaitan dengan Aplikasi yang Dibuat untuk Periklanan.
- coppaRisk: Variabel sasaran yang mengindikasikan apakah aplikasi ini berisiko tidak mematuhi COPPA atau tidak.

### Tahapan Visualisasi dan Eksplorasi Data (EDA) 

1. **Informasi Umum Dataset (`df.info()`)**
   - Menampilkan jumlah baris, kolom, tipe data tiap kolom, dan jumlah nilai non-null.
   - Digunakan untuk mengidentifikasi data kosong dan memahami struktur awal dataset.

2. **Statistik Deskriptif (`df.describe(include='all')`)**
   - Menyediakan ringkasan statistik untuk fitur numerik dan kategorikal.
   - Meliputi nilai minimum, maksimum, rata-rata, standar deviasi, serta jumlah nilai unik.

3. **Profiling Data Otomatis dengan `ydata-profiling`**
   - Menghasilkan laporan eksplorasi data yang menyeluruh, termasuk:
     - Distribusi nilai
     - Korelasi antar fitur
     - Nilai hilang (missing values)
     - Duplikasi
     - Statistik ringkasan per kolom
   - Mempermudah pemahaman awal dataset secara visual dan menyeluruh.

4. **Matriks Korelasi Numerik (`df.corr()`)**
   - Menghitung korelasi Pearson antar fitur numerik.
   - Digunakan untuk mengidentifikasi hubungan linear yang signifikan antar fitur.

5. **Visualisasi Korelasi dengan Heatmap (`seaborn.heatmap`)**
   - Menampilkan korelasi antar fitur dalam bentuk visual.
   - Warna digunakan untuk merepresentasikan kekuatan dan arah korelasi:
     - Merah/oranye: Korelasi positif tinggi
     - Biru: Korelasi negatif
     - Putih: Tidak ada korelasi
   - Membantu menentukan fitur yang mungkin redundant atau saling berkaitan kuat.

6. **Pendeteksian Nilai Anomali atau Inkonsisten**
   - EDA memungkinkan identifikasi:
     - Outlier (nilai ekstrim)
     - Nilai kosong
     - Tipe data tidak sesuai
     - Nilai kategori yang salah eja atau tidak konsiste


- _Data Profiling_, Laporan profil data dapat dihasilkan menggunakan library ydata-profiling untuk memberikan informasi statistik deskriptif, distribusi variabel, dan pola data yang mungkin ada dalam dataset.

Berikut fungsi dari ydp.ProfileReport(df):
  - Menganalisis semua data di DF DataFrame. 
 
  - mengarah ke ringkasan statistik deskripsi. 
 
  - Distribusi berharga yang ditampilkan secara visual untuk setiap kolom. 
 
  - mendeteksi nilai yang hilang, data abnormal dan ganda. 
 
  - memberikan gambaran umum tentang kualitas data dan tipe data. 
 
  - dapat disimpan secara langsung dalam bentuk file HTML untuk dilihat di browser.


Adapun hasil dari analisis yang diambil dari Data Profiling. Diketahui bahwa data terdiri dari 17 variabel dengan total 7.000 observasi. Secara keseluruhan, terdapat 26.484 sel data yang kosong, yang setara dengan sekitar 22,3% dari keseluruhan nilai dalam dataset. Angka ini menunjukkan bahwa penanganan missing values merupakan langkah penting dalam tahap praproses data. Selain itu, ditemukan adanya 3 baris data duplikat, yang meskipun hanya mewakili kurang dari 0,1% dari total data, tetap perlu diperhatikan untuk menjaga integritas analisis. Dari sisi penggunaan memori, dataset ini memiliki ukuran total sekitar 882 KB, dengan rata-rata ukuran per baris sebesar 129 byte. Mengenai tipe data, terdapat distribusi yang cukup seimbang: 2 variabel bertipe teks, 4 bertipe numerik, 8 bertipe kategorikal, dan 3 bertipe boolean. Variasi tipe data ini menunjukkan bahwa pendekatan analisis dan praproses perlu disesuaikan secara khusus untuk masing-masing jenis fitur, terutama dalam konteks transformasi data dan pemilihan algoritma pemodelan. dengan diketahui beberapa kolom yang imbalanced dan beberapa bernilai null.


- Distribusi Nilai numerik.

Distribusi nilai numerik seperti gambar 1 adalah distribusi atau distribusi nilai variabel dalam satu set data, menggambarkan frekuensi atau probabilitas nilai -nilai ini yang didistribusikan.
  Gambar 1. Distribusi Nilai Numerik

  [![HL19XAN.md.png](https://iili.io/3LLSgrF.md.png)](https://freeimage.host/i/3LLSgrF)

Fitur userRatingCount dan adSpent memiliki distribusi yang sangat condong ke kanan (right-skewed), yang berarti sebagian besar aplikasi memiliki jumlah ulasan dan anggaran iklan yang rendah, sementara hanya sedikit aplikasi yang memiliki nilai sangat tinggi. Ini mengindikasikan keberadaan outlier pada kedua fitur tersebut. Fitur isCorporateEmailScore menunjukkan distribusi yang sangat sempit di kisaran nilai tinggi, khususnya mendekati 100, yang menandakan bahwa mayoritas aplikasi berasal dari alamat email yang memiliki skor tinggi dalam identifikasi sebagai email institusi atau korporat. Sementara itu, appAge memiliki distribusi yang lebih menyebar merata, dengan konsentrasi tertinggi pada usia sekitar 50 minggu, mencerminkan keragaman umur aplikasi dalam dataset. Fitur averageUserRating menampilkan pola diskrit dengan puncak pada nilai 0, 4, dan 5. Hal ini menunjukkan bahwa pengguna cenderung memberikan penilaian ekstrem—baik sangat rendah atau sangat tinggi—dan jarang memberikan nilai tengah. Informasi distribusi ini sangat penting untuk memahami skala, outlier, serta pola umum dalam data sebelum dilakukan pemodelan lebih lanjut.



- Correlation Matrix.

Correlation matrix seperti pada gambar 2 adalah tabel yang menampilkan koefisien korelasi yang mengukur kekuatan dan arah hubungan antara variabel.
  Gambar 2. Struktur data pada kolom Numerik

  [![HL19ePs.md.png](https://iili.io/3LLrU1S.md.png)](https://freeimage.host/i/3LLrU1S)

Gambar correlation matrix di atas menunjukkan hubungan linier antar fitur numerik dalam dataset. Nilai korelasi berkisar antara -1 hingga 1, yang mencerminkan kekuatan dan arah hubungan antar variabel. Terlihat bahwa fitur userRatingCount memiliki korelasi yang sangat kuat dengan adSpent, yaitu sebesar 0.87. Hal ini mengindikasikan bahwa semakin besar anggaran iklan suatu aplikasi, maka cenderung semakin banyak pula jumlah ulasan yang diterima aplikasi tersebut. Korelasi yang tinggi ini menunjukkan hubungan positif yang signifikan dan dapat menjadi informasi penting dalam pemodelan. Di sisi lain, fitur-fitur seperti isCorporateEmailScore, appAge, averageUserRating, dan coppaRisk memiliki korelasi rendah terhadap fitur lainnya, termasuk antar sesama mereka. Misalnya, isCorporateEmailScore memiliki korelasi negatif lemah dengan averageUserRating (-0.11), yang menunjukkan bahwa email dengan skor korporat yang tinggi tidak terlalu berhubungan dengan penilaian pengguna. Secara umum, sebagian besar fitur numerik menunjukkan hubungan yang lemah atau hampir tidak ada hubungan linier, kecuali antara userRatingCount dan adSpent.



## Data Preparation
_Data Preparation_ adalah tahapan penting dalam proses pemodelan yang melibatkan transformasi dan pembersihan data agar siap digunakan dalam pembuatan model. Berikut adalah tahapan atau urutan pada data preparation:

1. **Teknik Data cleaning & value replacement:**
   - Menggabungkan nilai tidak jelas pada `developerCountry` (seperti `'CANNOT IDENTIFY COUNTRY'`, `'ADDRESS NOT LISTED IN PLAYSTORE'`, `'STATUTORY MASKING ENABLED'`, dll.) menjadi satu nilai `'UNKNOWN'`.
   
   Tahapan tersebut dilakukan untuk menyederhanakan variasi nilai yang tidak informatif dan menghindari sparsity pada data kategorikal.

2. **Teknik Label Encoding:**
   - Mengubah kolom kategorikal menjadi angka menggunakan `LabelEncoder`, termasuk:
     - `developerCountry`
     - `primaryGenreName`
     - `countryCode`
     
     Encoding diperlukan agar fitur kategorikal bisa diolah sebagai input model.

3. **Teknik One-Hot Encoding:**
   - Menerapkan one-hot encoding pada kolom `deviceType` untuk mengonversi tipe perangkat menjadi kolom biner, contohnya:
     - `deviceType_smartphone`
     - `deviceType_tablet`
     - `deviceType_connected-tv/ott`, dll.
     
     Tujuannya untuk menghindari penciptaan urutan buatan pada data kategorikal nominal yang tidak memiliki hierarki

4. **Teknik Feature transformation:** 
   - Mengubah nilai teks seperti `"10,000 - 50,000"` menjadi angka dengan menghitung rata-rata dari rentang tersebut.
   
   Format teks tidak bisa digunakan dalam pemodelan. Tujuannya mengubahnya menjadi angka membuat fitur ini bisa dianalisis secara kuantitatif

5. **Teknik Binary Mapping:**  
    Konversi Boolean ke Numerik
   - Kolom seperti `hasPrivacyLink` dan `hasTermsOfServiceLink` dikonversi dari:
     - `True` → `1`
     - `False` → `0`
    
    Tujuan dari konversi ini membuat fitur logis bisa digunakan dalam model.

6. **Teknik Ordinal Encoding:**
   - Fitur dengan nilai ordinal dikonversi ke angka, misalnya:
     - `low` → `0`
     - `medium` → `1`
     - `high` → `2`
   - Berlaku pada kolom seperti:
     - `hasTermsOfServiceLinkRating`
     - `appContentBrandSafetyRating`
     - `appDescriptionBrandSafetyRating`
     - `mfaRating`

7. **Konversi Tipe Data:**
   - Menggunakan `pd.to_numeric()` untuk memastikan semua kolom numerik memiliki tipe data yang sesuai dan tidak mengandung teks/error.

8. **Teknik Imputasi Data**

    Imputasi nilai kosong (NaN) dalam dataset ini dilakukan dengan mempertimbangkan adanya ketidakseimbangan distribusi kelas (imbalanced data) pada beberapa fitur kategorikal. Untuk fitur seperti hasPrivacyLink dan isCorporateEmailScore, imputasi dilakukan dengan mengganti nilai yang hilang menggunakan nilai dari kelas minoritas. Pendekatan ini diambil untuk mencegah dominasi kelas mayoritas dalam distribusi data, yang dapat mempengaruhi hasil analisis dan model jika dibiarkan tidak seimbang. Sementara itu, pada fitur appContentBrandSafetyRating dan hasTermsOfServiceLink yang memiliki tiga kelas (0, 1, 2), dilakukan teknik imputasi acak tetapi tetap menjaga proporsi yang seimbang antar kelas. Nilai-nilai kosong pada kolom tersebut diisi secara acak dengan kelas 0, 1, dan 2 dalam jumlah yang sama atau sedekat mungkin, untuk memastikan tidak ada kelas yang mendominasi secara tidak wajar setelah proses imputasi. Strategi ini penting untuk mempertahankan representasi yang adil dari setiap kategori dalam data, sehingga model yang dibangun nantinya tidak bias terhadap kelas tertentu.



## Modeling

Tahap modeling ini menjadi salah satu tahap dalam membangun model. Model bertujuan untuk memprediksi Risiko Coppa Risk pada suatu aplikasi. Model yang digunakan adalah Deep Learning menggunakan neural network dan juga machine learning menggunakan Support Vector Machine (SVM). Pada tahap ini, data dilatih sehingga membentuk model. Evaluasi akan menggunakan confusion matrix karena bentuk dataset dan variabel target adalah berbentuk klasifikasi. 

### SVM 
 SVM Menggunakan SVC (Klasifikasi Vektor Dukungan) Perpustakaan Scikit Lear adalah bagian dari metode mesin vektor dukungan. SVC menemukan hyperveil terbaik yang memisahkan kelas data dari tepi terbesar.Data dari preprocessing (fungsi numerik) digunakan untuk melatih model SVM. Tidak ada konversi tambahan seperti tfidfvectionizers, karena fungsinya dalam bentuk numerik. Model dilatih dengan data pelatihan (x_train, y_train), diuji dengan data uji (X_TET) untuk mengukur kinerja. Evaluasi dilakukan dengan menggunakan matriks kebingungan dan akurasi metrik untuk menentukan kualitas klasifikasi.

 Tahapan:
 - Membangun pipeline SVC dengan kernel linear dan random state 42
 - Latih model SVM menggunakan svm_model.fit(X_train, y_train)
 - Prediksi menggunakan model SVM menggunakan y_pred_svm = svm_model.predict(X_test)

 Kekuatan: 
 - Efektif untuk Klasifikasi Biner: SVM sangat cocok untuk dua kelas kasus, seperti memprediksi risiko COPPA. 
 - Kemampuan untuk memproses data nonlinier: Menggunakan kernel 
 -  SVM dapat menetapkan data ke dimensi yang lebih tinggi untuk menemukan pemisah terbaik. 
 - Kuat untuk Pencilan kecil: Menggunakan parameter regulasi 
 - C, SVM dapat mengontrol tepi data yang sedikit terdistorsi. 
 
 
 Kelemahan: 
 - Data besar tidak efisien: Jika ukuran catatan data sangat besar, waktu pelatihan dan prediksi bisa lambat. 
 - Sensitif terhadap Pemilihan Parameter: Output model sangat tergantung pada pilihan nilai inti dan nilai parameter C, gamma, dll. 
  
### Deep Learning (Neural Network)

Model deep learning yang dibangun bertujuan untuk melakukan klasifikasi biner terhadap target coppaRisk. Sebelum model dilatih, dilakukan praproses berupa standardization menggunakan StandardScaler untuk memastikan bahwa setiap fitur numerik memiliki distribusi dengan rata-rata nol dan standar deviasi satu. Ini penting karena model neural network sangat sensitif terhadap skala data. Dataset kemudian dibagi menjadi data latih dan data uji dengan proporsi 80:20 menggunakan train_test_split. Proses ini penting untuk mengevaluasi kinerja model secara objektif pada data yang belum pernah dilihat. Model dikompilasi menggunakan optimizer Adam, yang merupakan algoritma adaptif populer karena efisiensi dan keandalannya. Loss function yang digunakan adalah binary crossentropy, sesuai dengan tugas klasifikasi dua kelas. Model dilatih selama 20 epoch dengan batch size 32 dan 10% dari data latih digunakan sebagai validation split untuk memantau performa selama pelatihan.

Tahapan:

Model deep learning yang digunakan dibangun menggunakan pendekatan sequential dan dirancang untuk menyelesaikan tugas klasifikasi biner terhadap variabel target coppaRisk. Arsitektur model terdiri dari beberapa lapisan yang disusun secara bertahap. Lapisan pertama adalah layer dense dengan 128 neuron dan fungsi aktivasi ReLU (Rectified Linear Unit), yang dipilih karena kemampuannya mengatasi masalah vanishing gradient dan efisiensi komputasinya. Untuk mengurangi risiko overfitting, diterapkan teknik regularisasi berupa Dropout dengan rasio 0.3 setelah layer ini. Lapisan kedua juga merupakan layer dense, kali ini dengan 64 neuron dan fungsi aktivasi ReLU yang sama, diikuti kembali oleh dropout dengan tingkat yang sama. Selanjutnya, terdapat layer ketiga dengan 32 neuron dan aktivasi ReLU, tanpa dropout, yang berfungsi untuk menangkap pola representasi yang lebih ringkas. Terakhir, model ditutup dengan sebuah output layer yang memiliki satu neuron dan menggunakan fungsi aktivasi sigmoid, yang sangat cocok untuk kasus klasifikasi biner karena menghasilkan output dalam bentuk probabilitas antara 0 dan 1. Kombinasi struktur berlapis dan penggunaan ReLU serta sigmoid sebagai fungsi aktivasi memungkinkan model untuk belajar representasi non-linear yang kompleks, sekaligus menjaga generalisasi melalui dropout.

- Membangun model neural network menggunakan `Sequential API` dari TensorFlow/Keras. Model terdiri dari beberapa layer `Dense` (fully connected) dengan fungsi aktivasi `ReLU` dan `sigmoid`.
- Layer pertama dan kedua berfungsi sebagai *hidden layer* yang menangkap representasi fitur secara non-linear. Layer terakhir (output) menggunakan aktivasi `sigmoid` karena target adalah klasifikasi biner (risiko atau tidak).
- Model dikompilasi menggunakan optimizer `Adam`, fungsi loss `binary_crossentropy`, dan metrik `accuracy`.
- Pelatihan dilakukan menggunakan `model.fit` pada data latih selama sejumlah epoch (20 epoch), dengan `batch_size` tertentu, serta validasi pada data uji untuk memantau performa.

Paramater yang digunakan pada deep learning
- epochs=20
- batch_size=32
- optimizer=adam
- loss=binary_crossentropy
- Layer konfigurasi (Dense, Dropout)

Kelebihan:
- **Mampu memodelkan relasi kompleks**: Neural network sangat baik dalam menangkap hubungan non-linear antar fitur, yang mungkin sulit ditangkap oleh model klasik.
- **Fleksibel dan dapat ditingkatkan**: Struktur model bisa dengan mudah disesuaikan (jumlah layer, neuron, dll.) sesuai kebutuhan dan kompleksitas data.
- **Kinerja meningkat dengan banyak data**: Model deep learning cenderung memberikan hasil lebih baik seiring bertambahnya jumlah data pelatihan.

Kekurangan:
- **Waktu pelatihan lebih lama**: Dibanding model tradisional seperti SVM atau regresi, deep learning memerlukan sumber daya komputasi lebih tinggi dan waktu pelatihan yang lebih lama.
- **Lebih sulit diinterpretasikan**: Model deep learning dikenal sebagai “black box” karena sulit untuk menjelaskan secara langsung bagaimana keputusan dibuat oleh jaringan.
- **Mudah overfitting jika data terbatas**: Tanpa teknik regularisasi atau augmentasi data, model neural network dapat terlalu menyesuaikan data latih dan kehilangan generalisasi.

Proyek ini menggunakan dua algoritma pada solution statement, model terbaik sebagai solusi. SVM sebagai model terbaik.
Karena SVM sangat efektif untuk dataset yang memiliki banyak fitur, termasuk hasil dari proses seperti one-hot encoding atau TF-IDF. Jika data mengandung banyak kolom fitur, SVM tetap dapat bekerja dengan baik.
 
## Evaluation
Dalam proyek ini, metrik evaluasi yang digunakan adalah melihat dari _Accuracy Score dan Confussion Matrix_. 

- Accuracy

Akurasi adalah metrik evaluasi untuk mengevaluasi pengukuran frekuensi pengukuran frekuensi yang membuat model membuat perkiraan yang akurat. Secara matematis, akurasi dihitung dengan rasio antara nomor prediksi yang akurat dan total. [Referensi](https://developers.google.com/machine-learning/glossary#accuracy)

Berikut rumus dari accuracy: 

[![3LPzAhb.md.png](https://iili.io/3LtRDan.md.png)](https://freeimage.host/i/3LtRDan)
  

Hasil dari modeling didapatkan evaluasi seperti tabel 1 dibawah ini. 

Tabel 1. Evaluasi Deep Learning dan SVM

|Model Name        |Accuracy Train |Accuracy Test|
|------------------|---------------|-------------|
|Deep Learning     |0.9027         |0.9000       |
|SVM               |0.8993         |0.9100       |



- Confussion matrix

Confussion Matrix adalah tabel yang digunakan dalam masalah klasifikasi untuk menilai di mana kesalahan dalam model dibuat. Baris-barisnya mewakili kelas-kelas aktual yang seharusnya. Sedangkan kolom-kolomnya mewakili prediksi yang telah kita buat. Dengan menggunakan tabel ini, mudah untuk melihat prediksi mana yang salah. [Referensi](https://www.w3schools.com/python/python_ml_confusion_matrix.asp)

Selain itu, confussion matrix, berfungsi dengan membandingkan hasil prediksi model klasifikasi label aktual dalam bentuk tabel 2x2 (untuk kasus klasifikasi biner) atau penerbit (untuk beberapa lapisan). Tabel ini menunjukkan prediksi yang akurat dan buruk untuk setiap kelas, termasuk

- True Positive (TP): Model memprediksi positif dan benar (label sebenarnya juga positif).

- True Negative (TN): Model memprediksi negatif dan benar (label sebenarnya juga negatif).

- False Positive (FP): Model memprediksi positif, tapi salah (label sebenarnya negatif) 

- False Negative (FN): Model memprediksi negatif, tapi salah (label sebenarnya positif)


  Gambar 3.

  [![3LPzAhb.md.png](https://iili.io/3LPzAhb.md.png)](https://freeimage.host/i/3LPzAhb)
  
  Gambar 4.

  [![HL19NoX.md.png](https://iili.io/3LPROmP.md.png)](https://freeimage.host/i/3LPROmP)


Diatas merupakan gambar 3 dan 4 sebagai confusion matrix. Dari gambar tersebut dapat dilihat bahwa banyak kasus risiko yang berhasil dikenali dengan benar (TP) dan yang terlewatkan (FN). Karena ini adalah problem klasifikasi biner risiko, maka False Negative (FN) adalah hal yang perlu diminimalkan dengan catatan yang berisiko (True) tidak boleh sampai diklasifikasikan sebagai tidak berisiko (false).Jadi berdasarkan akurasi dan confussion matrix. Deep Learning memiliki nilai akurasi tes sebesar 90% sedangkan SVM sebesar 91%. Sehingga SVM dapat dikatakan memiliki performa sedikit lebih baik dibanding menggunakan Deep Learning (Neural Network).

Model klasifikasi yang dibangun menggunakan Deep Learning dan SVM berhasil menjawab permasalahan utama, yaitu mengidentifikasi aplikasi dengan risiko tinggi atau rendah terhadap pelanggaran COPPA. Kedua model menunjukkan kemampuan dalam mengenali pola risiko, dengan Deep Learning unggul dalam menangani pola kompleks dan SVM sebagai pembanding yang kuat untuk data yang terstandarisasi. Proses pembersihan data—melalui imputasi, penghapusan duplikasi, dan standardisasi—berhasil meningkatkan kualitas data sehingga mendukung kinerja model secara optimal. Evaluasi menggunakan metrik akurasi dan confusion matrix menunjukkan hasil yang memuaskan, serta memperkuat bahwa solusi yang diterapkan berdampak nyata pada tujuan bisnis, yaitu membantu pengembang dalam mengelola kepatuhan aplikasi terhadap regulasi COPPA secara otomatis dan efisien.



## Conclusion
 
 Proses dimulai dengan memahami masalah, eksplorasi dan proses data pada tahap pemodelan dan evaluasi, menunjukkan fitur -fitur seperti  kategori aplikasi, rating, dan lain-lain yang berperan penting dalam prediksi tingkat risiko. Agar data menjadi bersih dilakukan tahapan Teknik Data cleaning & value replacement, Teknik Label Encoding, Teknik One-Hot Encoding,  Teknik Feature transformation, Teknik Binary Mapping, Teknik Ordinal Encoding, dan konversi tipe data. Setelah itu dilakukan modeling dengan hasil evaluasi bahwa walaupun Deep Learning terlihat bagus di data train, SVM lebih unggul di generalisasi. Maka, SVM adalah pilihan terbaik untuk model final.

## References
[1]	I. Reyes et al., “‘Won’t Somebody Think of the Children?’ Examining COPPA Compliance at Scale,” Proceedings on Privacy Enhancing Technologies, vol. 2018, no. 3, pp. 63–83, Jun. 2018, doi: 10.1515/popets-2018-0021.

[2]	J. Warmund, “Can COPPA Work? An Analysis of the Parental Consent Measures Can COPPA Work? An Analysis of the Parental Consent Measures in the Children’s Online Privacy Protection Act in the Children’s Online Privacy Protection Act.” [Online]. Available: https://ir.lawnet.fordham.edu/iplj/vol11/iss1/7

[3]	S. Finnegan, “FINN EGA N-FOR MAT TE D (DO NOT DELETE) HOW FACEBOOK BEAT THE CHILDREN’S ONLINE PRIVACY PROTECTION ACT: A LOOK INTO THE CONTINUED INEFFECTIVENESS OF COPPA AND HOW TO HOLD SOCIAL MEDIA SITES ACCOUNTABLE IN THE FUTURE.” [Online]. Available: https://www.commonsensemedia.org/sites/de

[4]	D. S. Skowronski, “COPPA and Educational Technologies: The Need for Additional COPPA and Educational Technologies: The Need for Additional Online Privacy Protections for Students Online Privacy Protections for Students.” [Online]. Available: https://readingroom.law.gsu.edu/gsulr/vol38/iss4/12