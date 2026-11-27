# PROTOKOL PENGAMBILAN DATA: TIM 9

**Arah Gerak Keluar-Masuk di Depan Meja**
Pilar: Gerak | Pembimbing: Ahmad Luky Ramdani
Rencana: `rencana/t09-arah-gerak.yaml` | Seed: 1009 | Ulangan: 3

---

## 1. Ringkasan sesi

| Butir | Nilai |
|---|---|
| Tujuan | Merekam data untuk mengenali 6 lintasan di depan meja dari pose kamera, PIR, dan sensor pintu kotak |
| Slot operator | Kamis, 12 November 2026, 08.00-09.00 |
| Partisipan | 6 orang (P01-P06), satu partisipan = satu blok |
| Kondisi per blok | 18 (6 lintasan x 3 ulangan, diacak) |
| Total kondisi | 108 |
| Durasi tiap kondisi | 8 detik |
| Jeda antarkondisi | 4 detik (alat mati; partisipan pindah ke posisi awal berikutnya) |
| Perekam aktif | Kamera 640x480 setiap 0,1 detik; sensor ESP32 (getaran, arus, suhu, kelembapan, lux, PIR, pintu) |
| Perkiraan waktu | 108 x 12 detik = sekitar 21,6 menit rekam, ditambah sekitar 6 menit untuk 6 kali pergantian blok |

Satu detik pertama setiap kondisi dibuang pada tahap analisis karena alat atau partisipan masih dalam peralihan. Partisipan tetap bergerak seperti biasa selama 8 detik penuh.

---

## 2. Pembagian peran

| No | Nama | NIM | Peran |
|---|---|---|---|
| 1 | Eka Fidiya Putri | 122450045 | Operator Perekam |
| 2 | Lidia Natasyah Marpaung | 123450015 | Operator Alat |
| 3 | Raihana Adelia Putri | 123450041 | Koordinator Partisipan (Penjaga Bingkai/Suara) |
| 4 | Aisyah Musfirah | 123450084 | Pencatat Kejadian |

### 2.1 Operator Perekam (Eka Fidiya Putri)
- Menjalankan skrip `meja.py` dan memastikan rencana yang dimuat adalah versi `jeda_detik: 4`.
- Pada layar "Siapkan: Partisipan Pxx siap di posisi", menekan **Enter** hanya setelah Koordinator Partisipan menyatakan partisipan siap.
- Memantau layar selama kondisi berjalan (skrip maju sendiri tanpa Enter) dan memastikan hitungan mencapai 108 kondisi.
- Memastikan `manifest.json` terbentuk di akhir sesi.

### 2.2 Operator Alat (Lidia Natasyah Marpaung)
- Menyiapkan dan memeriksa kamera, sensor ESP32 (terutama PIR dan pintu), serta pintu kotak sebelum slot.
- Memeriksa pratinjau kamera agar seluruh bagian tubuh yang dibutuhkan masuk bingkai.
- Menutup kembali pintu kotak pada jeda setelah kondisi "berhenti di meja".
- Mengembalikan meja ke keadaan acuan setelah slot.

### 2.3 Koordinator Partisipan, Penjaga Bingkai/Suara (Raihana Adelia Putri)
- Memanggil partisipan, memastikan formulir persetujuan sudah ditandatangani, dan memastikan partisipan sudah di posisi sebelum Enter ditekan.
- Membacakan definisi lintasan (bagian 3) dengan kalimat yang sama kepada setiap partisipan.
- Segera setelah layar berganti, memberi tahu **posisi awal** untuk kondisi berikutnya (luar bingkai kiri, luar bingkai kanan, atau depan meja).
- Menjaga bingkai: memastikan partisipan tidak keluar jalur dan tidak menutupi sendi (topi, masker, jaket bertudung).
- Menjaga suara: ruang tenang dan tidak ada orang lain yang masuk bingkai kamera atau melintas di depan sensor PIR selama kondisi berjalan.

### 2.4 Pencatat Kejadian (Aisyah Musfirah)
- Mengisi lembar sesi: tanggal, jam mulai dan selesai, partisipan per blok, dan siapa pun yang hadir.
- Mencatat **nomor kondisi yang terganggu** (lihat bagian 6).
- Memotret lembar sesi setelah slot dan menyerahkannya ke repositori tim.

> Catatan: pembagian tugas rinci di atas disusun dari isi dokumen tim. Cocokkan dengan pembagian peran di `00-metode-eksperimen-lapangan.md` dan sesuaikan bila ada perbedaan.

---

## 3. Definisi operasional lintasan

Bacakan kalimat berikut **persis sama** kepada setiap partisipan. Layar menampilkan instruksi seperti `LAKUKAN: MASUK DARI KIRI`; skrip maju sendiri tanpa Enter.

| Lintasan | Definisi operasional |
|---|---|
| **Masuk dari kiri** | Mulai di luar bingkai pada sisi kiri, lalu berjalan ke depan meja. |
| **Keluar ke kiri** | Mulai di depan meja, lalu berjalan keluar bingkai ke sisi kiri. |
| **Masuk dari kanan** | Mulai di luar bingkai pada sisi kanan, lalu berjalan ke depan meja. |
| **Keluar ke kanan** | Mulai di depan meja, lalu berjalan keluar bingkai ke sisi kanan. |
| **Melintas tanpa berhenti** | Berjalan dari satu sisi ke sisi lain tanpa berhenti di meja. |
| **Berhenti di meja** | Datang ke meja dan membuka pintu kotak sebagai tanda berhenti. |

Pegangan umum untuk semua lintasan:
- Berjalan dengan kecepatan wajar, tidak berlari.
- Satu lintasan dilakukan dalam 8 detik yang ditampilkan di layar.
- Jangan hanya berdiri di posisi awal lalu langsung ke posisi akhir; bergeraklah sepanjang kondisi.

### 3.1 Posisi awal per lintasan

| Lintasan | Posisi awal | Selotip |
|---|---|---|
| Masuk dari kiri | Luar bingkai kiri | Warna A |
| Masuk dari kanan | Luar bingkai kanan | Warna B |
| Keluar ke kiri | Depan meja | Warna C |
| Keluar ke kanan | Depan meja | Warna C |
| Melintas tanpa berhenti | Satu sisi (kiri atau kanan, sesuai arahan koordinator) | Warna A atau B |
| Berhenti di meja | Luar bingkai kiri atau kanan, sesuai arahan koordinator | Warna A atau B |

> Isi warna selotip sesuai yang dipakai tim. Untuk lintasan melintas dan berhenti di meja, tetapkan sisi awal bersama pembimbing sebelum slot dan terapkan sama untuk semua partisipan.

---

## 4. Persiapan sebelum slot

Dikerjakan paling lambat sebelum Kamis, 12 November 2026, 08.00.

### 4.1 Perangkat lunak dan rencana
- [ ] Rencana yang dipakai adalah versi `jeda_detik: 4` dengan `seed: 1009` dan `ulangan: 3`.
- [ ] Ekstensi berkas sesuai perintah (`.yaml`; berkas bernama `.yml` perlu disamakan).
- [ ] Simulasi dijalankan: `python meja.py --simulasi jalankan ../rencana/t09-arah-gerak.yaml`
- [ ] Skrip mencetak **108 kondisi**.
- [ ] Pemotong jendela tim diuji pada folder sesi simulasi.
- [ ] Pembanding non-DL (statistik sudut sendi dan kecepatan titik pose + Random Forest) berjalan di data simulasi.

### 4.2 Partisipan
- [ ] Enam partisipan (P01-P06) siap, diutamakan dari luar tim.
- [ ] Formulir persetujuan untuk setiap partisipan sudah ditandatangani.
- [ ] Daftar ID samaran (P01-P06) sudah dibuat; nama asli tidak masuk ke data.
- [ ] Setiap partisipan diminta memakai pakaian yang **kontras dengan latar** dan melepas topi, masker, atau jaket bertudung yang menutupi sendi.

### 4.3 Perangkat dan tata ruang
- [ ] Titik mulai ditandai dengan selotip berwarna berbeda: luar bingkai kiri, luar bingkai kanan, dan depan meja.
- [ ] Pratinjau kamera diperiksa: seluruh tubuh yang dibutuhkan masuk bingkai pada ketiga titik mulai dan sepanjang jalur.
- [ ] Sensor PIR dan sensor pintu kotak menyala dan terbaca (uji buka-tutup pintu).
- [ ] Pintu kotak dalam keadaan **tertutup** sebelum rekaman dimulai.
- [ ] Lembar sesi dan pulpen siap di tangan Pencatat.

---

## 5. Alur saat slot

1. **Mulai blok.** Layar menampilkan, misalnya, `Siapkan: Partisipan P01 siap di posisi (formulir persetujuan sudah ditandatangani)`.
2. **Konfirmasi.** Koordinator Partisipan memastikan partisipan yang dipanggil sudah di posisi dan sudah mendengar definisi lintasan. Setelah itu Operator Perekam menekan **Enter**.
3. **Kondisi berjalan.** Layar menampilkan `LAKUKAN: <LINTASAN>` dengan huruf besar. Skrip maju sendiri; tidak ada Enter di antara kondisi.
4. **Posisi awal berikutnya.** Segera setelah layar berganti, Koordinator Partisipan menyebutkan posisi awal untuk kondisi berikutnya.
5. **Jeda 4 detik.** Alat mati. Partisipan pindah ke posisi awal. Untuk lintasan "berhenti di meja", Operator Alat menutup kembali pintu kotak pada jeda.
6. **Gangguan.** Pencatat menulis nomor kondisi yang terganggu pada lembar sesi (lihat bagian 6).
7. **Pergantian blok.** Setelah 18 kondisi, layar kembali ke "Siapkan". Ulangi langkah 1-6 untuk partisipan berikutnya. Urutan blok diacak (seed 1009), jadi ikuti urutan yang ditampilkan layar, bukan urutan P01 ke P06.

Total: 6 blok x 18 kondisi = **108 kondisi**.

---

## 6. Pencatatan gangguan

Pencatat menulis di lembar sesi: **nomor kondisi**, **partisipan**, **jenis gangguan**, dan **jam kejadian**. Contoh kejadian yang dicatat:

- Partisipan belum sampai di posisi awal saat kondisi dimulai (lintasan terpotong).
- Partisipan salah lintasan atau salah arah.
- Orang lain masuk bingkai atau melintas.
- Sensor pintu atau PIR tidak merespons.
- Kamera tersendat atau bingkai terhalang.
- Pintu kotak lupa ditutup pada jeda.
- Gangguan suara atau hal lain di luar rencana.

Kondisi terganggu **tidak diulang di tempat** kecuali disepakati bersama pembimbing; cukup dicatat dan diputuskan saat analisis apakah dibuang.

---

## 7. Pemeriksaan mutu data (hari yang sama)

- [ ] Setiap partisipan memiliki **18 kondisi** (6 lintasan x 3 ulangan).
- [ ] Total **108 kondisi** tercatat di `manifest.json`.
- [ ] Perubahan PIR terjadi pada hampir semua kondisi.
- [ ] Sensor pintu berubah pada kondisi "berhenti di meja" saja.
- [ ] Kamera merekam sekitar 80 frame per kondisi (8 detik, setiap 0,1 detik), 70 frame setelah detik pertama dibuang.
- [ ] Daftar kondisi terganggu sudah dikompilasi dari lembar sesi.

Bila ada kekurangan, laporkan ke pembimbing hari itu juga.

---

## 8. Setelah slot

1. Tunggu `manifest.json` terbentuk.
2. Data disalin oleh pengelola.
3. Meja dikembalikan ke keadaan acuan (pintu kotak, selotip, kabel, posisi alat).
4. Lembar sesi difoto.
5. Data dimasukkan ke repositori tim.

(Ikuti bagian 7 dokumen `00-metode-eksperimen-lapangan.md` untuk rincian lengkap.)

---

## 9. Aturan pemakaian data

- Pemisahan data uji **per partisipan**. Seluruh jendela dari satu partisipan berada pada lipatan yang sama dan tidak pernah terbagi antara data latih dan data uji.
- Normalisasi dihitung dari **data latih saja**.
- Data sekunder (pelacakan pejalan kaki) hanya pelengkap: pralatih, augmentasi, atau uji tambahan. Tabel komparasi utama, pemilihan jalur, dan uji purwarupa memakai **data uji meja**.
- Ujilah juga dengan **sebagian pose dihilangkan** agar model tidak hanya mengandalkan posisi awal dan akhir.

---

## 10. Jadwal penting

| Tanggal | Kegiatan |
|---|---|
| Kamis, 12 Nov 2026, 08.00-09.00 | Slot operator (pengambilan data) |
| Kamis, 19 Nov 2026 | Demo Sprint 1 |
| 20-27 Nov 2026 | Sprint 2 |
| Selasa, 24 Nov 2026, 11.30-12.00 | Uji purwarupa |

**Definition of Done Sprint 1:** setiap jalur (CNN, RNN, Transformer, AE) sudah dilatih pada data primer meja (minimal satu seed), pembanding non-DL berjalan, dan `metrics.json` valid.