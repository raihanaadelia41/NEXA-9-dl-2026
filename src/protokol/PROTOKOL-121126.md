# Protokol Eksperimen Tim 9

1. **Pembuangan Awal**: Buang 1 detik pertama setiap kondisi karena alat atau partisipan masih dalam peralihan.
2. **Pemisahan Data Uji**: Data uji dipisah per partisipan (Grup). Seluruh jendela dari satu partisipan berada pada lipatan yang sama dan tidak terbagi antara data latih dan data uji (mencegah kebocoran data).
3. **Normalisasi**: Normalisasi dihitung **hanya** dari data latih, lalu diterapkan apa adanya pada data validasi dan uji.
4. **Metrik Utama**: Kinerja model diukur menggunakan **F1-makro**.
5. **Evaluasi Akhir**: Menggunakan tiga seed. Hasil akhir dilaporkan sebagai rata-rata ± simpangan dari 3 seed, beserta ukuran model ONNX (KB) dan latensi (ms) di laptop meja.
