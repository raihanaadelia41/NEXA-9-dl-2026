# Catatan model 123450015 (Transfer Learning — ResNet-18)

## Alasan optimizer
* **Optimizer yang dipilih:** `Adam` (`lr=0.001`, `weight_decay=1e-4`).
* **Alasan pemilihan:** 
  Pada skema *Transfer Learning* dengan backbone yang dibekukan (*frozen*), hanya layer klasifikasi akhir (`Linear` 512 -> 6) yang dilatih. Optimizer Adam dipilih karena memiliki laju pembelajaran adaptif (*adaptive learning rate*) per parameter dengan estimasi momen pertama dan kedua, sehingga konvergensi tercapai lebih cepat dan stabil pada ruang parameter kecil dibandingkan SGD standar.
* **Scheduler:** `StepLR` (step_size=5, gamma=0.5) untuk menurunkan laju pembelajaran secara bertahap saat pelatihan mendekati epoch akhir agar tidak *overshooting*.

## Regularisasi
1. **Backbone Freezing:** Membekukan seluruh parameter ekstraktor fitur konvolusional ResNet-18 bawaan (`param.requires_grad = False`) untuk mencegah rusaknya bobot representasi visual yang sudah dipelajari (*catastrophic forgetting*) dan mencegah overfitting pada data berukuran kecil.
2. **Weight Decay (L2 Regularization):** Ditambahkan sebesar `1e-4` pada optimizer Adam untuk membatasi magnitudo bobot matriks linier classifier.
3. **Normalisasi Min-Max Per Sampel:** Menjaga rentang nilai matriks masukan agar terdistribusi dalam interval `[0, 1]` sebelum di-interpolasi ke resolusi citra 64×64.

## Pencarian hyperparameter (tabel)

| Percobaan | Arsitektur | Learning Rate | Batch Size | Epoch | Scheduler | Keterangan / Hasil |
|---|---|---|---|---|---|---|
| H1 | ResNet-18 (frozen) | 0.01 | 32 | 15 | Tanpa scheduler | LR terlalu besar, loss berfluktuasi |
| H2 | ResNet-18 (frozen) | 0.0001 | 32 | 20 | Tanpa scheduler | Konvergensi sangat lambat |
| **H3 (Terpilih)** | **ResNet-18 (frozen)** | **0.001** | **32** | **20** | **StepLR (step=5, $\gamma$=0.5)** | **Loss turun mulus dan stabil** |

## Ablation (>= 2)

1. **Ablation 1: Pembekuan Bobot Backbone (*Frozen* vs *Full Fine-Tuning*)**
   * *Uji:* Membandingkan pelatihan hanya pada head `model.fc` versus melatih seluruh layer ResNet-18 dari awal.
   * *Temuan:* Melatih seluruh layer pada sampel kecil/simulasi menyebabkan overfitting parah dan durasi komputasi meningkat drastis. Pembekuan backbone menjaga efisiensi komputasi (~3–5 detik per seed) dan stabilitas gradien.
2. **Ablation 2: Normalisasi Min-Max Per Sampel vs Tanpa Normalisasi**
   * *Uji:* Membandingkan matriks sekuensial yang dinormalisasi ke `[0, 1]` sebelum dibesarkan ke 64×64 vs nilai mentah.
   * *Temuan:* Tanpa normalisasi min-max, magnitudo input yang bervariasi menyebabkan gradien awal terlalu besar dan fluktuasi loss pada epoch-epoch awal.

## Hasil 3 seed konfigurasi akhir

*(Hasil pengujian pipeline pada data simulasi Sprint 0)*

| Konfigurasi | Seed | val_acc | val_macro_f1 | Durasi (detik) | Perangkat |
|---|:---:|:---:|:---:|:---:|:---:|
| ResNet-18 + Adam (lr=1e-3) | 1 | 0.25 | 0.180912 | 4.1 | CPU Laptop |
| ResNet-18 + Adam (lr=1e-3) | 2 | 0.10 | 0.061111 | 4.0 | CPU Laptop |
| ResNet-18 + Adam (lr=1e-3) | 3 | 0.30 | 0.250000 | 4.1 | CPU Laptop |
| **Rerata ± Std** | - | **0.217 ± 0.104** | **0.164 ± 0.095** | **4.1 ± 0.1 s** | - |
