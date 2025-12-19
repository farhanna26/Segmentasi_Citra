import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# ======================================================
# 1. SETUP & FUNGSI UTILITAS
# ======================================================

def load_image(path, limit=300):
    img = cv2.imread(path)
    if img is None:
        print(f"WARNING: Tidak dapat membaca file {path}")
        return np.zeros((100, 100), dtype=np.uint8)
    
    # Langsung convert ke Grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape
    if max(h, w) > limit:
        scale = limit / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    return gray

def mse(imgA, imgB):
    if imgA.shape != imgB.shape:
        imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))
    
    err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
    err /= float(imgA.shape[0] * imgA.shape[1])
    return err

# ======================================================
# 2. NOISE & FILTER (Mean & Median)
# ======================================================

def apply_salt_pepper(img, ratio):
    noisy = img.copy()
    rand_map = np.random.rand(*noisy.shape)
    noisy[rand_map < ratio / 2] = 255
    noisy[rand_map > 1 - ratio / 2] = 0
    return noisy

def apply_gaussian(img, mu, sigma):
    gauss = np.random.normal(mu, sigma, img.shape)
    noisy = img + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def mean_filter(img, size=3):
    return cv2.blur(img, (size, size))

def median_filter(img, size=3):
    return cv2.medianBlur(img, size)

# ======================================================
# 3. METODE SEGMENTASI (4 METODE)
# ======================================================

def convolve(img, kernel_x, kernel_y):
    img = img.astype(float)
    gx = cv2.filter2D(img, -1, kernel_x)
    gy = cv2.filter2D(img, -1, kernel_y)
    mag = np.sqrt(gx**2 + gy**2)
    return np.clip(mag, 0, 255).astype(np.uint8)

def sobel(img):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    return convolve(img, kx, ky)

def prewitt(img):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    return convolve(img, kx, ky)

def roberts(img):
    h, w = img.shape
    res = np.zeros_like(img, dtype=float)
    img = img.astype(float)
    for y in range(h-1):
        for x in range(w-1):
            gx = img[y, x] - img[y+1, x+1]
            gy = img[y, x+1] - img[y+1, x]
            res[y, x] = np.sqrt(gx**2 + gy**2)
    return np.clip(res, 0, 255).astype(np.uint8)

def frei_chen(img):
    s2 = math.sqrt(2)
    kx = np.array([[-1, 0, 1], [-s2, 0, s2], [-1, 0, 1]])
    ky = np.array([[-1, -s2, -1], [0, 0, 0], [1, s2, 1]])
    return convolve(img, kx, ky)

seg_methods = {
    "Sobel": sobel,
    "Prewitt": prewitt,
    "Roberts": roberts,
    "Frei-Chen": frei_chen
}

# ======================================================
# 4. MAIN PROGRAM: EKSEKUSI GABUNGAN
# ======================================================

# A. LOAD GAMBAR & GROUND TRUTH
print("--- 1. MENYIAPKAN GAMBAR & KUNCI JAWABAN (GROUND TRUTH) ---")
# GANTI NAMA FILE DISINI
img_ori = load_image("Pemandangan.jpg") 

ground_truths = {}
for name, func in seg_methods.items():
    ground_truths[name] = func(img_ori)

# B. GENERATE NOISE
print("--- 2. MEMBUAT VARIASI NOISE ---")
noises = [
    ("SP 0.02 (Level 1)", apply_salt_pepper(img_ori, 0.02)),
    ("SP 0.10 (Level 2)", apply_salt_pepper(img_ori, 0.10)),
    ("Gauss 10 (Level 1)", apply_gaussian(img_ori, 0, 10)),
    ("Gauss 40 (Level 2)", apply_gaussian(img_ori, 0, 40))
]

data_mse = []

print("\n--- 3. MULAI PROSES VISUALISASI & PERHITUNGAN ---")
print("(Jendela gambar akan muncul satu per satu. Tutup jendela untuk melanjutkan ke gambar berikutnya)")

for noise_label, noisy_img in noises:
    for filter_name, filter_func in [("Mean", mean_filter), ("Median", median_filter)]:
        
        # 1. RESTORASI
        restored_img = filter_func(noisy_img)
        
        # List untuk visualisasi Grid Gambar
        display_list = []
        display_list.append((f"NOISY:\n{noise_label}", noisy_img))
        display_list.append((f"RESTORED:\nFilter {filter_name}", restored_img))
        
        # 2. SEGMENTASI & HITUNG MSE
        for seg_name, seg_func in seg_methods.items():
            # Lakukan segmentasi pada hasil restorasi
            seg_result = seg_func(restored_img)
            
            # Simpan gambar untuk ditampilkan
            display_list.append((f"Seg: {seg_name}", seg_result))
            
            # Hitung MSE (Bandingkan dengan Ground Truth)
            gt_img = ground_truths[seg_name]
            score = mse(gt_img, seg_result)
            
            data_mse.append({
                "Noise Type": noise_label,
                "Restoration Filter": filter_name,
                "Segmentation Method": seg_name,
                "MSE": score
            })
            
        # 3. TAMPILKAN GRID GAMBAR (VISUALISASI)
        plt.figure(figsize=(15, 9))
        plt.suptitle(f"Visualisasi: {noise_label}  -->  Filter {filter_name}", fontsize=16, fontweight='bold')
        
        # Grid 2 Baris x 3 Kolom
        for i, (label, img_show) in enumerate(display_list):
            plt.subplot(2, 3, i + 1)
            plt.imshow(img_show, cmap='gray')
            plt.title(label, fontsize=10, fontweight='bold')
            plt.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show() # Program akan berhenti di sini sampai user menutup window gambar

# ======================================================
# 5. OUTPUT AKHIR: GRAFIK & TABEL
# ======================================================

df = pd.DataFrame(data_mse)

# --- FUNGSI PRINT TABEL ---
def print_table(dataframe):
    print("\n" + "="*80)
    print("TABEL RANGKUMAN MSE (Akurasi Segmentasi)")
    print("="*80)
    dataframe['Scenario'] = dataframe['Noise Type'] + " + " + dataframe['Restoration Filter']
    pivot_tb = dataframe.pivot_table(index='Scenario', columns='Segmentation Method', values='MSE')
    print(pivot_tb)
    print("-" * 80)

# --- FUNGSI PLOT GRAFIK SATU PER SATU ---
def plot_charts(dataframe):
    unique_noises = dataframe['Noise Type'].unique()
    
    print("\nMenampilkan Grafik Analisis MSE Satu per Satu...")
    
    for noise in unique_noises:
        subset = dataframe[dataframe['Noise Type'] == noise]
        
        # Pivot Data
        pivot_chart = subset.pivot_table(index='Segmentation Method', columns='Restoration Filter', values='MSE')
        
        plt.figure(figsize=(10, 6))
        ax = pivot_chart.plot(kind='bar', figsize=(10, 6), width=0.7, color=['#ff9999', '#66b3ff'], edgecolor='black')
        
        plt.title(f"Analisis MSE Segmentasi pada: {noise}\n(Mana yang lebih akurat?)", fontsize=14, fontweight='bold')
        plt.ylabel("Nilai MSE (Lebih Rendah = Lebih Bagus)", fontsize=12)
        plt.xlabel("Metode Segmentasi", fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(title="Filter Restorasi")
        
        # Label angka
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', fontsize=10, padding=3)
            
        plt.tight_layout()
        plt.show()

# Jalankan Output Akhir
print_table(df)
plot_charts(df)

print("\nSelesai. Semua proses telah ditampilkan.")
