import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. FUNGSI DEFINISI KERNEL (TIDAK BERUBAH) ---
def get_kernels(method):
    if method == 'roberts':
        kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        ky = np.array([[0, -1], [1, 0]], dtype=np.float32)
    elif method == 'prewitt':
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    elif method == 'sobel':
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    elif method == 'freichen':
        sqrt2 = np.sqrt(2)
        kx = np.array([[-1, 0, 1], [-sqrt2, 0, sqrt2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -sqrt2, -1], [0, 0, 0], [1, sqrt2, 1]], dtype=np.float32)
    else:
        return None, None
    return kx, ky

# --- 2. FUNGSI EDGE DETECTION (TIDAK BERUBAH) ---
def edge_detection(img, method):
    img_float = img.astype(np.float32)
    kx, ky = get_kernels(method)
    ix = cv2.filter2D(img_float, -1, kx)
    iy = cv2.filter2D(img_float, -1, ky)
    magnitude = np.sqrt(ix*2 + iy*2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude

# --- 3. FUNGSI NOISE (LEVEL 1 & 2) ---
def add_salt_pepper(img, level):
    ratio = 0.02 if level == 1 else 0.05
    noisy = img.copy()
    
    num_salt = np.ceil(ratio * img.size * 0.5)
    coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[tuple(coords_salt)] = 255
    
    num_pepper = np.ceil(ratio * img.size * 0.5)
    coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[tuple(coords_pepper)] = 0
    return noisy

def add_gaussian(img, level):
    sigma = 10 if level == 1 else 20
    gauss = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

# --- 4. LOAD & PROCESS ---

# Ganti dengan nama file kamu
IMG_PATH = 'Greyscale.png' 
img_original = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

if img_original is None:
    print(f"ERROR: Gambar '{IMG_PATH}' tidak ditemukan. Mohon ganti nama file.")
else:
    # Resize opsional
    img_original = cv2.resize(img_original, (400, 300))

    images = {
        "Original Grayscale": img_original,
        "Salt & Pepper Lvl 1": add_salt_pepper(img_original, level=1),
        "Salt & Pepper Lvl 2": add_salt_pepper(img_original, level=2),
        "Gaussian Lvl 1 (Sig 10)": add_gaussian(img_original, level=1),
        "Gaussian Lvl 2 (Sig 20)": add_gaussian(img_original, level=2),
    }

    methods = ['roberts', 'prewitt', 'sobel', 'freichen']
    
    num_images = len(images)
    num_cols = len(methods) + 1 

    # --- SETTING VISUALISASI ---
    # facecolor='black' agar background hitam
    plt.figure(figsize=(18, 4 * num_images), facecolor='black')
    
    idx = 1
    for img_name, img_data in images.items():
        # Plot Input
        plt.subplot(num_images, num_cols, idx)
        plt.imshow(img_data, cmap='gray')
        # color='lime' agar tulisan jadi HIJAU CERAH (Neon)
        plt.title(f"Input: {img_name}", fontsize=10, color='lime', fontweight='bold') 
        plt.axis('off')
        idx += 1

        # Plot Hasil Deteksi Tepi
        for method in methods:
            result = edge_detection(img_data, method)
            
            plt.subplot(num_images, num_cols, idx)
            plt.imshow(result, cmap='gray')
            # color='lime' agar tulisan jadi HIJAU CERAH (Neon)
            plt.title(f"{method.capitalize()}", fontsize=10, color='lime', fontweight='bold')
            plt.axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()