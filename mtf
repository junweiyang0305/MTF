import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
 
def smooth_signal(signal, window_len=1): 
    window = np.ones(window_len) / window_len 
    return np.convolve(signal, window, mode='same') 
 
def calculate_mtf_from_edge(image_path, roi=None, smoothing_window=5): 
 
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    if img is None: 
        raise FileNotFoundError(f"無法載入影像：{image_path}") 
 
    if roi is not None: 
        x, y, w, h = roi 
        img = img[y:y+h, x:x+w] 
  
    esf = np.mean(img, axis=0) 
  
    esf_smooth = smooth_signal(esf, window_len=smoothing_window) 
 
    lsf = np.gradient(esf_smooth) 
 
    otf = np.fft.fft(lsf) 
    mtf = np.abs(otf) 
    freqs = np.fft.fftfreq(len(lsf), d=1.0) 
    pos = freqs >= 0 
    mtf_norm = mtf[pos] / mtf[pos][0] 
    return freqs[pos], mtf_norm 
def visualize_and_plot_mtf(image_path, roi, smoothing_window=5): 
    img_color = cv2.imread(image_path) 
    if img_color is None: 
        raise FileNotFoundError(f"無法載入影像：{image_path}") 
    x, y, w, h = roi 
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 3) 
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB) 
    plt.figure(figsize=(8, 5)) 
    plt.imshow(img_rgb) 
    plt.axis('off') 
    plt.title('ROI 區域') 
    plt.show() 
    freqs, mtf = calculate_mtf_from_edge(image_path, 
                                          roi=roi, 
                                          smoothing_window=smoothing_window) 
    plt.figure() 
    plt.plot(freqs, mtf, '-o', markersize=4) 
    plt.xlabel('Spatial Frequency (cycles/pixel)') 
    plt.ylabel('Normalized MTF') 
    plt.title('MTF Curve') 
    plt.grid(True) 
    plt.show() 
if __name__ == '__main__': 
    image_path = 'Metal_1_3.bmp' 
    roi = (430, 550, 50, 60)    
    visualize_and_plot_mtf(image_path, roi, smoothing_window=7)
