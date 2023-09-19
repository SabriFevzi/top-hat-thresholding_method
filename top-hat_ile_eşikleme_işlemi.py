import cv2
import numpy as np
import matplotlib.pyplot as plt

# Girdi görüntüsünü yükle
input_image = cv2.imread('Rice.png', 0)  # Gri tonlamalı olarak yükle

# Disk şeklinde yapılandırma elemanlarını oluştur
radii = [3, 6, 9, 12, 15]
structuring_elements = [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) for r in radii]

# Top-hat işlemini uygula
tophat_results = []
for se in structuring_elements:
    # Açılım işlemi
    opening = cv2.morphologyEx(input_image, cv2.MORPH_OPEN, se)
    # Girdi görüntüsünden açılım sonucunu çıkar
    tophat = cv2.subtract(input_image, opening)
    tophat_results.append(tophat)

# Ortak bir eşik değeri belirle
threshold_value = 40
# Eşikleme işlemini uygula
thresholded_results = [cv2.threshold(tophat, threshold_value, 255, cv2.THRESH_BINARY)[1] for tophat in tophat_results]

# Bağlantılı bileşen analizi yap
total_rice_count = []
total_rice_area = []
average_rice_area = []

for i, thresholded in enumerate(thresholded_results):
    # Bağlantılı bileşenleri etiketle
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded, connectivity=8)

    # İlk bileşeni dışarıda tut (arka plan)
    num_labels -= 1

    # Bileşen analizi için toplam pirinç sayısı ve alanı hesapla
    rice_count = num_labels
    rice_area = np.sum(stats[1:, cv2.CC_STAT_AREA])

    total_rice_count.append(rice_count)
    total_rice_area.append(rice_area)

    # Ortalama pirinç alanını hesapla
    average_area = rice_area / rice_count
    average_rice_area.append(average_area)
    print(f"Top-hat (r={radii[i]}): Rice Count={rice_count}, Total Rice Area={rice_area}, Average Rice Area={average_area}")


# Grafik çizdirme
plt.figure(figsize=(10, 6))
plt.errorbar(radii, total_rice_count, yerr=np.sqrt(total_rice_count), fmt='o-', label='Rice Count')
plt.errorbar(radii, total_rice_area, yerr=np.sqrt(total_rice_area), fmt='o-', label='Total Rice Area')
plt.errorbar(radii, average_rice_area, yerr=np.sqrt(average_rice_area), fmt='o-', label='Average Rice Area')
plt.xlabel('Structuring Element Radius')
plt.ylabel('Measurements')
plt.title('Measurements vs Structuring Element Radius')
plt.legend()
plt.grid(True)
plt.show()

# Sonuçları görüntüle
for i, thresholded in enumerate(thresholded_results):
    cv2.imshow(f"Thresholded Top-hat (r={radii[i]})", thresholded)

cv2.waitKey(0)
cv2.destroyAllWindows()
