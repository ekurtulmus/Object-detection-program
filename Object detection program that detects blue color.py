print("RENK İLE NESNE TESPİTİ")
import cv2
import numpy as np
from collections import deque

# nesne merkezini depolayacak veri tipi
buffer_size = 16
pts = deque(maxlen=buffer_size)

# mavi renk aralığı HSV formatında
blueLower = (84, 98, 0)
blueUpper = (179, 255, 255)

# Kamera yakalama
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)

while True:
    # Kamera görüntüsünü yakalama ve işleme
    success, imgOriginal = cap.read()
    if success:
        # Görüntüyü bulanıklaştırma
        blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0)

        # HSV renk uzayına dönüştürme
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV image", hsv)

        # Mavi için maske oluşturma
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        cv2.imshow("mask Image", mask)

        # Maskenin etrafındaki gürültüyü temizleme
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cv2.imshow("Mask+erozyon ve genişleme", mask)

        # Kontur bulma
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        if len(contours) > 0:
            # En büyük konturu al
            c = max(contours, key=cv2.contourArea)

            # Dikdörtgene çevirme
            rect = cv2.minAreaRect(c)
            ((x, y), (width, height), rotation) = rect
            s = "x:{}, y:{}, width:{}, rotation:{}".format(np.round(x), np.round(y), np.round(width), np.round(height), np.round(rotation))
            print(s)

            # Kutucuk oluşturma
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Moment hesaplama
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Konturu çizme
            cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)

            # Merkeze bir nokta çizme
            cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)

            # Bilgileri ekrana yazdırma
            cv2.putText(imgOriginal, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        # Sonuçları gösterme
        cv2.imshow("Orijinal Tespit", imgOriginal)

    # Çıkış tuşu kontrolü
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
