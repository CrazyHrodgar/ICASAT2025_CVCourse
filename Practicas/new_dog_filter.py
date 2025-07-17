import cv2
import numpy as np

def overlay_image_alpha(img, overlay, x, y, w, h):
    """
    Superpone `overlay` (BGRA) sobre `img` (BGR) en la posición (x,y)
    con tamaño (w,h), respetando recortes y suavizando bordes vía canal alfa.
    """
    # 1) Redimensionar overlay
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    # 2) Separar canales B,G,R y alfa
    b, g, r, a = cv2.split(overlay_resized)
    overlay_rgb = cv2.merge((b, g, r))
    # 3) Suavizar máscara alfa y normalizar [0,1]
    alpha = a.astype(float) / 255.0
    alpha = cv2.GaussianBlur(alpha, (7,7), 0)
    alpha = np.dstack((alpha, alpha, alpha))
    # 4) Calcular región válida en la imagen destino
    h_img, w_img = img.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    if x1 >= x2 or y1 >= y2:
        return  # fuera de rango
    # 5) Coordenadas dentro del overlay redimensionado
    ov_x1, ov_y1 = x1 - x, y1 - y
    ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)
    overlay_crop = overlay_rgb[ov_y1:ov_y2, ov_x1:ov_x2]
    alpha_crop   = alpha[ov_y1:ov_y2, ov_x1:ov_x2]
    # 6) Extraer ROI y mezclar
    roi = img[y1:y2, x1:x2].astype(float) / 255.0
    blended = alpha_crop * (overlay_crop.astype(float) / 255.0) + (1 - alpha_crop) * roi
    img[y1:y2, x1:x2] = (blended * 255).astype(np.uint8)


# --- Carga de recursos ---
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
dog_filter = cv2.imread('dog_face_full.png', cv2.IMREAD_UNCHANGED)
if dog_filter is None or dog_filter.shape[2] != 4:
    raise IOError("Asegúrate de tener 'dog_filter.png' con canal alfa (BGRA).")

# --- Configuración de ventana y sliders ---
cv2.namedWindow('Filtro Perrito')
# scale_w y scale_h: 100→1.00, 300→3.00
cv2.createTrackbar('scale_w', 'Filtro Perrito', 130, 300, lambda v: None)
cv2.createTrackbar('scale_h', 'Filtro Perrito', 130, 300, lambda v: None)
# offset_x y offset_y: 0→-1.00, 100→0.00, 200→+1.00
cv2.createTrackbar('offset_x', 'Filtro Perrito', 100, 200, lambda v: None)
cv2.createTrackbar('offset_y', 'Filtro Perrito',  50, 200, lambda v: None)

# --- Captura de vídeo ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("No se puede acceder a la cámara.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Leer parámetros de los sliders
    sw = cv2.getTrackbarPos('scale_w',  'Filtro Perrito') / 100.0
    sh = cv2.getTrackbarPos('scale_h',  'Filtro Perrito') / 100.0
    ox = (cv2.getTrackbarPos('offset_x','Filtro Perrito') - 100) / 100.0
    oy = (cv2.getTrackbarPos('offset_y','Filtro Perrito') - 100) / 100.0

    # 2) Detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # 3) Aplicar filtro a cada cara
    for (x, y, w, h) in faces:
        fw = int(w * sw)
        fh = int(h * sh)
        fx = x + int(ox * w)
        fy = y + int(oy * h)
        overlay_image_alpha(frame, dog_filter, fx, fy, fw, fh)

    # 4) Mostrar resultado
    cv2.imshow('Filtro Perrito', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
