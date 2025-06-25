import cv2
import numpy as np
import os
import base64
from pyzbar.pyzbar import decode
from PIL import Image, ImageFile
from flask import Flask, request, jsonify
import requests
from io import BytesIO

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Permite abrir imagens corrompidas

# --- 1. Função para Detecção de Desfoque em memória ---
def is_blurry_in_memory(image_np: np.ndarray, blur_threshold: float = 5.0) -> tuple[bool, float]:
    if image_np is None or image_np.size == 0:
        return True, 0.0
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian_variance = cv2.Laplacian(processed_gray, cv2.CV_64F).var()
    return laplacian_variance < blur_threshold, laplacian_variance

# --- 2. Função para Ler QR Code ---
def read_qr_code(image_np: np.ndarray) -> str | None:
    if image_np is None or image_np.size == 0:
        return None
    img_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    decoded_objects = decode(img_pil)
    if decoded_objects:
        for obj in decoded_objects:
            if obj.type == 'QRCODE':
                qr_data = obj.data.decode('utf-8')
                return qr_data[-6:]
    return None

# --- 3. Função Auxiliar: Encontrar o Melhor Template ---
def find_best_match_template(img_input_gray: np.ndarray, preloaded_templates: list[dict], orb_detector: cv2.ORB, bf_matcher: cv2.BFMatcher, ratio_thresh: float = 0.75, min_matches_req: int = 10):
    kp_input, des_input = orb_detector.detectAndCompute(img_input_gray, None)
    if des_input is None:
        return -1, None, None, None, None, None
    best_num_matches, best_template_idx, best_H, best_good_matches = -1, -1, None, None
    for idx, template_data in enumerate(preloaded_templates):
        img_template = template_data['image']
        kp_template = template_data['kp']
        des_template = template_data['des']
        if des_template is None:
            continue
        matches = bf_matcher.knnMatch(des_template, des_input, k=2)
        current_good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
        if len(current_good_matches) >= min_matches_req:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in current_good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_input[m.trainIdx].pt for m in current_good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                num_inliers = np.sum(mask)
                if num_inliers > best_num_matches:
                    best_num_matches, best_template_idx = num_inliers, idx
                    best_H, best_good_matches = H, current_good_matches
    if best_template_idx != -1:
        best_template_data = preloaded_templates[best_template_idx]
        return best_template_idx, best_template_data['image'], best_template_data['kp'], best_template_data['des'], best_good_matches, best_H
    return -1, None, None, None, None, None

# --- 4. Processamento do cartão ---
def process_document_card_v2(input_image, preloaded_templates_data, orb_detector, bf_matcher):
    gray_for_match = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    qr_code = read_qr_code(input_image)
    if not qr_code:
        return False, "Nenhum QR Code encontrado", None
    is_blurry_flag, _ = is_blurry_in_memory(input_image)
    if is_blurry_flag:
        return False, "Imagem borrada", None
    best_template_idx, best_img_template, _, _, _, H = find_best_match_template(gray_for_match, preloaded_templates_data, orb_detector, bf_matcher)
    if best_template_idx == -1 or H is None:
        return False, "Alinhamento com template falhou", None
    h_template, w_template, _ = best_img_template.shape
    pts_template = np.float32([[0, 0], [w_template - 1, 0], [w_template - 1, h_template - 1], [0, h_template - 1]]).reshape(-1, 1, 2)
    dst_transformed = cv2.perspectiveTransform(pts_template, H)
    center_x, center_y = np.mean(dst_transformed[:, 0, 0]), np.mean(dst_transformed[:, 0, 1])
    for i in range(4):
        offset_x, offset_y = dst_transformed[i, 0, 0] - center_x, dst_transformed[i, 0, 1] - center_y
        dst_transformed[i, 0, 0] += offset_x * 0.01
        dst_transformed[i, 0, 1] += offset_y * 0.01
    target = np.float32([[0, 0], [w_template - 1, 0], [w_template - 1, h_template - 1], [0, h_template - 1]])
    M = cv2.getPerspectiveTransform(dst_transformed, target)
    corrected_image = cv2.warpPerspective(input_image, M, (w_template, h_template))
    return True, qr_code, corrected_image

# --- Encode para base64 ---
def image_to_base64(image_np):
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    _, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode('utf-8')

# --- API Flask ---
app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API running"}), 200

@app.route('/process-image', methods=['POST'])
def process_image():
    image_url = request.json.get('image_url', None)
    if not image_url:
        return jsonify({"success": False, "message": "URL da imagem não fornecida"}), 200
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raw.decode_content = True
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            return jsonify({"success": False, "message": f"URL não retornou uma imagem. Content-Type: {content_type}"}), 200
        try:
            #image_pil = Image.open(response.raw).convert("RGB")
            image_pil = Image.open(BytesIO(response.content)).convert("RGB")
            
            #os.makedirs("temp", exist_ok=True); image_pil.save("temp/debug_image.jpg")

        except Exception:
            with open("temp_image.jpg", "wb") as f:
                f.write(response.content)
            image_pil = Image.open("temp_image.jpg").convert("RGB")
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        preloaded_templates = []
        for filename in os.listdir("compare_templates"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join("compare_templates", filename)
                img_template = cv2.imread(path)
                gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray_template, None)
                if des is not None:
                    preloaded_templates.append({'name': filename, 'image': img_template, 'kp': kp, 'des': des})
        if not preloaded_templates:
            return jsonify({"success": False, "message": "Nenhum template válido carregado."}), 200
        success, qr_code, processed_img = process_document_card_v2(image_np, preloaded_templates, orb, matcher)
        if not success:
            return jsonify({"success": False, "message": qr_code}), 200
        base64_img = image_to_base64(processed_img)
        return jsonify({"success": True, "message": "Imagem processada com sucesso", "ticket_number": qr_code, "processed_image_base64": base64_img}), 200
    except Exception as e:
        return jsonify({"success": False, "message": f"Erro inesperado: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
