import cv2
import numpy as np
import os
import shutil
import base64
from pyzbar.pyzbar import decode
from PIL import Image
from flask import Flask, request, jsonify
import requests
from io import BytesIO

# --- 1. Função para Detecção de Desfoque em memória ---
def is_blurry_in_memory(image_np: np.ndarray, blur_threshold: float = 5.0) -> tuple[bool, float]:
    """
    Verifica se uma imagem (array NumPy) está borrada usando a variância do Laplaciano.
    Aplica um pré-filtro gaussiano para reduzir ruído e obter uma medida mais robusta.

    Args:
        image_np (np.ndarray): O array NumPy da imagem (geralmente BGR).
        blur_threshold (float): Limite numérico. Se a variância do Laplaciano for
                                menor que este valor, a imagem é considerada borrada.

    Returns:
        tuple[bool, float]: Uma tupla contendo:
                            - True se a imagem é considerada borrada, False caso contrário.
                            - O valor da variância do Laplaciano calculado.
    """
    if image_np is None or image_np.size == 0:
        return True, 0.0  # Considera borrada se a imagem não for válida

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian_variance = cv2.Laplacian(processed_gray, cv2.CV_64F).var()

    return laplacian_variance < blur_threshold, laplacian_variance

# --- 2. Função para Ler QR Code ---
def read_qr_code(image_np: np.ndarray) -> str | None:
    """
    Tenta ler um QR Code de uma imagem (array NumPy) e retorna os 6 últimos dígitos do QR Code.

    Args:
        image_np (np.ndarray): O array NumPy da imagem contendo o QR Code.

    Returns:
        str | None: Os 6 últimos dígitos do QR Code, ou None se não encontrado.
    """
    if image_np is None or image_np.size == 0:
        return None

    img_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    decoded_objects = decode(img_pil)

    if decoded_objects:
        for obj in decoded_objects:
            if obj.type == 'QRCODE':
                # Pega os 6 últimos dígitos da string do QR Code
                qr_data = obj.data.decode('utf-8')
                return qr_data[-6:]  # Pega os últimos 6 dígitos
    return None


# --- 3. Função Auxiliar: Encontrar o Melhor Template ---
def find_best_match_template(img_input_gray: np.ndarray, preloaded_templates: list[dict], orb_detector: cv2.ORB, bf_matcher: cv2.BFMatcher, ratio_thresh: float = 0.75, min_matches_req: int = 10) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, list | None, np.ndarray | None]:
    kp_input, des_input = orb_detector.detectAndCompute(img_input_gray, None)
    if des_input is None:
        return -1, None, None, None, None, None

    best_num_matches = -1
    best_template_idx = -1
    best_H = None
    best_good_matches = None
    
    for idx, template_data in enumerate(preloaded_templates):
        img_template = template_data['image']
        kp_template = template_data['kp']
        des_template = template_data['des']

        if des_template is None:
            continue

        matches = bf_matcher.knnMatch(des_template, des_input, k=2)
        current_good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                current_good_matches.append(m)

        if len(current_good_matches) >= min_matches_req:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in current_good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_input[m.trainIdx].pt for m in current_good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None:
                num_inliers = np.sum(mask)
                if num_inliers > best_num_matches:
                    best_num_matches = num_inliers
                    best_template_idx = idx
                    best_H = H
                    best_good_matches = current_good_matches

    if best_template_idx != -1:
        best_template_data = preloaded_templates[best_template_idx]
        return best_template_idx, best_template_data['image'], best_template_data['kp'], best_template_data['des'], best_good_matches, best_H
    else:
        return -1, None, None, None, None, None

# --- 4. Função de Processamento de Cartão ---
def process_document_card_v2(input_image: np.ndarray, preloaded_templates_data: list[dict], orb_detector: cv2.ORB, bf_matcher: cv2.BFMatcher) -> tuple[bool, str, np.ndarray | None]:
    gray_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    qr_code = read_qr_code(input_image)
    if not qr_code:
        return False, "Nenhum QR Code encontrado", None

    is_blurry_flag, _ = is_blurry_in_memory(input_image)
    if is_blurry_flag:
        return False, "Imagem borrada", None

    best_template_idx, best_img_template, best_kp_template, best_des_template, best_good_matches, H = find_best_match_template(gray_input, preloaded_templates_data, orb_detector, bf_matcher)
    
    if best_template_idx == -1 or H is None:
        return False, "Falha: Não foi possível encontrar um bom alinhamento com nenhum dos templates.", None
    
    h_template, w_template, _ = best_img_template.shape
    pts_template_corners = np.float32([[0, 0], [w_template - 1, 0], [w_template - 1, h_template - 1], [0, h_template - 1]]).reshape(-1, 1, 2)
    dst_transformed_corners = cv2.perspectiveTransform(pts_template_corners, H)

    center_x = np.mean(dst_transformed_corners[:, 0, 0])
    center_y = np.mean(dst_transformed_corners[:, 0, 1])

    expansion_factor = 0.01
    for i in range(4):
        offset_x = dst_transformed_corners[i, 0, 0] - center_x
        offset_y = dst_transformed_corners[i, 0, 1] - center_y
        dst_transformed_corners[i, 0, 0] += offset_x * expansion_factor
        dst_transformed_corners[i, 0, 1] += offset_y * expansion_factor

    target_corners = np.float32([[0, 0], [w_template - 1, 0], [w_template - 1, h_template - 1], [0, h_template - 1]])
    M_transform = cv2.getPerspectiveTransform(dst_transformed_corners, target_corners)
    corrected_image = cv2.warpPerspective(input_image, M_transform, (w_template, h_template))

    return True, qr_code, corrected_image

# --- Função para Conversão de Imagem para Base64 ---
def image_to_base64(image_np: np.ndarray) -> str:
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

    _, buffer = cv2.imencode('.jpg', image_np)
    return base64.b64encode(buffer).decode('utf-8')


# --- API Flask ---
app = Flask(__name__)

#Rota de testes básica
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API running"}), 200


@app.route('/process-image', methods=['POST'])
def process_image():
    image_url = request.json.get('image_url', None)
    if not image_url:
        return jsonify({"success": False, "message": "URL da imagem não fornecida"}), 400

    try:
        response = requests.get(image_url)
        image_np = np.array(Image.open(BytesIO(response.content)))

        orb_detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Carregar templates de comparação da pasta
        preloaded_templates_data = []
        COMPARE_TEMPLATES_FOLDER = "compare_templates"  # Defina o caminho correto para os templates
        for filename in os.listdir(COMPARE_TEMPLATES_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                template_path = os.path.join(COMPARE_TEMPLATES_FOLDER, filename)
                img_template = cv2.imread(template_path)
                gray_template = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
                kp_template, des_template = orb_detector.detectAndCompute(gray_template, None)

                if des_template is not None:
                    preloaded_templates_data.append({
                        'name': filename,
                        'image': img_template,
                        'kp': kp_template,
                        'des': des_template
                    })

        if not preloaded_templates_data:
            return jsonify({"success": False, "message": "Nenhum template válido carregado."}), 400

        # Processar a imagem
        success, qr_code, processed_image = process_document_card_v2(image_np, preloaded_templates_data, orb_detector, bf_matcher)

        if not success:
            return jsonify({"success": False, "message": qr_code}), 400

        processed_image_base64 = image_to_base64(processed_image)

        return jsonify({
            "success": True,
            "message": "Imagem processada com sucesso",
            "ticket_number": qr_code,
            "processed_image_base64": processed_image_base64
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
