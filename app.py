import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import traceback
import time

# Nonaktifkan JIT compilation jika menyebabkan masalah
tf.config.optimizer.set_jit(False)

# 1. MEMUAT MODEL
try:
    trained_models = {
        'resnet50': load_model('best_model_resnet50.h5'),
        'vgg19': load_model('best_model_vgg19.h5'),
        'densenet121': load_model('best_model_densenet121.h5')
    }
    print("Semua model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# 2. FUNGSI-FUNGSI PEMBANTU
def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D): return layer.name
    raise ValueError("Tidak ditemukan layer Conv2D dalam model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()

def show_gradcam(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img
    return np.clip(superimposed_img, 0, 255).astype(np.uint8)

# 3. FUNGSI-FUNGSI PEMROSESAN GAMBAR
def prepare_input_for_model(x, model_name):
    img_array = np.expand_dims(x, axis=0)
    if model_name == 'resnet50': from tensorflow.keras.applications.resnet50 import preprocess_input
    elif model_name == 'vgg19': from tensorflow.keras.applications.vgg19 import preprocess_input
    elif model_name == 'densenet121': from tensorflow.keras.applications.densenet import preprocess_input
    else: raise ValueError("Model tidak dikenal.")
    return preprocess_input(img_array.copy() * 255)

def to_rgb(x):
    if len(x.shape) == 2: return np.stack([x] * 3, axis=-1)
    if len(x.shape) == 3 and x.shape[-1] == 1: return np.repeat(x, 3, axis=-1)
    return x

def inverse_preprocess_for_display(img_array):
    return np.clip(img_array * 255, 0, 255).astype(np.uint8)

# 4. FUNGSI-FUNGSI PENJELASAN (XAI) DAN INTERPRETASI
classes = [f"Knee OA Level {i}" for i in range(5)]

def generate_detailed_xai_explanation(heatmap, prediction_label):
    heatmap_resized = cv2.resize(heatmap, (200, 200))
    regions = {
        "Celah Sendi (Joint Space)": (90, 110, 40, 160), "Kompartemen Medial (Dalam)": (90, 120, 40, 100),
        "Kompartemen Lateral (Luar)": (90, 120, 100, 160), "Kondilus Femoralis Medial (Tulang Paha Dalam)": (70, 90, 40, 100),
        "Kondilus Femoralis Lateral (Tulang Paha Luar)": (70, 90, 100, 160), "Plateau Tibialis Medial (Tulang Kering Dalam)": (110, 130, 40, 100),
        "Plateau Tibialis Lateral (Tulang Kering Luar)": (110, 130, 100, 160),
    }
    region_intensities = {name: np.mean(heatmap_resized[y1:y2, x1:x2]) for name, (y1, y2, x1, x2) in regions.items() if heatmap_resized[y1:y2, x1:x2].size > 0}
    if not region_intensities: return "Tidak dapat menganalisis wilayah heatmap."
    focus_region, max_intensity = max(region_intensities.items(), key=lambda item: item[1])
    explanation = f"Analisis untuk prediksi <b>{prediction_label}</b>:<br>"
    if max_intensity > 0.4:
        explanation += f"- Model memfokuskan perhatian utamanya pada area <b>{focus_region}</b>. "
        if "Medial" in focus_region or "Lateral" in focus_region: explanation += "Fokus di area ini seringkali terkait dengan <b>penyempitan celah sendi</b> atau pembentukan <b>osteofit (taji tulang)</b>.<br>"
        elif "Celah Sendi" in focus_region: explanation += "Ini mengindikasikan bahwa <b>penyempitan celah sendi</b> kemungkinan merupakan faktor utama.<br>"
        else: explanation += "Ini bisa mengindikasikan adanya perubahan struktural pada tepi tulang, seperti <b>peruncingan (spurring) atau sklerosis</b>.<br>"
    else:
        explanation += "- Model tidak menunjukkan fokus visual yang jelas, analisis di bawah didasarkan pada hasil klasifikasi.<br>"
    explanation += "<br><b>Untuk Pasien:</b><br>"
    if max_intensity > 0.4:
        if ("Medial" in focus_region or "Lateral" in focus_region or "Celah Sendi" in focus_region):
            explanation += f"Area yang ditandai (sekitar <b>{focus_region}</b>) menunjukkan kemungkinan adanya <b>kerusakan tulang rawan atau penyempitan celah sendi</b>."
        elif ("Kondilus" in focus_region or "Plateau" in focus_region):
            explanation += f"Area yang ditandai pada <b>tepi tulang ({focus_region})</b> bisa berarti adanya <b>taji tulang (osteofit)</b>."
        if "Level 2" in prediction_label:
            explanation += f" Karena ini terdeteksi sebagai <b>{prediction_label}</b>, disarankan berkonsultasi dengan dokter untuk penanganan dini."
        elif "Level 3" in prediction_label or "Level 4" in prediction_label:
            explanation += f" Karena ini terdeteksi sebagai <b>{prediction_label}</b>, konsultasi segera dengan dokter spesialis sangat disarankan."
    else:
        explanation += "Model tidak dapat menyorot area spesifik dengan jelas. Namun, berdasarkan hasil klasifikasi:<br>"
        if "Level 0" in prediction_label or "Level 1" in prediction_label:
            explanation += f"Hasil prediksi adalah <b>{prediction_label}</b>, yang mengindikasikan kondisi lutut normal atau mendekati normal. Tetap jaga kesehatan sendi dengan gaya hidup sehat."
        elif "Level 2" in prediction_label:
            explanation += f"Hasil prediksi adalah <b>{prediction_label}</b>, yang kemungkinan menandakan Osteoarthritis tahap awal. Disarankan berkonsultasi dengan dokter untuk penanganan dini seperti fisioterapi atau penyesuaian gaya hidup."
        elif "Level 3" in prediction_label:
            explanation += f"Hasil prediksi adalah <b>{prediction_label}</b>, menandakan Osteoarthritis tingkat menengah. Penting untuk segera berkonsultasi dengan dokter untuk membahas pilihan penanganan yang lebih intensif."
        elif "Level 4" in prediction_label:
            explanation += f"Hasil prediksi adalah <b>{prediction_label}</b>, menandakan Osteoarthritis tingkat lanjut. Konsultasi segera dengan dokter spesialis sangat disarankan untuk menentukan rencana pengobatan yang komprehensif."
    return f"<div class='xai-explanation-box'>{explanation}</div>"

def create_about_app_info():
    return """
    <div class='interpretation-box'>
        <h4>Tentang Aplikasi & Knee Osteoarthritis (OA)</h4>
        <p><strong>Tujuan Aplikasi:</strong> Aplikasi ini menggunakan teknologi Deep Learning untuk membantu menganalisis gambar X-ray lutut dan memberikan prediksi tingkat keparahan Osteoarthritis (OA) berdasarkan Kellgren-Lawrence (KL) grade. Ini bertujuan sebagai alat bantu (bukan pengganti) bagi tenaga medis.</p>
        <p><strong>Apa itu Knee OA?</strong> Osteoarthritis lutut adalah penyakit sendi degeneratif yang ditandai oleh kerusakan tulang rawan pada sendi lutut, menyebabkan rasa sakit, kaku, dan pembengkakan.</p>
        <p><strong>Penyebab & Faktor Risiko:</strong> Usia, berat badan berlebih (obesitas), riwayat cedera lutut, faktor genetik, dan aktivitas fisik berat yang berulang.</p>
        <p><strong>Penanganan Umum:</strong> Melibatkan kombinasi dari perubahan gaya hidup (penurunan berat badan), fisioterapi, obat pereda nyeri, hingga operasi penggantian sendi pada kasus yang parah.</p>
    </div>
    """

def create_heatmap_legend():
    return """
    <div class='interpretation-box'>
        <h4>Cara Membaca Heatmap (Peta Panas)</h4>
        <p>Heatmap menunjukkan area mana pada gambar yang paling mempengaruhi keputusan model. Berdasarkan output model ini, skala warna berikut digunakan:</p>
        <ul class='legend-list'>
            <li><span style='color:blue; font-weight:bold;'>🔵 Biru</span>: Fokus paling kuat. Area ini memiliki bobot paling tinggi dalam prediksi.</li>
            <li><span style='color:green; font-weight:bold;'>🟢 Hijau</span>: Fokus kuat.</li>
            <li><span style='color:gold; font-weight:bold;'>🟡 Kuning</span>: Fokus sedang.</li>
            <li><span style='color:orange; font-weight:bold;'>🟠 Oranye</span>: Fokus rendah.</li>
            <li><span style='color:red; font-weight:bold;'>🔴 Merah</span>: Kontribusi paling rendah atau dianggap sebagai latar belakang oleh model.</li>
        </ul>
        <p>Dengan melihat area mana yang berwarna, Anda bisa memahami dasar pemikiran visual dari model AI.</p>
    </div>
    """

# 5. FUNGSI UTAMA PREDIKSI DAN PENJELASAN
def predict_and_explain(image):
    try:
        if image is None: raise ValueError("Silakan unggah gambar atau gunakan kamera terlebih dahulu.")
        time.sleep(1)
        img = np.array(image)
        if len(img.shape) == 3 and img.shape[-1] == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (200, 200)) / 255.0
        img_3ch_normalized = to_rgb(img)
        img_display = inverse_preprocess_for_display(img_3ch_normalized)

        heatmaps, probs_all_models = [], []
        for model_name in ['resnet50', 'vgg19', 'densenet121']:
            model = trained_models[model_name]
            last_conv_layer = find_last_conv_layer_name(model)
            img_input = prepare_input_for_model(img_3ch_normalized, model_name)
            preds = model.predict(img_input, verbose=0)
            probs_all_models.append(preds[0])
            heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer)
            heatmaps.append(cv2.resize(heatmap, (200, 200)))

        heatmap_ensemble = np.mean(heatmaps, axis=0)
        probs_ensemble = np.mean(probs_all_models, axis=0)
        pred_ensemble_class, pred_ensemble_label = np.argmax(probs_ensemble), classes[np.argmax(probs_ensemble)]
        resnet_label, vgg19_label, densenet_label = classes[np.argmax(probs_all_models[0])], classes[np.argmax(probs_all_models[1])], classes[np.argmax(probs_all_models[2])]
        resnet_xai, vgg19_xai, densenet_xai = generate_detailed_xai_explanation(heatmaps[0], resnet_label), generate_detailed_xai_explanation(heatmaps[1], vgg19_label), generate_detailed_xai_explanation(heatmaps[2], densenet_label)
        ensemble_xai = generate_detailed_xai_explanation(heatmap_ensemble, pred_ensemble_label)
        imgs_out = [Image.fromarray(show_gradcam(img_display, hm)) for hm in heatmaps + [heatmap_ensemble]]
        conf_texts = [f"<div class='confidence-box'>{classes[np.argmax(p)]} ({np.max(p)*100:.2f}%)</div>" for p in probs_all_models]
        ensemble_result = f"<div class='confidence-box ensemble'>{pred_ensemble_label} ({np.max(probs_ensemble)*100:.2f}%)</div>"

        # PERUBAHAN: Mengembalikan update untuk menyembunyikan teks format
        return (
            gr.Image(value=image, interactive=False, label=""),
            *imgs_out, resnet_xai, vgg19_xai, densenet_xai, ensemble_xai,
            *conf_texts, ensemble_result,
            gr.Column(visible=True), gr.Row(visible=False), gr.Row(visible=True),
            gr.HTML(value=create_heatmap_legend()),
            gr.Markdown(visible=False) # Sembunyikan teks format
        )
    except Exception as e:
        traceback.print_exc()
        error_message = f"Terjadi kesalahan: {e}"
        error_html = f"<div class='error-box'>{error_message}</div>"
        return [gr.Image(interactive=True, label="Unggah, Kamera, atau Tempel Gambar")] + [None]*4 + [error_html]*4 + [""]*3 + ["", gr.Column(visible=True), gr.Row(visible=True), gr.Row(visible=False), gr.HTML(value=create_about_app_info()), gr.Markdown(visible=True)]

# 6. MEMBANGUN ANTARMUKA GRADIO
custom_css = """
/* CSS untuk animasi loading pop-up */
@keyframes lds-ellipsis { 0% { top: 36px; left: 36px; width: 0; height: 0; opacity: 0; } 4.9% { top: 36px; left: 36px; width: 0; height: 0; opacity: 0; } 5% { top: 36px; left: 36px; width: 0; height: 0; opacity: 1; } 100% { top: 0px; left: 0px; width: 72px; height: 72px; opacity: 0; } }
#loading-indicator { display: flex; justify-content: center; align-items: center; height: 100%; width: 100%; background-color: rgba(255, 255, 255, 0.8); z-index: 9999; position: fixed; top: 0; left: 0; }
.lds-ellipsis { display: inline-block; position: relative; width: 80px; height: 80px; }
.lds-ellipsis div { position: absolute; border: 4px solid #3b82f6; opacity: 1; border-radius: 50%; animation: lds-ellipsis 1s cubic-bezier(0, 0.2, 0.8, 1) infinite; }
.lds-ellipsis div:nth-child(2) { animation-delay: -0.5s; }

/* CSS lainnya */
#clear-btn, #re-predict-btn { border: 1px solid #3b82f6 !important; color: #3b82f6 !important; }
#clear-btn:hover, #re-predict-btn:hover { background-color: #eff6ff !important; color: #1d4ed8 !important; border-color: #1d4ed8 !important; }
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
#upload-box button.w-full.h-full.flex.items-center.justify-center { display: flex; flex-direction: column; gap: 8px; font-size: 0.85rem; color: #555; }
#upload-box button.w-full.h-full.flex.items-center.justify-center svg { width: 32px !important; height: 32px !important; }
.card-box { padding: 16px; border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.confidence-box { padding: 12px; text-align: center; font-weight: bold; color: white; border-radius: 8px; background-color: #ff4500; }
.confidence-box.ensemble { background-color: #007bff; }
.xai-explanation-box { font-size: 14px; text-align: justify; padding: 12px; border-left: 4px solid #007bff; background-color: #f8f9fa; border-radius: 0 8px 8px 0; }
.interpretation-box { padding: 16px; background-color: #eef2f9; border-radius: 12px; border: 1px solid #d1d9e6; }
.legend-list { list-style: none; padding-left: 0; } .legend-list li { margin-bottom: 8px; }
.error-box { color: red; text-align: center; padding: 10px; font-weight: bold; }
.card-row { align-items: stretch !important; }
.card-row > div { display: flex !important; flex-direction: column !important; }
.card-row > div > .gr-group { flex-grow: 1 !important; display: flex !important; flex-direction: column !important; }
.xai-explanation-box { flex-grow: 1 !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, title="Diagnosis Knee Osteoarthritis") as demo:
    loading_indicator = gr.HTML("""<div id="loading-indicator"><div class="lds-ellipsis"><div></div><div></div><div></div><div></div></div></div>""", visible=False)

    gr.Markdown("# Diagnosis Knee Osteoarthritis dengan Deep Learning dan XAI")
    gr.Markdown("Unggah gambar X-ray lutut atau gunakan kamera untuk mendapatkan prediksi tingkat keparahan Osteoarthritis beserta penjelasan visual (XAI).")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group(elem_id="upload-box"):
                input_image = gr.Image(type="pil", label="Unggah, Kamera, atau Tempel Gambar", sources=["upload", "webcam", "clipboard"], height=400)
                # PERUBAHAN: Teks diformat di tengah dan diberi variabel `supported_formats_text`
                supported_formats_text = gr.Markdown("<div style='text-align: center;'><i>Format yang didukung: PNG, JPG, JPEG, WEBP</i></div>")

            with gr.Row() as initial_buttons:
                predict_btn = gr.Button("Submit", variant="primary", scale=2)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1, elem_id="clear-btn")

            with gr.Row(visible=False) as after_analysis_buttons:
                upload_new_btn = gr.Button("Upload Gambar Baru", variant="primary", scale=2)
                re_predict_btn = gr.Button("Analisis Ulang", variant="secondary", scale=2, elem_id="re-predict-btn")

            with gr.Group():
                interpretation_text = gr.HTML(value=create_about_app_info())

        with gr.Column(scale=5, visible=False) as results_col:
            gr.Markdown("### Hasil Analisis Model")
            with gr.Row(elem_classes="card-row"):
                with gr.Column():
                    with gr.Group(elem_classes="card-box"):
                        gr.Markdown("<h4 style='text-align: center;'>ResNet50</h4>")
                        resnet_heatmap = gr.Image(label="Heatmap Grad-CAM", interactive=False)
                        resnet_conf = gr.HTML()
                        resnet_xai = gr.HTML()
                with gr.Column():
                    with gr.Group(elem_classes="card-box"):
                        gr.Markdown("<h4 style='text-align: center;'>VGG19</h4>")
                        vgg19_heatmap = gr.Image(label="Heatmap Grad-CAM", interactive=False)
                        vgg19_conf = gr.HTML()
                        vgg19_xai = gr.HTML()
                with gr.Column():
                    with gr.Group(elem_classes="card-box"):
                        gr.Markdown("<h4 style='text-align: center;'>DenseNet121</h4>")
                        densenet_heatmap = gr.Image(label="Heatmap Grad-CAM", interactive=False)
                        densenet_conf = gr.HTML()
                        densenet_xai = gr.HTML()

            gr.Markdown("<hr>")
            gr.Markdown("<h3 style='text-align: center;'>Hasil Ensemble (Gabungan)</h3>")
            with gr.Row():
                with gr.Column(scale=1): pass
                with gr.Column(scale=2):
                        with gr.Group(elem_classes="card-box"):
                            ensemble_heatmap = gr.Image(label="Heatmap Grad-CAM Ensemble", interactive=False)
                            ensemble_result = gr.HTML()
                            ensemble_xai = gr.HTML()
                with gr.Column(scale=1): pass

    # PERUBAHAN: Menambahkan 'supported_formats_text' ke daftar output
    outputs_list = [
        input_image,
        resnet_heatmap, vgg19_heatmap, densenet_heatmap, ensemble_heatmap,
        resnet_xai, vgg19_xai, densenet_xai, ensemble_xai,
        resnet_conf, vgg19_conf, densenet_conf, ensemble_result,
        results_col, initial_buttons, after_analysis_buttons,
        interpretation_text,
        supported_formats_text
    ]

    def start_process():
        return gr.HTML(visible=True)
    def finish_process():
        return gr.HTML(visible=False)

    predict_btn.click(fn=start_process, outputs=loading_indicator).then(fn=predict_and_explain, inputs=input_image, outputs=outputs_list).then(fn=finish_process, outputs=loading_indicator)
    re_predict_btn.click(fn=start_process, outputs=loading_indicator).then(fn=predict_and_explain, inputs=input_image, outputs=outputs_list).then(fn=finish_process, outputs=loading_indicator)

    def clear_func():
        # PERUBAHAN: Mengembalikan update untuk semua komponen yang relevan, termasuk menampilkan kembali teks format
        return (
            gr.Image(value=None, interactive=True, label="Unggah, Kamera, atau Tempel Gambar"),
            gr.Column(visible=False),
            gr.Row(visible=True),
            gr.Row(visible=False),
            gr.HTML(visible=False),
            gr.HTML(value=create_about_app_info()), # Kembalikan ke info awal
            gr.Markdown(visible=True) # Tampilkan kembali teks format
        )

    # PERUBAHAN: Menambahkan 'supported_formats_text' ke daftar output untuk tombol clear/upload baru
    clear_outputs = [input_image, results_col, initial_buttons, after_analysis_buttons, loading_indicator, interpretation_text, supported_formats_text]
    clear_btn.click(fn=clear_func, inputs=None, outputs=clear_outputs)
    upload_new_btn.click(fn=clear_func, inputs=None, outputs=clear_outputs)


# 7. MENJALANKAN APLIKASI
if __name__ == '__main__':
    demo.launch(share=True, debug=True, server_name="0.0.0.0", server_port=7860)
    
