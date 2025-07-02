# 🦵 Knee Osteoarthritis Prediction (Gradio Deployment)

Aplikasi ini merupakan alat bantu diagnosis tingkat keparahan *Knee Osteoarthritis* (OA) dari citra X-ray menggunakan *Deep Learning* (ResNet50, VGG19, DenseNet121) dan Grad-CAM untuk interpretasi visual.

---

## 🏗️ Arsitektur Sistem

```
User (Web Browser)
│
▼
[VPN ITS] ---> [Server Pribadi / VM]
│
▼
[Gradio App]
```

Akses ke aplikasi dilakukan melalui jaringan VPN ITS ke server pribadi yang menjalankan Gradio.

---

## ✅ Prasyarat

- Server pribadi (VPS/VM) dengan akses root
- Telah terhubung ke VPN ITS
- Port 7860 dibuka di firewall (`ufw`)

---

## 🔐 Masuk ke Server

```bash
ssh root@[IP_PRIBADI]
```

---

## ⚙️ Persiapan Sistem

```bash
sudo apt update
sudo apt install -y git git-lfs python3 python3-pip
sudo apt-get install -y ffmpeg libsm6 libxext6
```

---

## 📥 Instalasi

1. Masuk ke direktori kerja:
   ```bash
   mkdir -p ~/opt
   cd ~/opt
   ```

2. Clone repositori:
   ```bash
   git clone https://github.com/ilyasash/knee-oa-fastapi.git
   cd kneeOA-Prediction
   ```

3. Buat dan aktifkan lingkungan virtual:
   ```bash
   sudo apt install -y python3-venv
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Instal dependensi Python:
   ```bash
   pip install -r requirement.txt
   ```

---

## 🚀 Menjalankan Aplikasi

Jalankan di latar belakang menggunakan `nohup`:
```bash
nohup python app.py > log.txt 2>&1 &
```

Aplikasi akan berjalan di latar belakang dan log akan tersimpan di `log.txt`.

---

## 🔐 Firewall (Jika Dibutuhkan)

```bash
sudo ufw status
sudo ufw allow 7860
```

---

## 🧪 Cek Status Proses

```bash
ps aux | grep app.py
```

Jika aplikasi berjalan, akan terlihat proses `python app.py` aktif dengan PID.

---

## 🌐 Akses Aplikasi

Buka di peramban:
```
http://[IP_PRIBADI]:7860
```

Pastikan kamu sudah terhubung ke VPN ITS agar bisa mengakses.

---

## ⛔️ Menghentikan Server (Shutdown App)

1. Temukan PID proses aplikasi:
   ```bash
   ps aux | grep app.py
   ```

2. Hentikan proses:
   ```bash
   kill <PID>
   ```

---

## 📌 Catatan

- Pastikan koneksi VPN ITS stabil agar akses ke server tidak terputus.
