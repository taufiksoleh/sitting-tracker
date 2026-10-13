# Sitting Tracker

Sistem Python untuk membaca video/CCTV (file atau RTSP), mendeteksi customer dengan YOLO,
mengenali wajah secara anonim (Re-ID), dan menghitung berapa lama tiap customer duduk secara realtime.

## Alur sistem

1. **Tandai zona** kursi/meja dengan `zone_tool.py` pada frame pertama → `zones.json`.
2. **Deteksi + tracking** orang dengan YOLO11 + ByteTrack (`main.py`).
3. **Uji zona** (`dwell.anchor`): `overlap` (default) memilih zona yang menutupi porsi bbox terbesar
   (>= `min_overlap`), sehingga orang yang duduk di samping meja atau membelakangi kamera tetap terhitung.
   Alternatif: titik `bottom_center` / `center` bbox diuji dengan `cv2.pointPolygonTest`.
4. **Durasi duduk** (`dwell.py`, logika murni): sesi dimulai setelah berada di zona >= `min_sit_seconds`,
   tidak terputus jika hilang/keluar zona <= `leave_grace_seconds`, dan dipecah saat pindah zona.
5. **Re-ID wajah anonim** (`face_reid.py`): tiap N frame, hanya untuk track yang belum punya ID,
   wajah di 45% bagian atas bbox diubah menjadi embedding lalu dicocokkan ke galeri → ID `C001`, `C002`, ...
6. **Output**: overlay realtime, `out.mp4` (opsional), dan `sessions.csv`. Timer `mm:ss` tampil di atas
   setiap pengunjung sejak masuk zona (hijau = belum `min_sit_seconds`, merah = dihitung duduk) dan tetap
   tampil selama masa grace saat deteksi putus sesaat. Garis zona disembunyikan (`output.draw_zones: false`).

## Instalasi

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. Model YOLO dan InsightFace (`buffalo_s`) diunduh otomatis pada pemakaian pertama.

## Cara pakai

```bash
# 1. Tandai zona (klik kiri = titik, n = selesai zona, u = undo, s = simpan, q = keluar)
python zone_tool.py --source sample.mp4

# 2. Jalankan pelacakan
python main.py                       # memakai config.yaml
python main.py --source rtsp://user:pass@192.168.1.10:554/stream1
```

Pengaturan ada di `config.yaml` (sumber, model, device `cpu`/`cuda:0`/`mps`, ambang, opsi output).
Untuk produksi cukup ganti `source` dari file video menjadi URL RTSP.

Tes logika durasi: `python -m pytest test_dwell.py`

## Format `sessions.csv`

| kolom | arti |
|---|---|
| `track_id` | ID track dari ByteTrack |
| `customer_id` | ID anonim (`C001`, ...), kosong jika wajah tidak pernah terbaca |
| `zone` | nama zona (mis. `table_1`) |
| `start_time`, `end_time` | file video: offset `HH:MM:SS.mmm` dari awal video; RTSP: waktu jam dinding ISO |
| `duration_s` | durasi sesi dalam detik |

## Keterbatasan

- Wajah di CCTV harus berukuran >= 40 px dan menghadap kamera; jika tidak, `customer_id` kosong.
- Jika track putus (terhalang lama, ID berganti), sesi baru dibuat. Penggabungan sesi per `customer_id`
  lintas track belum diimplementasikan.
- Deteksi: CCTV ramai butuh minimal `yolo11s` dengan `imgsz` ~1088; `yolo11n@640` melewatkan sebagian besar
  pengunjung. Pengunjung yang hampir seluruhnya tertutup orang lain tetap bisa terlewat.
- Kinerja: `device: mps` (Apple Silicon) atau `cuda:0` disarankan; realtime pada banyak kamera/resolusi tinggi
  membutuhkan GPU (`device: cuda:0`).
- Deteksi duduk berbasis zona, bukan pose: orang berdiri di dalam zona tetap dihitung duduk.
- Galeri Re-ID hanya di memori; ID di-reset setiap program dijalankan ulang.

## Privasi

- Tidak ada nama dan tidak ada foto wajah yang disimpan. Embedding wajah hanya ada di memori
  dan hilang saat program berhenti.
- Pemrosesan data wajah/biometrik tunduk pada UU Pelindungan Data Pribadi (UU PDP No. 27 Tahun 2022);
  pastikan dasar hukum dan kebijakan retensi sesuai.
- Pasang pemberitahuan CCTV yang jelas di area yang dipantau.
