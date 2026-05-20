# Arabic Handwritten OCR

Dit project gebruikt het `Qwen2.5-VL` model (specifiek de `sherif1313/Arabic-English-handwritten-OCR-v3` fine-tune) om Optical Character Recognition (OCR) uit te voeren op handgeschreven Arabische tekst.

## Kenmerken
- Geoptimaliseerd voor Arabische handgeschreven tekst.
- Ondersteunt zowel lokale installatie als VS Code Dev Containers.
- GPU-versnelling ondersteuning via CUDA 12.1.
- Automatische voorbewerking van afbeeldingen.

---

## Vereisten

Voordat je begint, zorg ervoor dat je machine aan de volgende eisen voldoet:

- **NVIDIA GPU**: Sterk aanbevolen (minimaal 8GB VRAM). Het model kan op de CPU draaien, maar dit zal extreem traag zijn.
- **NVIDIA Drivers**: Geïnstalleerd op de host-machine.
- **Python 3.10+**: (Indien lokaal uitgevoerd).
- **Docker & NVIDIA Container Toolkit**: (Indien Dev Containers worden gebruikt).

---

## Installatie-instructies (A tot Z)

### Optie 1: Lokale Setup (Linux/Ubuntu) - Met `uv` (aanbevolen)

1. **Clone de repository**:
   ```bash
   git clone https://github.com/your-repo/arabic-handwritten-to-text.git
   cd arabic-handwritten-to-text
   ```

2. **Voer het setup-script uit**:
   Dit script installeert `uv` (als dit nog niet geïnstalleerd is), maakt een virtuele omgeving aan en installeert alle afhankelijkheden (PyTorch met CUDA 12.1 ondersteuning, Transformers, etc.) in seconden.
   ```bash
   chmod +x setup_arabic_ocr_env.sh
   ./setup_arabic_ocr_env.sh
   ```

3. **Activeer de omgeving**:
   ```bash
   source .venv/bin/activate
   ```

**Voordelen van `uv`:**
- ⚡ 10-100x sneller dan pip/conda
- 🔧 Geen conda problemen (geen "RootModel" fouten meer)
- 📦 Automatische Python versie management
- 🛡️ Betrouwbare dependency resolution

### Optie 2: Dev Container Setup (Aanbevolen voor VS Code gebruikers)

Als je VS Code en Docker hebt geïnstalleerd:

1. Open de projectmap in VS Code.
2. Wanneer gevraagd wordt om "Reopen in Container", klik op **Reopen in Container**.
   - Alternatief: druk op `F1`, typ `Dev Containers: Reopen in Container`.
3. De container zal automatisch bouwen en alle afhankelijkheden installeren. Dit kan enkele minuten duren omdat het PyTorch GPU image wordt gedownload.

### Belangrijk: De venv activeren
Voordat je een script uitvoert (`preprocess_images.py` of `ocr_arabic.py`), moet je **altijd** de virtuele omgeving activeren:
```bash
source .venv/bin/activate
```
Als je dit vergeet, krijg je foutmeldingen zoals `ModuleNotFoundError: No module named 'cv2'`.

## Gebruik

### 1. Afbeeldingen Voorbereiden
Plaats je handgeschreven Arabische afbeeldingen in de `images/` map. Ondersteunde formaten: `.jpg`, `.png`, `.jpeg`, `.bmp`, `.webp`, `.tif`.

### 2. Voorbewerking (Optioneel maar aanbevolen)
Voer het voorbewerkingsscript uit om de kwaliteit van de afbeeldingen te verbeteren voor betere OCR-resultaten:
```bash
python preprocess_images.py
```
Dit slaat de bewerkte afbeeldingen op in de `processed_images/` map.

### 3. OCR Uitvoeren
Om de tekstextractie te starten:
```bash
# Zorg dat je omgeving geactiveerd is
python ocr_arabic.py
```

Het script zal:
- Het model laden van Hugging Face (de eerste keer wordt ~10GB aan gewichten gedownload).
- Alle afbeeldingen in de `processed_images/` map verwerken.
- De resultaten opslaan in `output.txt`.

---

## Probleemoplossing

### CUDA/GPU Problemen
- **"CUDA is not available"**: Controleer of de NVIDIA-drivers en CUDA-toolkit correct zijn geïnstalleerd. Als je Docker gebruikt, zorg dan dat de `nvidia-container-toolkit` op je host is geïnstalleerd.
- **Out of Memory (OOM)**: Het model is groot. Het script gebruikt `expandable_segments:True` om te helpen, maar als je nog steeds OOM-fouten krijgt, sluit dan andere GPU-intensieve applicaties.
  ```bash
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python ocr_arabic.py
  ```

### `uv` Installatieproblemen
- **"uv command not found"**: Na het installeren moet je je shell herstarten of dit uitvoeren:
  ```bash
  source $HOME/.cargo/env
  ```
- **Wil terug naar pip?**: Je kunt handmatig pip gebruiken:
  ```bash
  source .venv/bin/activate
  pip install -e .
  ```

---

## Projectstructuur
- `ocr_arabic.py`: Hoofdscript voor OCR-uitvoering.
- `setup_arabic_ocr_env.sh`: Geautomatiseerde omgevingssetup.
- `preprocess_images.py`: Script voor beeldverbetering.
- `images/`: Plaats hier je invoerbestanden.
- `processed_images/`: Waar verbeterde afbeeldingen worden opgeslagen.
- `.devcontainer/`: Configuratie voor Docker-gebaseerde ontwikkeling.
