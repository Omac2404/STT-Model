import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import os

# MODELIN YUKLENMESI
model = whisper.load_model("small")

def ses_kaydet(sure=5, frekans=16000):
    print("Kayıt başladı...")
    ses = sd.rec(int(sure * frekans), samplerate=frekans, channels=1, dtype='int16')
    sd.wait()
    print("Kayıt tamamlandı.")
    return np.squeeze(ses)

def transkribe_et(ses_verisi, frekans=16000):
    # temp.wav dosyasını oluştur
    scipy.io.wavfile.write("temp.wav", frekans, ses_verisi)

    # Modelle transkripsiyon
    sonuc = model.transcribe("temp.wav", language="turkish", fp16=False)

    # 'kayıtlar' klasörünü oluştur (yoksa)
    if not os.path.exists("kayıtlar"):
        os.makedirs("kayıtlar")

    # Sonucu dosyaya kaydet
    with open("kayıtlar/transkript.txt", "w", encoding="utf-8") as f:
        f.write(sonuc["text"])

    return sonuc["text"]

# Ana işlem
if __name__ == "__main__":
    veri = ses_kaydet()
    metin = transkribe_et(veri)
    print("Çözümleme:", metin)