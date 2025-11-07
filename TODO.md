# Proje: LLaMA 3.1 8B ile Gelişmiş Hikaye Üretimi

Bu TODO listesi, mevcut modelin LLaMA 3.1 8B ile değiştirilmesi ve yeni veri setleriyle yeniden eğitilmesi sürecini takip eder.

## Adım 1: Proje Temizliği ve Hazırlık

- [x] Mevcut fine-tuned modelin (`fine_tuned_model`) silinmesi.
- [x] Eski log dosyalarının (`train_error.log`, `train_output.log`) silinmesi.
- [x] MLflow ve WandB log klasörlerinin (`mlruns`, `wandb`) silinmesi.
- [x] Eski veri işleme ve eğitim betiklerinin (`process_kaggle_data.py`, `train_model.py`) silinmesi.
- [x] `requirements.txt` dosyasının LLaMA 3.1 8B ve QLoRA/PEFT için güncellenmesi.

## Adım 2: Veri Setlerinin Hazırlanması

- [x] Children's Book Test (CBT) veri setinin indirilmesi ve işlenmesi.
- [x] Project Gutenberg - Children's Literature veri setinin indirilmesi ve işlenmesi.
- [x] Kaggle Storytelling veri setinin indirilmesi ve işlenmesi.
- [x] Tüm veri setlerini birleştirip fine-tuning için tek bir formatta hazırlanması.

## Adım 3: Model Eğitimi (Fine-tuning)

- [x] LLaMA 3.1 8B modelini QLoRA/PEFT ile fine-tuning yapmak için yeni bir eğitim betiğinin (`train_llama.py`) oluşturulması.
- [x] Modelin hazırlanan veri seti ile eğitilmesi.
- [x] Eğitim sürecinin takip edilmesi ve en iyi modelin kaydedilmesi.

## Adım 4: API Entegrasyonu ve Test

- [x] `main.py` dosyasının, yeni fine-tune edilmiş LLaMA modelini kullanacak şekilde güncellenmesi.
- [x] API'nin yeni modelle doğru çalıştığından emin olmak için testlerin yapılması.
- [ ] (Opsiyonel) Çok dilli desteğin yeni modele entegre edilmesi.

## Gelecek İyileştirmeler

- [ ] Çocuk hikayesi olduğu için şiddet, cinsellik vb. +18 unsurlara dikkat etmeliyiz hikaye oluştururken.