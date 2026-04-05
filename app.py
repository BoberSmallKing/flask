import os
from flask import Flask, render_template, request, jsonify

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

app = Flask(__name__)

# 🔥 Загрузка модели ОДИН РАЗ при старте
print("Загрузка модели...")

embedding_model = SentenceTransformer('ai-forever/sbert_large_nlu_ru')
topic_model = BERTopic.load("voice_bot_model_v3", embedding_model=embedding_model)

topic_labels_dict = topic_model.topic_labels_

print("Модель загружена!")

# ✅ ЧЕЛОВЕЧЕСКИЕ НАЗВАНИЯ (настрой под себя)
HUMAN_LABELS = {
    0: "Проблема соплатой",
    1: "Получение посылки",
    2: "Ячейка не открывается",
    3: "Связь с оператором",
    4: "Ошибка постамата",
    5: "Нет кода получения",
}

# 🔁 fallback по ключевым словам (если ID не найден)
KEYWORD_LABELS = {
    "оператор": "Связь с оператором",
    "алло": "Связь с оператором",
    "позвон": "Связь с оператором",
    "оплат": "Проблема с оплатой",
    "карта": "Проблема с оплатой",
    "ячейк": "Ячейка не открывается",
    "не откры": "Ячейка не открывается",
    "код": "Нет кода получения",
}

# 🧠 Функция классификации
def predict_category(user_message):
    topics, probs = topic_model.transform([user_message])
    
    topic_id = topics[0]

    # ❌ шум
    if topic_id == -1:
        return "Неизвестный запрос", 0.0

    # 1️⃣ пробуем по ID
    if topic_id in HUMAN_LABELS:
        category = HUMAN_LABELS[topic_id]
    else:
        # 2️⃣ fallback — берем сырой label
        raw_label = topic_labels_dict.get(topic_id, "")

        # 3️⃣ пробуем по ключевым словам
        category = None
        for key, value in KEYWORD_LABELS.items():
            if key in raw_label.lower():
                category = value
                break

        # 4️⃣ если ничего не нашли
        if not category:
            category = raw_label.replace("_", " ").capitalize()

    prob = float(probs[0][topic_id])

    return category, prob


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/chat")
def chat_page():
    return render_template('chat.html')


@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.json
    user_text = data.get("message")

    if not user_text:
        return jsonify({"status": "error", "message": "Пустое сообщение"})

    category, confidence = predict_category(user_text)

    return jsonify({
        "status": "ok",
        "message": user_text,
        "category": category,
        "confidence": round(confidence * 100, 1)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)