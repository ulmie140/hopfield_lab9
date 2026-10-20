from flask import Flask, render_template, request, jsonify
from src.domain.models import get_default_patterns
from src.services.hopfield import HopfieldNetwork

app = Flask(__name__)

# Загружаем 3 эталонных образа и обучаем сеть при старте
patterns = get_default_patterns()
hopfield_net = HopfieldNetwork(patterns)

@app.route("/")
def index():
    """Главная страница приложения."""
    return render_template("index.html")

@app.route("/api/init", methods=["GET"])
def init_data():
    """Отдает эталонные образы и параметры сети."""
    return jsonify({
        "patterns": [
            {"id": p.id, "name": p.name, "width": p.width, "height": p.height, "vector": p.vector}
            for p in patterns
        ],
        "neurons": hopfield_net.n,
        "capacity": hopfield_net.theoretical_capacity()
    })

@app.route("/api/noise", methods=["POST"])
def apply_noise():
    """Генерирует зашумленный образ."""
    data = request.json
    vector = data.get("vector", [])
    noise = float(data.get("noise", 20))
    noisy_vector = hopfield_net.add_noise(vector, noise)
    return jsonify({"noisy_vector": noisy_vector})

@app.route("/api/recognize", methods=["POST"])
def recognize():
    """Запускает процесс распознавания искаженного образа."""
    data = request.json
    vector = data.get("vector", [])
    mode = data.get("mode", "async")
    result = hopfield_net.recognize(vector, mode=mode)
    return jsonify(result)

@app.route("/api/experiment", methods=["POST"])
def experiment():
    """Запускает полное исследование устойчивости сети к шуму (12 уровней по 10 тестов)."""
    data = request.json or {}
    mode = data.get("mode", "async")
    trials = int(data.get("trials", 10))
    report = hopfield_net.run_noise_experiment(trials_per_level=trials, mode=mode)
    return jsonify(report)