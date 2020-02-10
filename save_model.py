import model

ckpt_path = "./models/v1"

def save_best():
    best = model.best_model(5)
    best.save_weights(ckpt_path)