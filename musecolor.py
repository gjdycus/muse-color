import model
import save_model

loaded_model = model.create_model()
loaded_model.load_weights(save_model.ckpt_path)
[_, score] = loaded_model.evaluate(model.x_test, model.y_test, batch_size=64)
print("Best score: " + str(score))
