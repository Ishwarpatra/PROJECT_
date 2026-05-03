import pickle

# 1. Define the simple logic (The "Model")
# In a real scenario, this would be a Scikit-Learn model object.
# For teaching, we use a simple class to show how objects are pickled.
class SalaryModel:
    def __init__(self):
        self.base = 30000
        self.exp_coeff = 5000
        self.skill_coeff = 2000

    def predict(self, years, skill):
        return self.base + (years * self.exp_coeff) + (skill * self.skill_coeff)

# 2. Instantiate the model
model = SalaryModel()

# 3. Save the model as a Pickle file
# 'wb' stands for Write Binary
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl")