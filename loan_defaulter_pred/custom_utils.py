
# Handle categorical columns
home_ownership_mapping = {'MORTGAGE': 0, 'RENT': 1, 'OWN': 2, 'OTHER': 3}
loan_grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
default_on_file_mapping = {'N': 0, 'Y': 1}

mlflow_tracking_uri = "https://stunning-fortnight-wj5j4q5w5543gj6q-5000.app.github.dev/"

registered_model_name = "loan-defaulter-pred-model"
