# 1. Initialization (Replace 'model' with your actual model object)
# Example: if your model is named 'tea_model', write model=tea_model
visualizer = jd_tsne_cl(model=model) 

# 2. Feature Extraction (Replace 'test_loader' with your actual data loader)
visualizer.extract_and_compute(test_loader)

# 3. Generate Generic Class Names (Class 1, Class 2, ...)
# This automatically counts your classes and generates names
num_of_classes = len(np.unique(visualizer.labels)) 
my_generic_names = [f"Class {i+1}" for i in range(num_of_classes)]

# 4. Final Plotting
visualizer.plot(class_names=my_generic_names, modes=['all'], exclude=[])