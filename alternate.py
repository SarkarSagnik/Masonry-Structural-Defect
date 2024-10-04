# Load and preprocess the training data
train_generator = datagen.flow_from_directory(
    'Training',
    target_size=(224, 224),
    batch_size=5,
    class_mode='categorical',  # Multi-class classification
    subset='training'
)

# Load and preprocess the validation data
validation_generator = datagen.flow_from_directory(
    'Validation',
    target_size=(224, 224),
    batch_size=5,
    class_mode='categorical',
    subset='validation'
)

# Calculate class weights
classes = train_generator.classes  # Get the class labels
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weights_dict = dict(enumerate(class_weights))


layers.Dense(4, activation='softmax')  # Adjust for your number of classes



# Step 3: Compile the Model
optimizer = Adam(learning_rate=0.0001)  # Adjust learning rate for better performance
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    class_weight=class_weights_dict  # Add class weights here
)


# Evaluate on the test set
test_generator = datagen.flow_from_directory(
    'Validation',
    target_size=(224, 224),
    batch_size=5,
    class_mode='categorical'
)

# Evaluate the model
model.evaluate(test_generator)


# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Predict the labels for test data
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices)
disp.plot(cmap='Blues')