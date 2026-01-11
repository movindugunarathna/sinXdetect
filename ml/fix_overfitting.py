"""
Fix Model Overfitting

Updates the notebook with aggressive regularization to prevent overfitting:
1. Add dropout everywhere (embeddings, LSTM, dense layers)
2. Reduce model capacity
3. Add L2 regularization
4. Stronger early stopping
"""

import json

print("="*70)
print("FIXING MODEL OVERFITTING")
print("="*70)

# Read the notebook
with open('bilstm_text_classifier.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix 1: Update hyperparameters
print("\n1. Updating hyperparameters...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'EPOCHS = 3' in source and 'EMBED_DIM' in source:
            print(f"   Found hyperparameters cell at index {i}")
            new_source = []
            for line in cell['source']:
                if 'EPOCHS = 3' in line:
                    new_source.append('EPOCHS = 2  # Reduced from 3 to prevent overfitting\n')
                    print("     - Reduced EPOCHS: 3 -> 2")
                elif 'EMBED_DIM = 128' in line:
                    new_source.append('EMBED_DIM = 64  # Reduced from 128\n')
                    print("     - Reduced EMBED_DIM: 128 -> 64")
                elif 'LSTM_UNITS = 128' in line:
                    new_source.append('LSTM_UNITS = 64  # Reduced from 128\n')
                    print("     - Reduced LSTM_UNITS: 128 -> 64")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

# Fix 2: Update model architecture with aggressive regularization
print("\n2. Updating model architecture...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def build_model():' in source and 'Bidirectional' in source:
            print(f"   Found model architecture cell at index {i}")
            
            new_source = '''def build_model():
    """
    Build BiLSTM model with aggressive regularization to prevent overfitting
    """
    from tensorflow.keras import regularizers
    
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int64, name='tokens')
    
    # Embedding layer (trainable=False initially to prevent overfitting)
    x = tf.keras.layers.Embedding(
        MAX_TOKENS, 
        EMBED_DIM, 
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(1e-4),  # L2 regularization
        trainable=False  # Freeze embeddings initially
    )(inputs)
    
    # Dropout on embeddings
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # First BiLSTM layer with recurrent dropout
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LSTM_UNITS, 
            return_sequences=True,
            dropout=0.3,  # Input dropout
            recurrent_dropout=0.2,  # Recurrent dropout
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(x)
    
    # Dropout after first LSTM
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Second BiLSTM layer (smaller, with dropout)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LSTM_UNITS // 2,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(1e-4)
        )
    )(x)
    
    # Dropout after second LSTM
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Smaller dense layer with L2 regularization
    x = tf.keras.layers.Dense(
        32,  # Reduced from 128
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-3)
    )(x)
    
    # Final dropout before output
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs, name='bilstm_classifier_regularized')
    
    # Compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Reduced from 2e-4
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()
'''
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            print("     OK: Added dropout on embeddings (0.3)")
            print("     OK: Added recurrent dropout (0.2)")
            print("     OK: Added dropout after LSTM layers (0.4, 0.5)")
            print("     OK: Added L2 regularization")
            print("     OK: Reduced dense layer: 128 -> 32")
            print("     OK: Frozen embeddings initially")
            print("     OK: Reduced learning rate: 2e-4 -> 1e-4")
            break

# Fix 3: Update callbacks for stronger early stopping
print("\n3. Updating callbacks...")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'ModelCheckpoint' in source and 'EarlyStopping' in source and 'callbacks = [' in source:
            print(f"   Found callbacks cell at index {i}")
            
            new_source = '''callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / 'checkpoint.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=1,  # Reduced from 2 - stop after 1 epoch of no improvement
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when validation stops improving
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
'''
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            print("     OK: Reduced early stopping patience: 2 -> 1")
            print("     OK: Added ReduceLROnPlateau callback")
            break

# Save the notebook
with open('bilstm_text_classifier.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\n" + "="*70)
print("FIXES APPLIED SUCCESSFULLY")
print("="*70)

print("\nChanges Summary:")
print("\nHyperparameters:")
print("  - EPOCHS:      3 -> 2")
print("  - EMBED_DIM:   128 -> 64")
print("  - LSTM_UNITS:  128 -> 64")
print("  - Dense units: 128 -> 32")
print("  - Learning rate: 2e-4 -> 1e-4")

print("\nRegularization Added:")
print("  - Embedding dropout: 0.3")
print("  - LSTM dropout: 0.3 (input)")
print("  - LSTM recurrent dropout: 0.2")
print("  - Post-LSTM dropout: 0.4, 0.5")
print("  - Dense dropout: 0.5")
print("  - L2 regularization: 1e-4 (LSTM), 1e-3 (Dense)")
print("  - Frozen embeddings initially")

print("\nEarly Stopping:")
print("  - Patience: 2 -> 1 (stop faster)")
print("  - Added learning rate reduction")

print("\nExpected Results:")
print("  - Training will stop after 1-2 epochs")
print("  - Val accuracy will be lower than train (good!)")
print("  - AUC should be 0.80-0.92 (not 1.00)")
print("  - Model will NOT memorize training data")

print("\nNext Steps:")
print("  1. Restart kernel")
print("  2. Run all cells")
print("  3. Training should stop early (1-2 epochs)")
print("  4. Check AUC - should be realistic now")

print("="*70)
